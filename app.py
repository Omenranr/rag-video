#!/usr/bin/env python3
"""
Gradio UI for video → (transcript + detections) → hybrid search → chat (multi-LLM)
+ Providers: IBM watsonx.ai • OpenAI • Anthropic
+ Optional Web Search with Exa
+ Clickable timecodes to seek the video

NEW (2025-09-07):
- Answers now use **RAG + chat history**.
- We (1) rewrite the latest user message into a **standalone question** using recent chat history,
  (2) retrieve RAG context with that question,
  (3) route to web search if needed,
  (4) answer with the **conversation window + RAG + (optional) web snippets**.

Flow:
1) Contextualize the latest user message with chat history → standalone question.
2) Retrieve transcript context (hybrid + rerank + ±N neighbors) with that standalone question.
3) Ask the chosen LLM to decide if web search is needed:
   - If NO: LLM returns final answer (uses transcript context + chat history).
   - If YES: LLM returns an EXACT search query string.
4) If search needed: call Exa API → fetch results → second LLM call
   to answer using transcript context + web snippets + chat history.

Also:
- Reuse existing outputs per video (outputs/<video_stem>_timestamp).
- Clickable timecodes list (radio) that seeks the video player.

Run:
  pip install gradio==4.* pandas numpy tqdm rank-bm25 sentence-transformers requests python-dotenv
  python app_multillm.py

Notes:
- Choose your LLM provider in the UI. Only the credentials for the selected provider are used.
- For OpenAI, you may override the Base URL to use compatible endpoints; leave blank for api.openai.com.
- For Anthropic, the standard Messages API endpoint is used.
"""

from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import gradio as gr
import requests

# ---------------------
# Helpers
# ---------------------
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}


def srt_to_seconds(s: str) -> int:
    # "HH:MM:SS,mmm" → seconds (floor)
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", s.strip())
    if not m:
        return 0
    hh, mm, ss, _ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss


def list_videos(folder: str) -> List[str]:
    p = Path(folder).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    hits = []
    for ext in VIDEO_EXTS:
        hits.extend(str(x) for x in p.rglob(f"*{ext}"))
    hits.sort()
    return hits


def safe_stem(path: Path) -> str:
    s = path.stem.strip()
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s)
    return s or "video"


def find_existing_outputs_for_video(video_path: str, outputs_root: str) -> List[str]:
    """
    Look under outputs_root for subfolders starting with safe_stem(video) + '_'
    and containing transcript.srt.
    """
    root = Path(outputs_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    stem = safe_stem(Path(video_path))
    pattern = f"{stem}_*"
    matches = [p for p in root.glob(pattern) if p.is_dir()]
    matches = [p for p in matches if (p / "transcript.srt").exists()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in matches]


def latest_rag_index_dir(outputs_dir: str) -> Optional[str]:
    base = Path(outputs_dir)
    idx_dirs = [p for p in base.glob("rag_index_*") if p.is_dir()]
    if not idx_dirs:
        return None
    idx_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(idx_dirs[0])

# ---------------------
# Extractor / Indexer (lazy imports)
# ---------------------

def run_extractor(video_path: str, lang: str, vclass: str, fps: Optional[int],
                  device: str, batch_size: int, conf: float, iou: float, max_det: int,
                  progress=None) -> str:
    """
    Returns output directory containing: transcript.srt, detections.csv, detections_by_frame.csv
    Lazy-imports heavy code; falls back to subprocess if not importable.
    """
    local_ok = False
    try:
        from video_audio_multitool import run_pipeline as extractor_run_pipeline  # type: ignore
        local_ok = True
    except Exception:
        local_ok = False

    if local_ok:
        if progress:
            progress(0.05, desc="Starting extractor…")
        out_dir = extractor_run_pipeline(
            input_arg=video_path, lang=lang, vclass=vclass, fps=fps,
            device_choice=device, batch_size=batch_size,
            det_conf=conf, det_iou=iou, det_max=max_det
        )
        return str(out_dir)
    else:
        import subprocess
        cmd = [
            sys.executable, "video_audio_multitool.py", video_path,
            "--lang", lang, "--vclass", vclass, "--device", device,
            "--batch-size", str(batch_size), "--det-conf", str(conf),
            "--det-iou", str(iou), "--max-det", str(max_det)
        ]
        if fps:
            cmd += ["--fps", str(fps)]
        if progress:
            progress(0.05, desc="Starting extractor (subprocess)…")
        subprocess.run(cmd, check=True)
        outs = sorted((Path.cwd() / "outputs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not outs:
            raise RuntimeError("Extractor finished but no outputs/ folder found.")
        return str(outs[0])


def build_index_from_srt(transcript_path: str, window: int, anchor: str,
                         embed_model: str, device: Optional[str], progress=None) -> str:
    """
    Returns path to built index folder (rag_index_*)
    Lazy-imports heavy code; falls back to subprocess if not importable.
    """
    srt = Path(transcript_path)
    if not srt.exists():
        raise FileNotFoundError(f"Transcript not found: {srt}")

    local_ok = False
    try:
        from transcript_hybrid_rag import TranscriptRAGIndex  # type: ignore
        local_ok = True
    except Exception:
        local_ok = False

    if local_ok:
        if progress:
            progress(0.65, desc="Building hybrid index…")
        idx = TranscriptRAGIndex.build_from_srt(
            srt_path=srt, outdir=None, window_sec=window,
            model_name=embed_model, device=device, anchor=anchor
        )
        return str(idx.root)
    else:
        import subprocess
        if progress:
            progress(0.65, desc="Building hybrid index (subprocess)…")
        subprocess.run([
            sys.executable, "transcript_hybrid_rag.py", "build", str(srt),
            "--window", str(window), "--anchor", anchor, "--model", embed_model
        ], check=True)
        ridx = sorted(srt.parent.glob("rag_index_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ridx:
            raise RuntimeError("Index build finished but no rag_index_* folder was created.")
        return str(ridx[0])


# Cache of loaded indexes: {index_dir: TranscriptRAGIndex}
_INDEX_CACHE: Dict[str, object] = {}


def load_index(index_dir: str):
    key = str(Path(index_dir).resolve())
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    try:
        from transcript_hybrid_rag import TranscriptRAGIndex  # type: ignore
        idx = TranscriptRAGIndex.load(Path(key))
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location("transcript_hybrid_rag", Path("transcript_hybrid_rag.py"))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        idx = mod.TranscriptRAGIndex.load(Path(key))  # type: ignore
    _INDEX_CACHE[key] = idx
    return idx


def compile_context_blocks(idx, query: str, top_k: int, method: str, alpha: float,
                           rerank: str, rerank_model: str, overfetch: int,
                           ctx_before: int, ctx_after: int,
                           device: Optional[str], embed_model_override: Optional[str]) -> Tuple[List[Dict], str]:
    if method in ("bm25", "embed"):
        base_hits = (idx.query_bm25(query, k=max(top_k, overfetch))
                     if method == "bm25"
                     else idx.query_embed(query, k=max(top_k, overfetch),
                                          model_name=embed_model_override, device=device))
    else:
        base_hits = idx.query_hybrid(query, k=top_k, method=method, alpha=alpha,
                                     overfetch=overfetch, model_name=embed_model_override, device=device)

    if rerank == "cross":
        hits = idx.rerank_cross(query, base_hits, model_name=rerank_model, device=device, top_k=top_k)
    elif rerank == "mmr":
        hits = idx.rerank_mmr(query, base_hits, top_k=top_k, lambda_mult=0.7,
                              model_name=embed_model_override, device=device)
    else:
        hits = base_hits[:top_k]

    enriched = idx.attach_context(hits, before=ctx_before, after=ctx_after)

    blocks = []
    for h in enriched:
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        lines = [f"[{c['start_srt']}–{c['end_srt']}] {c['text']}".strip() for c in ctx_sorted]
        block = f"### Passage {h['rank']} | Main: {h['start_srt']}–{h['end_srt']}\n" + "\n".join(lines)
        blocks.append(block)
    full_context = "\n\n".join(blocks)
    return enriched, full_context

# ---------------------
# LLM clients (watsonx.ai, OpenAI, Anthropic)
# ---------------------

def watsonx_chat(
    base_url: str,
    api_key: str,
    model: str,
    project_id: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
    version: str = "2024-10-10",  # watsonx.ai API version
) -> str:
    """
    Call watsonx.ai chat endpoint with OpenAI-like inputs.
    Returns: Assistant message text.
    """
    # 1) Exchange API key -> IAM Bearer token
    iam_resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        timeout=timeout,
    )
    iam_resp.raise_for_status()
    iam_token = iam_resp.json()["access_token"]

    # 2) Convert messages to watsonx chat schema
    def _wx_content(role: str, content):
        if isinstance(content, list):
            return content
        if role == "user":
            return [{"type": "text", "text": str(content)}]
        if role == "system":
            return str(content)
        if role == "assistant":
            return str(content)
        return [{"type": "text", "text": str(content)}]

    wx_messages = [
        {"role": m.get("role", "user"), "content": _wx_content(m.get("role", "user"), m.get("content", ""))}
        for m in messages
    ]

    # 3) Call watsonx Chat API
    url = f"{base_url.rstrip('/')}/ml/v1/text/chat?version={version}"
    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model_id": model,
        "project_id": project_id,
        "messages": wx_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"watsonx.ai error {resp.status_code}: {resp.text}") from e

    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Bad LLM response: {data}")
    return data["choices"][0]["message"]["content"]


def openai_chat(
    api_key: str,
    model: str,
    messages: List[Dict],
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    """Call OpenAI-compatible Chat Completions API and return assistant text."""
    root = (base_url or "https://api.openai.com").rstrip("/")
    url = f"{root}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}") from e
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _anthropic_convert_messages(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Split out system content and convert messages to Anthropic Messages API format."""
    system_parts: List[str] = []
    converted: List[Dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Collect system content separately
        if role == "system":
            if isinstance(content, list):
                # concatenate any text parts
                txt = "\n\n".join(
                    [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                )
                system_parts.append(txt)
            else:
                system_parts.append(str(content))
            continue

        def to_text_list(c):
            if isinstance(c, list):
                # already a list of content blocks
                blocks = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        blocks.append(p)
                    else:
                        blocks.append({"type": "text", "text": str(p)})
                return blocks
            return [{"type": "text", "text": str(c)}]

        if role in ("user", "assistant"):
            converted.append({"role": role, "content": to_text_list(content)})
        else:
            # default to user
            converted.append({"role": "user", "content": to_text_list(content)})

    system_str = "\n\n".join([s for s in system_parts if s])
    return system_str, converted


def anthropic_chat(
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    """Call Anthropic Messages API and return assistant text."""
    system_str, conv = _anthropic_convert_messages(messages)
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: Dict = {
        "model": model,
        "messages": conv,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_str:
        payload["system"] = system_str

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Anthropic error {resp.status_code}: {resp.text}") from e
    data = resp.json()
    # content is a list of blocks; return concatenated text
    blocks = data.get("content", [])
    text = "\n".join([b.get("text", "") for b in blocks if isinstance(b, dict)])
    return text


# Unified chat wrapper

def llm_chat(
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    provider = (provider or "watsonx").lower()
    if provider == "watsonx":
        return watsonx_chat(
            base_url=cfg.get("wx_base_url", ""),
            api_key=cfg.get("wx_api_key", ""),
            model=cfg.get("wx_model", ""),
            project_id=cfg.get("wx_project_id", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "openai":
        return openai_chat(
            api_key=cfg.get("oa_api_key", ""),
            model=cfg.get("oa_model", ""),
            messages=messages,
            base_url=cfg.get("oa_base_url") or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "anthropic":
        return anthropic_chat(
            api_key=cfg.get("an_api_key", ""),
            model=cfg.get("an_model", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------
# Exa web search
# ---------------------

def exa_search_with_contents(query: str, api_key: str, num_results: int = 5, timeout: int = 60) -> List[Dict]:
    """
    Returns list of {title, url, snippet} using Exa /search + /contents.
    """
    base = "https://api.exa.ai"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 1) Search
    sr = requests.post(
        f"{base}/search",
        headers=headers,
        json={
            "query": query,
            "numResults": int(num_results),
            "type": "neural",
            "useAutoprompt": False,
        },
        timeout=timeout,
    )
    sr.raise_for_status()
    sdata = sr.json()
    results = sdata.get("results", []) or []

    if not results:
        return []

    # 2) Contents for each result (if not already present)
    ids = [r.get("id") for r in results if r.get("id")]
    content_by_id: Dict[str, str] = {}
    if ids:
        cr = requests.post(
            f"{base}/contents",
            headers=headers,
            json={"ids": ids},
            timeout=timeout,
        )
        if cr.ok:
            cdata = cr.json()
            for item in cdata.get("results", []):
                content_by_id[item.get("id", "")] = (item.get("text") or "").strip()

    out: List[Dict] = []
    for r in results:
        rid = r.get("id", "")
        title = r.get("title") or r.get("url") or "(untitled)"
        url = r.get("url") or ""
        text = (r.get("text") or "").strip()
        text = text or content_by_id.get(rid, "")
        if len(text) > 800:
            text = text[:790].rstrip() + "…"
        out.append({"title": title, "url": url, "snippet": text})
    return out


# ---------------------
# History helpers (NEW)
# ---------------------

def _format_history_as_text(history_msgs: List[Dict], max_turns: int = 8, max_chars: int = 6000) -> str:
    """
    Convert recent chat history into a compact text transcript for prompting.
    Keeps the last `max_turns` messages (user+assistant counts as 2).
    Also enforces an approximate char cap.
    """
    if not history_msgs:
        return ""
    # keep last N
    recent = history_msgs[-max_turns:]
    lines: List[str] = []
    total = 0
    for m in recent:
        role = m.get("role", "user")
        content = str(m.get("content", "")).strip()
        pref = "User" if role == "user" else "Assistant"
        line = f"{pref}: {content}"
        total += len(line)
        lines.append(line)
        if total >= max_chars:
            break
    return "\n".join(lines)


def _trim_history_messages(history_msgs: List[Dict], max_turns: int = 8, max_chars: int = 6000) -> List[Dict]:
    """
    Return recent history as a list of chat messages for the model,
    limited by turns and approx characters.
    """
    if not history_msgs:
        return []
    recent = history_msgs[-max_turns:]
    out: List[Dict] = []
    total = 0
    for m in recent:
        content = str(m.get("content", "")).strip()
        line_len = len(content)
        if total + line_len > max_chars:
            # If adding would exceed cap, try to add a truncated version
            content = content[: max(0, max_chars - total)]
        out.append({"role": m.get("role", "user"), "content": content})
        total += len(content)
        if total >= max_chars:
            break
    return out


def contextualize_question(
    provider: str,
    cfg: Dict,
    history_msgs: List[Dict],
    latest_user_msg: str,
) -> str:
    """
    Use an LLM to rewrite the latest user message into a standalone question,
    leveraging the recent chat history.
    """
    history_text = _format_history_as_text(history_msgs, max_turns=10, max_chars=6000)
    system = (
        "You rewrite user questions into a single, clear standalone question.\n"
        "Use the conversation so far to resolve pronouns, references, and ellipses.\n"
        "Output ONLY the rewritten question, with no commentary or quotes."
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message:\n{latest_user_msg}\n\n"
        f"Rewritten standalone question:"
    )
    try:
        out = llm_chat(
            provider=provider,
            cfg=cfg,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
        )
        # sanitize trivial outputs
        cleaned = (out or "").strip()
        # avoid empty or overly short fallback
        return cleaned if len(cleaned) >= 2 else str(latest_user_msg)
    except Exception:
        return str(latest_user_msg)


# ---------------------
# Search decision helper (history-aware, NEW)
# ---------------------

def wants_web_search_explicit(user_msg: str) -> bool:
    """
    Heuristic to flag explicit web search intent (English + French).
    """
    s = (user_msg or "").lower()
    patterns = [
        r"\b(search|google|web|internet|online|look\s*up|check\s*online|find on the web)\b",
        r"cherche( r)? sur (le|la|les)?\s*web|internet",
        r"recherche en ligne",
        r"regarde sur internet",
        r"sur le web",
    ]
    return any(re.search(p, s) for p in patterns)


def extract_json(text: str) -> Optional[Dict]:
    """
    Try to parse a JSON object from model output (raw or fenced).
    """
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    blob = None
    if m:
        blob = m.group(1)
    else:
        # greedy but okay since we ask the model to return JSON ONLY
        m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if m2:
            blob = m2.group(1)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


def llm_decide_search(
    provider: str,
    cfg: Dict,
    question: str,
    transcript_context: str,
    explicit_flag: bool,
    history_text: str = "",
) -> Dict:
    """
    Ask the selected LLM to decide whether to search the web, with awareness of chat history.
    Returns dict with keys:
      need_search: bool
      query: str | None
      answer: str | None
      reason: str
    """
    system = (
        "You are a retrieval QA router.\n"
        "Inputs: (a) the conversation so far, (b) a standalone user question, and (c) transcript context.\n"
        "If the user **explicitly** asks to search the web OR the answer is not found in the transcript context, "
        "you MUST request a web search by returning JSON ONLY.\n"
        "Otherwise, answer using ONLY the transcript context and return JSON ONLY.\n"
        "JSON schema:\n"
        '{"need_search": true|false, "query": string|null, "answer": string|null, "reason": string}\n'
        "Rules:\n"
        "- If need_search=true: 'query' MUST be a single, precise web search string; 'answer' MUST be null.\n"
        "- If need_search=false: 'answer' MUST contain the final answer grounded in transcript context; 'query' MUST be null.\n"
        "- Never include explanations outside the JSON."
    )
    user = (
        f"Explicit_web_search_flag={explicit_flag}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Standalone Question:\n{question}\n\n"
        f"Transcript context:\n{transcript_context}\n"
    )
    out = llm_chat(
        provider=provider,
        cfg=cfg,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=500,
    )
    data = extract_json(out) or {}
    need = bool(data.get("need_search"))
    query = (data.get("query") or "").strip() if need else None
    answer = (data.get("answer") or "").strip() if not need else None
    reason = (data.get("reason") or "").strip()
    return {"need_search": need, "query": query, "answer": answer, "reason": reason}


# ---------------------
# Gradio callbacks
# ---------------------

def do_scan(folder):
    vids = list_videos(folder)
    dd_update = gr.update(choices=vids, value=(vids[0] if vids else None))
    vid_update = gr.update(value=None, visible=bool(vids))
    return dd_update, vid_update


def _on_select(video, outputs_root):
    vp_update = gr.update(value=video, visible=True)
    matches = find_existing_outputs_for_video(video, outputs_root)
    dd_update = gr.update(choices=matches, value=(matches[0] if matches else None))
    msg = "Found existing outputs:\n" + ("\n".join(matches[:8]) if matches else "None")
    return vp_update, dd_update, msg


def on_use_existing(selected_outputs, video_path, state_dict,
                    window, anchor, embed_model, embed_device):
    if not selected_outputs:
        raise gr.Error("No outputs folder selected.")
    out_dir = Path(selected_outputs).resolve()
    if not out_dir.exists():
        raise gr.Error(f"Selected outputs folder doesn't exist: {out_dir}")

    idx_dir = latest_rag_index_dir(str(out_dir))
    built = False
    if idx_dir is None:
        idx_dir = build_index_from_srt(
            transcript_path=str(out_dir / "transcript.srt"),
            window=int(window), anchor=anchor,
            embed_model=embed_model,
            device=(None if embed_device == "auto" else embed_device),
            progress=None
        )
        built = True

    _ = load_index(idx_dir)

    sd = state_dict or {}
    sd[str(video_path)] = {"out_dir": str(out_dir), "index_dir": str(idx_dir)}

    msg = (
        f"Using existing outputs:\n"
        f"- outputs: {out_dir}\n"
        f"- index:   {idx_dir} {'(created now)' if built else '(existing)'}\n"
        f"- transcript: {out_dir / 'transcript.srt'}\n"
    )
    return msg, sd


def do_generate(folder, video_path, outputs_root,
                lang, vclass, fps, device, batch, conf, iou, max_det,
                window, anchor, embed_model, embed_device, state_dict, progress=gr.Progress(track_tqdm=True)):
    if not video_path:
        raise gr.Error("Please select a video.")
    progress(0.0, desc="Starting…")

    out_dir = run_extractor(
        video_path=video_path, lang=lang, vclass=vclass, fps=(None if not fps else int(fps)),
        device=device, batch_size=int(batch), conf=float(conf), iou=float(iou), max_det=int(max_det),
        progress=progress
    )
    progress(0.55, desc="Extraction completed.")

    srt_path = str(Path(out_dir) / "transcript.srt")
    idx_dir = build_index_from_srt(
        transcript_path=srt_path, window=int(window), anchor=anchor,
        embed_model=embed_model, device=(None if embed_device == "auto" else embed_device),
        progress=progress
    )
    progress(0.9, desc="Index ready.")

    _ = load_index(idx_dir)

    sd = state_dict or {}
    sd[str(video_path)] = {"out_dir": out_dir, "index_dir": idx_dir}
    progress(1.0, desc="Done.")

    status = (
        f"✅ Generated:\n"
        f"- outputs: {out_dir}\n"
        f"- index:   {idx_dir}\n"
        f"- transcript: {srt_path}\n"
    )
    return status, sd


def _provider_cfg(provider,
                  wx_base_url, wx_api_key, wx_model, wx_project_id,
                  oa_base_url, oa_api_key, oa_model,
                  an_api_key, an_model) -> Dict:
    p = (provider or "watsonx").lower()
    if p == "watsonx":
        return {
            "wx_base_url": (wx_base_url or "").strip(),
            "wx_api_key": (wx_api_key or "").strip(),
            "wx_model": (wx_model or "").strip(),
            "wx_project_id": (wx_project_id or "").strip(),
        }
    elif p == "openai":
        return {
            "oa_base_url": (oa_base_url or "").strip(),
            "oa_api_key": (oa_api_key or "").strip(),
            "oa_model": (oa_model or "").strip(),
        }
    elif p == "anthropic":
        return {
            "an_api_key": (an_api_key or "").strip(),
            "an_model": (an_model or "").strip(),
        }
    else:
        return {}


def _validate_provider_inputs(provider: str, cfg: Dict) -> Optional[str]:
    p = (provider or "watsonx").lower()
    if p == "watsonx":
        if not cfg.get("wx_base_url") or not cfg.get("wx_api_key") or not cfg.get("wx_model") or not cfg.get("wx_project_id"):
            return "Please set watsonx Base URL, API key, Model, and Project ID in the panel."
    elif p == "openai":
        if not cfg.get("oa_api_key") or not cfg.get("oa_model"):
            return "Please set OpenAI API Key and Model in the panel (Base URL optional)."
    elif p == "anthropic":
        if not cfg.get("an_api_key") or not cfg.get("an_model"):
            return "Please set Anthropic API Key and Model in the panel."
    return None


def on_chat(
    user_msg, history,
    video_path, ctx_before, ctx_after, top_k, method, alpha,
    rerank, rerank_model, overfetch,
    provider,  # NEW
    wx_base_url, wx_api_key, wx_model, wx_project_id,
    oa_base_url, oa_api_key, oa_model,
    an_api_key, an_model,
    embed_device, embed_model_override,
    enable_web, exa_api_key, exa_num_results,
    state_dict
):
    """
    Generator that yields (messages, ts_radio_update, ts_map_state).
    Chatbot(type='messages'): messages must be [{'role','content'}, ...]
    Now **history-aware**: we use chat history to contextualize the query and in the final answer.
    """
    # The chat as displayed to the user
    msgs = list(history or [])

    # basic validations
    if not video_path:
        msgs.append({"role": "assistant", "content": "Please select a video first."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}
        return

    cfg = _provider_cfg(provider, wx_base_url, wx_api_key, wx_model, wx_project_id,
                        oa_base_url, oa_api_key, oa_model,
                        an_api_key, an_model)
    err = _validate_provider_inputs(provider, cfg)
    if err:
        msgs.append({"role": "assistant", "content": err})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}
        return

    rec = (state_dict or {}).get(str(video_path))
    if not rec:
        msgs.append({"role": "assistant", "content": "No outputs mapped for this video. Click 'Use selected outputs' or 'Generate'."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}
        return

    # Append the user's latest message to the visible chat
    latest_user = str(user_msg)
    msgs.append({"role": "user", "content": latest_user})

    # Prepare a trimmed history window (EXCLUDES the latest user msg for clarity in prompts)
    history_window_for_rewrite = _trim_history_messages(history or [], max_turns=10, max_chars=6000)
    history_text_for_router = _format_history_as_text(history or [], max_turns=10, max_chars=6000)

    # === (1) Contextualize the question with history ===
    try:
        standalone_q = contextualize_question(
            provider=(provider or "watsonx"),
            cfg=cfg,
            history_msgs=history_window_for_rewrite,  # only prior messages
            latest_user_msg=latest_user,
        )
    except Exception:
        standalone_q = latest_user

    # === (2) Retrieval with the standalone question ===
    idx_dir = rec["index_dir"]
    idx = load_index(idx_dir)
    hits, ctx_text = compile_context_blocks(
        idx=idx, query=str(standalone_q), top_k=int(top_k), method=method, alpha=float(alpha),
        rerank=rerank, rerank_model=rerank_model, overfetch=int(overfetch),
        ctx_before=int(ctx_before), ctx_after=int(ctx_after),
        device=(None if embed_device == "auto" else embed_device),
        embed_model_override=(None if not embed_model_override else embed_model_override)
    )

    # Prepare timecode radio options now (from hits), to return with any yield
    labels = []
    label_to_start = {}
    for h in hits:
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        main = next((c for c in ctx_sorted if c.get("offset", 0) == 0), ctx_sorted[0])
        start_srt = main.get("start_srt", h.get("start_srt", "00:00:00,000"))
        end_srt = main.get("end_srt", h.get("end_srt", "00:00:00,000"))
        snippet = (main.get("text", "") or "").replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:97] + "…"
        label = f"{start_srt} – {end_srt} · {snippet}".strip()
        labels.append(label)
        label_to_start[label] = srt_to_seconds(start_srt)

    radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

    # === (3) Router: decide about web search (history-aware) ===
    explicit_flag = wants_web_search_explicit(latest_user)
    try:
        decision = llm_decide_search(
            provider=(provider or "watsonx"),
            cfg=cfg,
            question=str(standalone_q),
            transcript_context=ctx_text,
            explicit_flag=explicit_flag,
            history_text=history_text_for_router
        )
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"Routing error: {e}"})
        yield msgs, radio_update, label_to_start
        return

    need_search = bool(decision.get("need_search"))
    search_query = decision.get("query")
    direct_answer = decision.get("answer")

    # === (4A) If no web search needed, answer using transcript + history ===
    if not need_search:
        # Compose final messages with history window + a single user turn that includes context
        history_for_model = _trim_history_messages(msgs[:-1], max_turns=10, max_chars=6000)  # include all previous turns
        system = (
            "Tu es un assistant qui répond en combinant:\n"
            "1) le transcript (fiable pour les propos entendus et timestamps),\n"
            "2) l'historique de la conversation (pour le contexte et les suivis),\n"
            "Sans inventer au-delà du transcript. Indique les timecodes quand utiles."
        )
        # If the router provided an answer, we can return it directly; else ask model to compose an answer that uses ctx + history.
        if direct_answer:
            msgs.append({"role": "assistant", "content": direct_answer})
            yield msgs, radio_update, label_to_start
            return
        else:
            # Build a single 'user' message with the standalone question and the RAG context
            user_full = (
                f"Dernière question (standalone):\n{standalone_q}\n\n"
                f"Transcript context (passages):\n{ctx_text}\n"
            )
            try:
                final_answer = llm_chat(
                    provider=(provider or "watsonx"),
                    cfg=cfg,
                    messages=[{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}],
                    temperature=0.2,
                    max_tokens=900,
                )
            except Exception as e:
                final_answer = f"LLM error: {e}"

            msgs.append({"role": "assistant", "content": final_answer})
            yield msgs, radio_update, label_to_start
            return

    # === (4B) Web search suggested ===
    if not search_query:
        msgs.append({"role": "assistant", "content": "La recherche web a été suggérée, mais aucune requête n'a été fournie."})
        yield msgs, radio_update, label_to_start
        return

    if not enable_web:
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (la recherche web est désactivée)."})
        yield msgs, radio_update, label_to_start
        return

    if not exa_api_key:
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (ajoutez une clé Exa pour effectuer la recherche)."})
        yield msgs, radio_update, label_to_start
        return

    # Show interim "searching" message
    msgs.append({"role": "assistant", "content": f"🔎 Web search query: \"{search_query}\" (running Exa…)"})
    yield msgs, radio_update, label_to_start

    # 3) Exa search
    try:
        web_hits = exa_search_with_contents(search_query, exa_api_key, num_results=int(exa_num_results))
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"Exa search failed: {e}"})
        yield msgs, radio_update, label_to_start
        return

    if not web_hits:
        msgs.append({"role": "assistant", "content": "Aucun résultat web pertinent n'a été trouvé."})
        yield msgs, radio_update, label_to_start
        return

    # Build web context blob
    web_blocks = []
    for i, w in enumerate(web_hits, start=1):
        block = f"[{i}] {w['title']}  {w['url']}\n{w['snippet']}"
        web_blocks.append(block)
    web_context = "\n\n".join(web_blocks)

    # 4) Final LLM call with history + transcript + web context
    system = (
        "Tu es un assistant qui répond en combinant:\n"
        "1) l'historique de la conversation (pour le contexte et les suivis),\n"
        "2) le transcript (fiable pour les propos entendus/timestamps),\n"
        "3) des extraits web (fiables pour faits externes).\n"
        "Indique les timecodes du transcript quand utiles.\n"
        "Quand tu t'appuies sur le web, cite les sources avec leurs numéros [1], [2], etc."
    )
    user_full = (
        f"Dernière question (standalone):\n{standalone_q}\n\n"
        f"Transcript context:\n{ctx_text}\n\n"
        f"Web results:\n{web_context}"
    )

    history_for_model = _trim_history_messages(msgs[:-1], max_turns=10, max_chars=6000)
    try:
        final_answer = llm_chat(
            provider=(provider or "watsonx"),
            cfg=cfg,
            messages=[{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}],
            temperature=0.2,
            max_tokens=900,
        )
    except Exception as e:
        final_answer = f"LLM error (final): {e}"

    msgs.append({"role": "assistant", "content": final_answer})
    yield msgs, radio_update, label_to_start


# ---------------------
# Helper: toggle provider panels visibility
# ---------------------

def toggle_provider_panels(provider: str):
    p = (provider or "").lower()
    return (
        gr.update(visible=(p == "watsonx")),
        gr.update(visible=(p == "openai")),
        gr.update(visible=(p == "anthropic")),
    )

# ---------------------
# Gradio Layout
# ---------------------
with gr.Blocks(title="Video RAG Chat (Gradio + Multi-LLM + Exa)", fill_height=True) as demo:
    gr.Markdown("## 🎬 Video RAG Chat\nScan → select video → reuse **existing outputs** or **Generate** → chat with transcript.\nOptional: let the assistant trigger **web search** via Exa when needed.\n\n**LLM Providers:** IBM watsonx.ai • OpenAI • Anthropic\n\n**New:** answers use **RAG + chat history** for follow-ups and pronouns.")

    with gr.Row():
        with gr.Column(scale=3):
            folder_tb = gr.Textbox(label="Folder to scan for videos", value=str(Path.cwd() / "videos"))
            outputs_root_tb = gr.Textbox(label="Outputs folder", value=str(Path.cwd() / "outputs"))
            scan_btn = gr.Button("Scan videos")

            video_dd = gr.Dropdown(choices=[], label="Select a video")
            existing_outputs_dd = gr.Dropdown(choices=[], label="Existing outputs for this video")
            use_existing_btn = gr.Button("Use selected outputs")

            video_player = gr.Video(label="Preview", elem_id="video_preview")
            status_box = gr.Textbox(label="Status", lines=8)

            with gr.Accordion("Extraction settings (used only when you click Generate)", open=False):
                lang_dd = gr.Dropdown(choices=["EN","FR"], value="FR", label="Transcript language (Wit.ai key must exist)")
                vclass_dd = gr.Dropdown(choices=["sport","nature","screen","cctv"], value="sport", label="Detection pipeline")
                fps_tb = gr.Number(value=None, label="Override FPS (optional)")
                device_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Detector device")
                batch_tb = gr.Slider(8, 128, 64, step=8, label="Batch size")
                conf_tb = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Det conf")
                iou_tb = gr.Slider(0.1, 0.95, 0.7, step=0.05, label="NMS IoU")
                maxdet_tb = gr.Slider(50, 1000, 300, step=50, label="Max det/frame")

            with gr.Accordion("Index settings (also used if existing outputs lack an index)", open=False):
                window_tb = gr.Slider(3, 15, 5, step=1, label="Chunk window (seconds)")
                anchor_dd = gr.Dropdown(choices=["first","zero"], value="first", label="Window anchor")
                embed_model_tb = gr.Textbox(value="sentence-transformers/all-MiniLM-L6-v2", label="Embedding model")
                embed_device_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Embed device")

            with gr.Accordion("Web Search (Exa)", open=False):
                enable_web_chk = gr.Checkbox(value=False, label="Enable web search with Exa")
                exa_key_tb = gr.Textbox(label="Exa API Key", type="password")
                exa_num_tb = gr.Slider(1, 12, 5, step=1, label="Exa: number of results")

            generate_btn = gr.Button("🧪 Generate (transcript + detections + index)", variant="primary")

        with gr.Column(scale=4):
            with gr.Tab("Chat"):
                with gr.Accordion("Retrieval & Rerank", open=False):
                    ctx_before_tb = gr.Slider(0, 6, 3, step=1, label="Context: previous chunks")
                    ctx_after_tb = gr.Slider(0, 6, 3, step=1, label="Context: following chunks")
                    topk_tb = gr.Slider(1, 20, 6, step=1, label="Top-K chunks")
                    method_dd = gr.Dropdown(choices=["rrf","weighted","bm25","embed"], value="rrf", label="Base retrieval")
                    alpha_tb = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Weighted α (embed weight)")
                    rerank_dd = gr.Dropdown(choices=["none","cross","mmr"], value="none", label="Rerank")
                    rerank_model_tb = gr.Textbox(value="cross-encoder/ms-marco-MiniLM-L-6-v2", label="Cross-encoder model")
                    overfetch_tb = gr.Slider(10, 200, 50, step=10, label="Overfetch for rerank")
                    embed_model_override_tb = gr.Textbox(value="", label="Override embed model at query time (optional)")
                    embed_device_q_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Embed device (query)")

                with gr.Accordion("LLM Provider & connection", open=True):
                    provider_dd = gr.Dropdown(choices=["watsonx","openai","anthropic"], value="watsonx", label="Provider")

                    with gr.Group(visible=True) as wx_group:
                        gr.Markdown("#### watsonx.ai")
                        base_url_tb = gr.Textbox(label="watsonx.ai Base URL", placeholder="https://eu-de.ml.cloud.ibm.com")
                        api_key_tb = gr.Textbox(label="IBM Cloud API Key", type="password")
                        model_tb = gr.Textbox(label="Model ID", placeholder="ibm/granite-13b-chat-v2")
                        project_tb = gr.Textbox(label="Project ID (UUID)", placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

                    with gr.Group(visible=False) as oa_group:
                        gr.Markdown("#### OpenAI (Chat Completions)")
                        oa_base_url_tb = gr.Textbox(label="OpenAI Base URL (optional)", placeholder="https://api.openai.com")
                        oa_api_key_tb = gr.Textbox(label="OpenAI API Key", type="password")
                        oa_model_tb = gr.Textbox(label="OpenAI Model", placeholder="gpt-4o-mini or gpt-4o")

                    with gr.Group(visible=False) as an_group:
                        gr.Markdown("#### Anthropic (Messages API)")
                        an_api_key_tb = gr.Textbox(label="Anthropic API Key", type="password")
                        an_model_tb = gr.Textbox(label="Anthropic Model", placeholder="claude-3.5-sonnet")

                ts_radio = gr.Radio(choices=[], label="Timecodes (click to seek)", interactive=True, visible=False)
                ts_map_state = gr.State({})  # label -> start_seconds

                chat = gr.Chatbot(height=520, type="messages")
                chat_tb = gr.Textbox(placeholder="Ask about the video…", label="Message")
                send_btn = gr.Button("Send", variant="primary")

            state_dict = gr.State({})

    # Wiring
    scan_btn.click(fn=do_scan, inputs=[folder_tb], outputs=[video_dd, video_player])
    video_dd.change(fn=_on_select, inputs=[video_dd, outputs_root_tb], outputs=[video_player, existing_outputs_dd, status_box])

    provider_dd.change(toggle_provider_panels, inputs=[provider_dd], outputs=[wx_group, oa_group, an_group])

    use_existing_btn.click(
        fn=on_use_existing,
        inputs=[existing_outputs_dd, video_dd, state_dict, window_tb, anchor_dd, embed_model_tb, embed_device_dd],
        outputs=[status_box, state_dict]
    )

    generate_btn.click(
        fn=do_generate,
        inputs=[
            folder_tb, video_dd, outputs_root_tb,
            lang_dd, vclass_dd, fps_tb, device_dd, batch_tb, conf_tb, iou_tb, maxdet_tb,
            window_tb, anchor_dd, embed_model_tb, embed_device_dd, state_dict
        ],
        outputs=[status_box, state_dict]
    )

    # Click a timecode → seek the video (JS)
    ts_radio.change(
        inputs=[ts_radio, ts_map_state],
        outputs=[status_box],   # debug info; remove if you don't want logs
        js="""
        (label, map) => {
        const logs = [];
        logs.push(`label: ${label}`);

        const parseFromLabel = (lbl) => {
            const m = /\b(\d{2}):(\d{2}):(\d{2}),(\d{3})\b/.exec(lbl || "");
            if (!m) return NaN;
            const hh = +m[1], mm = +m[2], ss = +m[3];
            return hh*3600 + mm*60 + ss;
        };
        let seconds = (map && typeof map[label] === "number") ? map[label] : parseFromLabel(label);
        logs.push(`seconds: ${seconds}`);
        if (!Number.isFinite(seconds)) return `Bad seconds from label/map`;

        const host = document.querySelector("#video_preview");
        logs.push(`host found: ${!!host}`);

        const tryFindVideo = (root) => {
            if (!root) return null;
            let v = root.querySelector ? root.querySelector("video") : null;
            if (v) return v;
            const gv = root.querySelector ? root.querySelector("gradio-video") : null;
            if (gv && gv.shadowRoot) {
                v = gv.shadowRoot.querySelector("video");
                if (v) return v;
            }
            const all = root.querySelectorAll ? root.querySelectorAll("*") : [];
            for (const el of all) {
                if (el.shadowRoot) {
                    const vv = el.shadowRoot.querySelector("video");
                    if (vv) return vv;
                }
            }
            return null;
        };

        let video = tryFindVideo(host);
        if (!video) {
            const gvAll = Array.from(document.querySelectorAll("gradio-video"));
            for (const gv of gvAll) {
                if (gv.shadowRoot) {
                    const v2 = gv.shadowRoot.querySelector("video");
                    if (v2) { video = v2; break; }
                }
            }
        }
        logs.push(`video found: ${!!video}`);
        if (!video) return `No <video> element found`;

        const seek = () => {
            try {
                const dur = Number.isFinite(video.duration) ? video.duration : Infinity;
                const t = Math.max(0, Math.min(dur, seconds));
                video.currentTime = t;
                if (video.paused) { video.play().catch(()=>{}); }
                logs.push(`seeked to ${t}s`);
            } catch (e) {
                logs.push(`seek error: ${e}`);
            }
        };

        if (video.readyState >= 1) seek();
        else video.addEventListener("loadedmetadata", seek, { once: true });

        return logs.join("\\n");
        }
        """
    )

    # Chat callback (messages + timecode radio)
    def _chat_send(
        user_msg, chat_history,
        video_path, ctx_before, ctx_after, top_k, method, alpha,
        rerank, rerank_model, overfetch,
        provider,
        wx_base_url, wx_api_key, wx_model, wx_project_id,
        oa_base_url, oa_api_key, oa_model,
        an_api_key, an_model,
        embed_device, embed_model_override,
        enable_web, exa_api_key, exa_num_results,
        state_dict
    ):
        for out in on_chat(
            user_msg, chat_history,
            video_path, ctx_before, ctx_after, top_k, method, alpha,
            rerank, rerank_model, overfetch,
            provider,
            wx_base_url, wx_api_key, wx_model, wx_project_id,
            oa_base_url, oa_api_key, oa_model,
            an_api_key, an_model,
            embed_device, embed_model_override,
            enable_web, exa_api_key, exa_num_results,
            state_dict
        ):
            yield out

    send_btn.click(
        _chat_send,
        inputs=[
            chat_tb, chat,
            video_dd, ctx_before_tb, ctx_after_tb, topk_tb, method_dd, alpha_tb,
            rerank_dd, rerank_model_tb, overfetch_tb,
            provider_dd,
            base_url_tb, api_key_tb, model_tb, project_tb,
            oa_base_url_tb, oa_api_key_tb, oa_model_tb,
            an_api_key_tb, an_model_tb,
            embed_device_q_dd, embed_model_override_tb,
            enable_web_chk, exa_key_tb, exa_num_tb,
            state_dict
        ],
        outputs=[chat, ts_radio, ts_map_state]
    ).then(lambda: "", None, [chat_tb])

if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        share=True,
    )
