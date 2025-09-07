#!/usr/bin/env python3
"""
Gradio UI for video → (transcript + detections) → hybrid search → chat (watsonx.ai)

Fixes:
- Correctly updates Video component on scan (no tuple).
- Chat callback signature now matches inputs (no arity warnings).
- Lazy imports for heavy modules to avoid Paddle/Ultralytics noise on scan.
- Reuse existing outputs: choose from outputs/<video_stem>_*; auto-build index if missing.

Run:
  pip install gradio==4.* pandas numpy tqdm rank-bm25 sentence-transformers requests python-dotenv
  python app.py
"""

from __future__ import annotations

import re
import os
import sys
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
    import re
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

def run_extractor(video_path: str, lang: str, vclass: str, fps: Optional[int],
                  device: str, batch_size: int, conf: float, iou: float, max_det: int,
                  progress=None) -> str:
    """
    Returns output directory containing: transcript.srt, detections.csv, detections_by_frame.csv
    Lazy-imports heavy code; falls back to subprocess if not importable.
    """
    # Try in-process import
    local_ok = False
    try:
        from video_audio_multitool import run_pipeline as extractor_run_pipeline  # type: ignore
        local_ok = True
    except Exception:
        local_ok = False

    if local_ok:
        if progress: progress(0.05, desc="Starting extractor…")
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
        if progress: progress(0.05, desc="Starting extractor (subprocess)…")
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
        if progress: progress(0.65, desc="Building hybrid index…")
        idx = TranscriptRAGIndex.build_from_srt(
            srt_path=srt, outdir=None, window_sec=window,
            model_name=embed_model, device=device, anchor=anchor
        )
        return str(idx.root)
    else:
        import subprocess
        if progress: progress(0.65, desc="Building hybrid index (subprocess)…")
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
    # Lazy import
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

from typing import List, Dict
import requests

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

    Args:
        base_url: e.g. "https://eu-de.ml.cloud.ibm.com"
        api_key: IBM Cloud API key
        model: watsonx model_id (e.g. "mistralai/mistral-medium-2505")
        project_id: your watsonx.ai project UUID
        messages: OpenAI-style messages = [{"role": "...", "content": "..."}]
        temperature, max_tokens, timeout, version: usual controls
    Returns:
        Assistant message text.
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

    # 2) Convert OpenAI-style messages to watsonx chat schema
    def _wx_content(role: str, content):
        # watsonx requires user content as a list of parts; system can be plain string
        if isinstance(content, list):
            return content
        if role == "user":
            return [{"type": "text", "text": str(content)}]
        if role == "system":
            return str(content)
        # assistant can be a string; keep it simple
        if role == "assistant":
            return str(content)
        # default: wrap as text part
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
        "model_id": model,          # note: model_id (not "model")
        "project_id": project_id,   # required by watsonx.ai
        "messages": wx_messages,
        "temperature": temperature, # top-level for chat
        "max_tokens": max_tokens,   # top-level for chat
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # Helpful to inspect errors from service if any:
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"watsonx.ai error {resp.status_code}: {resp.text}") from e

    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Bad LLM response: {data}")
    return data["choices"][0]["message"]["content"]


# ---------------------
# Gradio callbacks
# ---------------------
def do_scan(folder):
    vids = list_videos(folder)
    # 1) Update the dropdown of videos
    dd_update = gr.update(choices=vids, value=(vids[0] if vids else None))
    # 2) Clear/Hide the video preview until a selection is made
    vid_update = gr.update(value=None, visible=bool(vids))
    return dd_update, vid_update

def _on_select(video, outputs_root):
    # Update Video player with selected file (show it)
    vp_update = gr.update(value=video, visible=True)
    # Scan outputs for this video and update the existing outputs dropdown
    matches = find_existing_outputs_for_video(video, outputs_root)
    dd_update = gr.update(choices=matches, value=(matches[0] if matches else None))
    # Status message
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


def on_chat(
    user_msg, history,
    video_path, ctx_before, ctx_after, top_k, method, alpha,
    rerank, rerank_model, overfetch, base_url, api_key, model, embed_device, embed_model_override,
    state_dict
):
    msgs = list(history or [])

    if not video_path:
        msgs.append({"role": "assistant", "content": "Please select a video first."})
        return msgs, gr.update(choices=[], value=None, visible=False), {}

    if not base_url or not api_key or not model:
        msgs.append({"role": "assistant", "content": "Please set your Watsonx base URL, API key, and model in the right panel."})
        return msgs, gr.update(choices=[], value=None, visible=False), {}

    rec = (state_dict or {}).get(str(video_path))
    if not rec:
        msgs.append({"role": "assistant", "content": "No outputs mapped for this video. Click 'Use selected outputs' or 'Generate'."})
        return msgs, gr.update(choices=[], value=None, visible=False), {}

    msgs.append({"role": "user", "content": str(user_msg)})

    idx_dir = rec["index_dir"]
    idx = load_index(idx_dir)

    hits, ctx_text = compile_context_blocks(
        idx=idx, query=str(user_msg), top_k=int(top_k), method=method, alpha=float(alpha),
        rerank=rerank, rerank_model=rerank_model, overfetch=int(overfetch),
        ctx_before=int(ctx_before), ctx_after=int(ctx_after),
        device=(None if embed_device == "auto" else embed_device),
        embed_model_override=(None if not embed_model_override else embed_model_override)
    )

    system = (
        "Tu es un assistant qui répond uniquement à partir du transcript du match.\n"
        "Cite les timecodes [HH:MM:SS,mmm–HH:MM:SS,mmm] pertinents.\n"
        "Si l'information n'est pas dans le contexte, dis-le honnêtement."
    )
    retrieval_blob = f"Contexte (extraits du transcript):\n\n{ctx_text}"
    user_with_ctx = f"Question:\n{user_msg}\n\n{retrieval_blob}"

    llm_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_with_ctx},
    ]

    try:
        answer = watsonx_chat(
            base_url=base_url, api_key=api_key, model=model, project_id="9917d93b-b2f9-4049-8dd9-d8772fc37cae",
            messages=llm_messages, temperature=0.2, max_tokens=700
        )
    except Exception as e:
        answer = f"LLM error: {e}"

    msgs.append({"role": "assistant", "content": answer})

    # Build a clickable list of main passages (start–end + short snippet)
    labels = []
    label_to_start = {}
    for h in hits:
        # choose the main line (offset==0) if present
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

    radio_update = gr.update(
        choices=labels,
        value=None,
        visible=bool(labels)
    )

    # Yield full triple: messages, radio choices update, and label->start map
    return msgs, radio_update, label_to_start


# ---------------------
# Gradio Layout
# ---------------------
with gr.Blocks(title="Video RAG Chat (Gradio + watsonx.ai)", fill_height=True) as demo:
    gr.Markdown("## 🎬 Video RAG Chat\nScan a folder → select a video → reuse **existing outputs** or **Generate** new → chat with the content.")

    with gr.Row():
        with gr.Column(scale=3):
            folder_tb = gr.Textbox(label="Folder to scan for videos", value=str(Path.cwd() / "videos"))
            outputs_root_tb = gr.Textbox(label="Outputs folder (where processed runs are stored)", value=str(Path.cwd() / "outputs"))
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

                with gr.Accordion("watsonx.ai connection", open=True):
                    base_url_tb = gr.Textbox(label="OpenAI-compatible Base URL (watsonx.ai Model Gateway or proxy)", placeholder="https://<your-gateway>")
                    api_key_tb = gr.Textbox(label="API Key", type="password")
                    model_tb = gr.Textbox(label="Model name", placeholder="ibm/granite-13b-chat-v2")

                ts_radio = gr.Radio(choices=[], label="Timecodes (click to seek)", interactive=True, visible=False)
                ts_map_state = gr.State({})  # label -> start_seconds

                chat = gr.Chatbot(height=520, type="messages")
                chat_tb = gr.Textbox(placeholder="Ask about the video…", label="Message")
                send_btn = gr.Button("Send", variant="primary")

            state_dict = gr.State({})


    scan_btn.click(fn=do_scan, inputs=[folder_tb], outputs=[video_dd, video_player])

    video_dd.change(fn=_on_select, inputs=[video_dd, outputs_root_tb], outputs=[video_player, existing_outputs_dd, status_box])

    use_existing_btn.click(
        fn=on_use_existing,
        inputs=[
            existing_outputs_dd, video_dd, state_dict,
            window_tb, anchor_dd, embed_model_tb, embed_device_dd
        ],
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

    # Wiring
    ts_radio.change(
        inputs=[ts_radio, ts_map_state],
        outputs=[status_box],   # <— TEMP: show debug info; remove later if you want
        js="""
        (label, map) => {
        const logs = [];
        logs.push(`label: ${label}`);

        // --- get seconds from state or parse from label ---
        const parseFromLabel = (lbl) => {
            const m = /\\b(\\d{2}):(\\d{2}):(\\d{2}),(\\d{3})\\b/.exec(lbl || "");
            if (!m) return NaN;
            const hh = +m[1], mm = +m[2], ss = +m[3];
            return hh*3600 + mm*60 + ss;
        };
        let seconds = (map && typeof map[label] === "number") ? map[label] : parseFromLabel(label);
        logs.push(`seconds: ${seconds}`);
        if (!Number.isFinite(seconds)) return `Bad seconds from label/map`;

        // --- find the <video> element (works with shadow DOM) ---
        const host = document.querySelector("#video_preview");
        logs.push(`host found: ${!!host}`);

        const tryFindVideo = (root) => {
            if (!root) return null;
            // 1) direct video under root
            let v = root.querySelector ? root.querySelector("video") : null;
            if (v) return v;
            // 2) explicit <gradio-video> child
            const gv = root.querySelector ? root.querySelector("gradio-video") : null;
            if (gv && gv.shadowRoot) {
            v = gv.shadowRoot.querySelector("video");
            if (v) return v;
            }
            // 3) search any child shadow roots for a video
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
            // last resort: look globally
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

        // --- seek ---
        const seek = () => {
            try {
            // clamp just in case
            const dur = Number.isFinite(video.duration) ? video.duration : Infinity;
            const t = Math.max(0, Math.min(dur, seconds));
            video.currentTime = t;
            // try to play (user gesture should allow it)
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


    # IMPORTANT: function signature matches inputs count
    def _chat_send(
        user_msg, chat_history,
        video_path, ctx_before, ctx_after, top_k, method, alpha,
        rerank, rerank_model, overfetch,
        base_url, api_key, model, embed_device, embed_model_override,
        state_dict
    ):
        return on_chat(
            user_msg, chat_history,
            video_path, ctx_before, ctx_after, top_k, method, alpha,
            rerank, rerank_model, overfetch, base_url, api_key, model, embed_device, embed_model_override,
            state_dict
        )

    send_btn.click(
        _chat_send,
        inputs=[
            chat_tb, chat,
            video_dd, ctx_before_tb, ctx_after_tb, topk_tb, method_dd, alpha_tb,
            rerank_dd, rerank_model_tb, overfetch_tb,
            base_url_tb, api_key_tb, model_tb,
            embed_device_q_dd, embed_model_override_tb,
            state_dict
        ],
        outputs=[chat, ts_radio, ts_map_state]
    ).then(lambda: "", None, [chat_tb])


# if __name__ == "__main__":
#     demo.queue(max_size=32).launch(
#         server_name="0.0.0.0",
#         server_port=int(os.getenv("PORT", "7860")),
#         show_error=True
#     )

if __name__ == "__main__":
    # If localhost is blocked, share must be True
    demo.queue(max_size=32).launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True
    )
