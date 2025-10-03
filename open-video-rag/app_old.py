#!/usr/bin/env python3
"""
Gradio UI for video → (transcript + detections) → hybrid search → chat (multi-LLM)
+ Providers: OpenAI • Anthropic
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
import numpy as np
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import gradio as gr
import requests
from datetime import datetime
import difflib
import csv as _csv
import shutil
from dotenv import load_dotenv

load_dotenv()


# ---------------------
# Helpers
# ---------------------
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


# Cache of loaded indexes: {index_dir: TranscriptRAGIndex}
_INDEX_CACHE: Dict[str, object] = {}


# --- Recency heuristics (NEW) ---
_RECENCY_PATTERNS = [
    r"\b(now|today|these days|currently|as of|latest|most recent|up[- ]to[- ]date|news|update[s]?)\b",
    r"\b(maintenant|aujourd'hui|actuel(?:le|s)?|derni(?:er|ères?)|réc(?:ent|entes?)|mise[s]?\s*à\s*jour)\b",
    r"\bas of\s*20\d{2}\b", r"\bin\s*20\d{2}\b",  # "as of 2025", "in 2025"
]


# Heuristics for AUTO routing
_VISUAL_PATTERNS = [
    r"\b(logo|brand|marque|embl[eè]me|badge|maillot|jersey)s?\b",
    r"\b(scoreboard|score|tableau d'affichage)\b",
    r"\b(color|couleur|couleurs)\b",
    r"\b(camera|cam[ée]ra|angle|shot|plan|transition|cut|montage)\b",
    r"\b(onscreen|on[- ]screen|à l'?écran|texte à l'?écran|OCR)\b",
    r"\b(scenes?|sc[eè]nes?|frame|cadre|image|diapo|slide|UI|interface)\b",
    r"\b(number on (the )?jersey|num[eé]ro sur (le )?maillot)\b",
    r"\b(what is shown|qu'est[- ]ce qui est montr[eé])\b",
    r"\b(at\s*\d{1,2}:\d{2}(:\d{2})?(,\d{3})?)\b",  # timestamps often imply visual inspection
]


_TRANSCRIPT_PATTERNS = [
    r"\b(said|say|says|mention|quote|dialog(ue)?|conversation|talk|speech|narrat(?:ion|or))\b",
    r"\b(a dit|dit|disent|mentionne|citation|dialogue|conversation|parle|voix off|audio)\b",
    r"\b(subtitle|subtitles|caption|srt|transcript(ion)?)\b",
    r"\b(what does .* (say|mean)|que (dit|signifie))\b",
]


# ---------- Entity search helpers (robust over spelling/variants) ----------
_ENTITY_INTENT_PATTERNS = [
    r"\b(?:all|toutes?|every|liste(?:r)?|montre(?:r)?)\s+(?:the\s+)?scenes?\s+(?:where|with|containing)\s+(?P<name>.+?)\b",
    r"\b(?:donne(?:z)?|give)\s+(?:moi\s+)?toutes?\s+les?\s+sc[eè]nes?\s+o[uù]\s+(?P<name>.+?)\s+(?:appara[iî]t|figure|est (?:pr[eé]sent[e]?)|se voit)\b",
    r"\b(?:find|trouve[r]?)\s+(?:all\s+)?scenes?\s+(?:with|featuring)\s+(?P<name>.+?)\b",
    r"\bscenes?\s+(?:with|de|où)\s+(?P<name>.+?)\b",
]


_SC_LINE_RE = re.compile(r"^\s*SCENE\s*N[°o]\s*(\d+)\s*:\s*([0-9:,–\- ]+)?", re.I | re.M)


ALLOWED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}  # add more if needed


import unicodedata
import re
from pathlib import Path


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def srt_to_seconds(s: str) -> int:
    # "HH:MM:SS,mmm" → seconds (floor)
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", s.strip())
    if not m:
        return 0
    hh, mm, ss, _ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss


def safe_stem(path: Path) -> str:
    s = path.stem.strip()
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s)
    return s or "video"


def format_source_label(src: str | None) -> str:
    s = (src or "transcript").lower()
    if s.startswith("vlm"): return "VLM"
    if s == "transcript":  return "SRT"
    return s.upper()


def on_upload_video(file_path: str, videos_dir: str):
    if not file_path:
        raise gr.Error("No file uploaded.")

    src = Path(file_path)
    if not src.exists():
        raise gr.Error("Upload failed: temporary file not found.")

    ext = src.suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise gr.Error(f"Invalid file type: {ext}. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}")

    dst_dir = Path(videos_dir).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    # unique name if collision
    dest = dst_dir / src.name
    if dest.exists():
        dest = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"

    shutil.copy2(src, dest)

    # refresh list
    vids = list_videos(str(dst_dir))  # you already have this helper
    return (
        gr.update(choices=vids, value=str(dest)),           # dropdown
        gr.update(value=str(dest), visible=True),           # video player
        f"✅ Uploaded to: {dest}"                            # status
    )


def _normalize_dash(s: str) -> str:
    return s.replace("–", "-").replace("—", "-")


def sanitize_scene_output(text: str, allowed_nums: set[int], times_by_num: dict[int, tuple[str,str]]) -> str:
    lines = text.splitlines()
    out = []
    for ln in lines:
        m = _SC_LINE_RE.search(ln)
        if not m:
            out.append(ln); continue
        try:
            num = int(m.group(1))
        except Exception:
            # skip malformed scene line
            continue
        if num not in allowed_nums:
            # Drop hallucinated scene
            continue
        start, end = times_by_num.get(num, ("00:00:00,000","00:00:00,000"))
        # Rebuild canonical prefix
        prefix = f"SCENE N°{num}: {start}–{end}"
        # Replace everything up to dash and keep the rest
        # Find " — " or " - " separator; if missing, add an em dash
        parts = re.split(r"\s+—\s+|\s+-\s+", _normalize_dash(ln), maxsplit=1)
        suffix = parts[1] if len(parts) > 1 else ln[m.end():].lstrip(" :-—–")
        clean = f"{prefix} — {suffix.strip()}" if suffix.strip() else prefix
        out.append(clean)
    return "\n".join(out)


def _normalize_entity_key(s: str) -> str:
    s = (s or "").strip()
    s = _strip_accents(s).lower()
    # drop punctuation → keep letters/digits/space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # optional: remove leading articles
    s = re.sub(r"^(the|la|le|les|l|une|un)\s+", "", s)
    return s


def _detect_entity_intent(question: str) -> tuple[bool, Optional[str]]:
    q = (question or "").strip()
    for pat in _ENTITY_INTENT_PATTERNS:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            name = (m.group("name") or "").strip().strip("'\"“”‘’.,:;!?")
            # prune trailing generic words
            name = re.sub(r"\b(?:appears?|appara[iî]t|figure|present[e]?|se voit)\b.*$", "", name, flags=re.I).strip()
            return True, name
    # Very short pattern: "nike scenes", "scenes nike"
    if re.search(r"\bscenes?\b", q, flags=re.I):
        # grab a plausible last token phrase
        tail = re.sub(r".*\bscenes?\b", "", q, flags=re.I).strip()
        if tail:
            return True, tail.strip(" .,:;!?")
    return False, None


def _load_entities_scenes_from_context(outputs_dir: str) -> tuple[dict, dict]:
    """
    Returns (entities_scenes, original_key_by_norm).
    entities_scenes: {original_key: [scene_ids]}
    original_key_by_norm: {normalized_key: original_key}
    """
    art = find_visual_artifacts_in(outputs_dir)
    cpath = art.get("context")
    if not cpath or not Path(cpath).exists():
        return {}, {}
    try:
        data = json.loads(Path(cpath).read_text(encoding="utf-8"))
        es = data.get("entities_scenes") or {}
        # build normalized map → pick longest/original repr deterministically
        norm_map = {}
        for k in es.keys():
            nk = _normalize_entity_key(k)
            if nk and nk not in norm_map:
                norm_map[nk] = k
        return es, norm_map
    except Exception:
        return {}, {}


def _fuzzy_pick_entity(name: str, norm_map: dict, threshold: float = 0.82) -> Optional[str]:
    """
    Return the ORIGINAL key from entities_scenes that best matches the user's entity name.
    Strategy:
      1) exact normalized key
      2) difflib closest
      3) token-set containment heuristic
    """
    if not name or not norm_map:
        return None
    nk = _normalize_entity_key(name)
    if nk in norm_map:
        return norm_map[nk]

    candidates = list(norm_map.keys())
    # difflib
    close = difflib.get_close_matches(nk, candidates, n=1, cutoff=threshold)
    if close:
        return norm_map[close[0]]

    # token-set overlap
    toks = set(nk.split())
    best, best_j = None, 0.0
    for cand in candidates:
        ctoks = set(cand.split())
        j = len(toks & ctoks) / max(1, len(toks | ctoks))
        if j > best_j:
            best, best_j = cand, j
    if best and best_j >= 0.6:
        return norm_map[best]
    return None


def _load_scene_rows_from_csv(outputs_dir: str, scene_ids: list[int]) -> list[dict]:
    """
    Read *.csv produced by visual_extraction to get generated_text + timecodes for given scenes.
    Returns list of rows: {scene_id:int, start_timecode:str, end_timecode:str, generated_text:str}
    """
    art = find_visual_artifacts_in(outputs_dir)
    csv_path = art.get("csv")
    rows = []
    if not csv_path or not Path(csv_path).exists():
        return rows
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for rec in r:
                try:
                    sid = int(rec.get("scene_id", "0"))
                except Exception:
                    continue
                if sid in scene_ids:
                    rows.append({
                        "scene_id": sid,
                        "start_timecode": rec.get("start_timecode") or rec.get("start_tc") or "00:00:00,000",
                        "end_timecode": rec.get("end_timecode") or rec.get("end_tc") or "00:00:00,000",
                        "generated_text": rec.get("generated_text", "") or "",
                    })
    except Exception:
        return []
    # keep scene order
    rows.sort(key=lambda x: x["scene_id"])
    return rows


def _hits_from_scene_rows(scene_rows: list[dict]) -> list[dict]:
    """
    Convert scene rows into 'hits' compatible with the context panel + timecode radio.
    """
    hits = []
    for i, r in enumerate(scene_rows, start=1):
        start, end = r["start_timecode"], r["end_timecode"]
        txt = r["generated_text"].strip()
        hits.append({
            "rank": i,
            "start_srt": start,
            "end_srt": end,
            "source": "vlm",
            "method": "entities_scenes",
            "hretr_score": None,
            "rerank_score": None,
            "context": [{
                "offset": 0,
                "start_srt": start,
                "end_srt": end,
                "text": txt,
                "source": "vlm",
            }],
        })
    return hits


def wants_recent_info(msg: str, current_year: Optional[int] = None) -> tuple[bool, list[int]]:
    """
    Return (needs_recent, years_mentioned).
    needs_recent is True if message asks for 'now/latest/recent' or mentions a year
    that is close to the present (>= current_year-1).
    """
    s = (msg or "").lower()
    if any(re.search(p, s) for p in _RECENCY_PATTERNS):
        years = [int(y) for y in _YEAR_RE.findall(s)]
        return True, years

    years = [int(y) for y in _YEAR_RE.findall(s)]
    if years and (current_year is not None) and any(y >= current_year - 1 for y in years):
        return True, years
    return False, years


def auto_select_sources_from_query(q: str) -> tuple[bool, bool, str]:
    """
    Returns (want_transcript, want_visual, reason)
    Strategy:
      - Only visual terms -> visual
      - Only transcript terms -> transcript
      - Both (or none) -> both (be generous)
    """
    s = (q or "").lower()
    v_hits = sum(1 for pat in _VISUAL_PATTERNS if re.search(pat, s))
    t_hits = sum(1 for pat in _TRANSCRIPT_PATTERNS if re.search(pat, s))

    if v_hits > 0 and t_hits == 0:
        return (False, True, "auto: visual cues only")
    if t_hits > 0 and v_hits == 0:
        return (True, False, "auto: transcription cues only")
    # ambiguous or none -> both
    return (True, True, "auto: ambiguous or mixed → both")


def find_existing_indexes(outputs_dir: str) -> Dict[str, Optional[str]]:
    """
    Returns {'transcript': <dir or None>, 'visual': <dir or None>} by
    scanning rag_index_* folders and inspecting their meta.sources.
    """
    base = Path(outputs_dir)
    result = {"transcript": None, "visual": None}
    idx_dirs = sorted([p for p in base.glob("rag_index_*") if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
    for p in idx_dirs:
        try:
            idx = load_index(str(p))
            sources = set(x.lower() for x in getattr(getattr(idx, "meta", None), "sources", []) or [])
        except Exception:
            # If we can’t load it, skip
            continue
        if ("transcript" in sources or "srt" in sources) and not result["transcript"]:
            result["transcript"] = str(p)
        if any(s.startswith("vlm") or s == "visual" for s in sources) and not result["visual"]:
            result["visual"] = str(p)
        if result["transcript"] and result["visual"]:
            break
    return result


def _ctx_md_from_hits_aggregated(enriched_hits: list[dict], title: str = "Retrieved passages") -> str:
    if not enriched_hits:
        return f"<h3>{_html_escape(title)}</h3><em>(no passages retrieved)</em>"

    html = [
        "<style>",
        "details > summary { cursor: pointer; }",
        ".ctx-block { margin: 6px 0 12px 0; padding: 8px 10px; border: 1px solid #333;"
        " border-radius: 8px; background: #0b0b0b; }",
        ".ctx-line { display: block; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        ".ctx-meta { opacity: .8; font-size: .9em; }",
        "</style>",
        f"<h3>{_html_escape(title)}</h3>"
    ]

    for h in enriched_hits:
        parts = [h.get("method","")]
        if "hretr_score" in h:
            try: parts.append(f"hyb={float(h['hretr_score']):.4f}")
            except Exception: pass
        if h.get("rerank_score") is not None:
            try: parts.append(f"rerank={float(h['rerank_score']):.4f}")
            except Exception: pass
        method_lbl = ", ".join([p for p in parts if p]) or "retrieval"
        src_lbl = format_source_label(h.get("source"))
        header = f"[{int(h.get('rank',0)):02}] {h.get('start_srt','??')} → {h.get('end_srt','??')}  ({src_lbl} • {method_lbl})"

        ctx_sorted = sorted(h.get("context", []), key=lambda c: (c.get("offset",0) != 0, c.get("offset",0)))
        body_lines = []
        for c in ctx_sorted:
            bullet = "•" if c.get("offset",0) == 0 else "·"
            tc = f"{c.get('start_srt','??')}–{c.get('end_srt','??')}"
            txt = _html_escape((c.get("text","") or "").strip())
            body_lines.append(f"<span class='ctx-line'><span class='ctx-meta'>{bullet} [{tc}]</span> {txt}</span>")

        html.append(
            f"<details><summary>{_html_escape(header)}</summary>"
            f"<div class='ctx-block'>{''.join(body_lines)}</div>"
            f"</details>"
        )
    return "\n".join(html)


def list_videos(folder: str) -> List[str]:
    p = Path(folder).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    hits = []
    for ext in VIDEO_EXTS:
        hits.extend(str(x) for x in p.rglob(f"*{ext}"))
    hits.sort()
    return hits


def find_existing_outputs_for_video(video_path: str, outputs_root: str) -> List[str]:
    """
    Look under outputs_root for subfolders starting with safe_stem(video) + '_'.
    We NO LONGER require transcript.srt to list them.
    """
    root = Path(outputs_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    stem = safe_stem(Path(video_path))
    pattern = f"{stem}_*"
    matches = [p for p in root.glob(pattern) if p.is_dir()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in matches]


def find_any_srt(outputs_dir: str) -> Optional[Path]:
    """
    Pick the best available SRT in outputs_dir.
    Priority: transcript.srt > *audio*.srt > newest *.srt
    """
    p = Path(outputs_dir)
    if not p.exists():
        return None
    # Exact name first
    t = p / "transcript.srt"
    if t.exists():
        return t
    # Any '*audio*.srt'
    audio_srts = sorted(p.glob("*audio*.srt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if audio_srts:
        return audio_srts[0]
    # Fallback: newest *.srt
    all_srts = sorted(p.glob("*.srt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return all_srts[0] if all_srts else None


def find_audio_file_in(outputs_dir: str) -> Optional[Path]:
    p = Path(outputs_dir)
    if not p.exists():
        return None
    cand: list[Path] = []
    for ext in AUDIO_EXTS:
        cand.extend(p.glob(f"*{ext}"))
    if not cand:
        return None
    cand.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cand[0]


def has_transcript(outputs_dir: str) -> bool:
    return find_any_srt(outputs_dir) is not None


def find_visual_artifacts_in(outputs_dir: str) -> Dict[str, Optional[Path]]:
    p = Path(outputs_dir)
    csvs = sorted(p.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    ctxs = sorted(p.glob("*_context.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    metas = sorted(p.glob("*_metadata.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return {
        "csv": (csvs[0] if csvs else None),
        "context": (ctxs[0] if ctxs else None),
        "meta": (metas[0] if metas else None),
    }


def has_visual_artifacts(outputs_dir: str) -> bool:
    a = find_visual_artifacts_in(outputs_dir)
    return bool(a["csv"] and a["context"] and a["meta"])


def latest_rag_index_dir(outputs_dir: str) -> Optional[str]:
    base = Path(outputs_dir)
    idx_dirs = [p for p in base.glob("rag_index_*") if p.is_dir()]
    if not idx_dirs:
        return None
    idx_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(idx_dirs[0])


def run_visual_strategy(
    video_path: str,
    outputs_dir: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    profile: str = "brand_heavy",
    max_concurrency: int = 8,
    stream: bool = False,
    stream_mode: str = "none",
    force: bool = False,
) -> Dict[str, str]:
    """
    Calls visual_extraction.py to generate:
      *_context.json, *_metadata.txt, *.csv
    We set cwd=outputs_dir so files land there.
    Returns dict with produced folder and a quick status string.
    """
    import subprocess
    from pathlib import Path

    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    # 1) Find the script next to this app file (fallback: repo root / CWD)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "visual_extraction.py",
        Path.cwd() / "visual_extraction.py",
    ]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        raise FileNotFoundError(
            "visual_extraction.py not found. Expected at:\n - "
            + "\n - ".join(str(p) for p in candidates)
        )

    # 2) Build the command with the ABSOLUTE script path
    cmd = [
        sys.executable, str(script),
        "--video", video_path,
        "--profile", profile,
        "--base-url", base_url,
        "--api-key", api_key,
        "--model", model,
        "--max-concurrency", str(int(max_concurrency)),
    ]
    if stream:
        cmd += ["--stream", "--stream-mode", stream_mode]

    # 3) Keep cwd=outputs_dir so artifacts land there
    subprocess.run(cmd, check=True, cwd=str(outputs_dir))
    return {"out_dir": outputs_dir, "status": f"VLM artifacts written to {outputs_dir}"}


def build_visual_index(outputs_dir: str, embed_model: str, embed_device: Optional[str]):
    from visual_rag_index import build_visual_index_from_outputs  # new module
    return build_visual_index_from_outputs(
        outputs_dir=outputs_dir,
        model_name=embed_model,
        device=(None if embed_device == "auto" else embed_device)
    )


# ---------------------
# Extractor / Indexer (lazy imports)
# ---------------------
def run_extractor(
    video_path: str, lang: str, vclass: str, fps: Optional[int],
    device: str, batch_size: int, conf: float, iou: float, max_det: int,
    progress=None,
    enable_detection: bool = True,
) -> str:
    """
    Returns output directory containing: transcript.srt, detections.csv, detections_by_frame.csv
    """
    local_ok = False
    try:
        from video_audio_multitool import run_pipeline as extractor_run_pipeline  # type: ignore
        local_ok = True
    except Exception:
        local_ok = False

    # If detection is disabled, DO NOT use the local API (it tends to load YOLO anyway).
    # Use the CLI path where we can pass --no-detect.
    if not enable_detection:
        local_ok = False

    det_conf = conf
    det_iou = iou
    det_max = (0 if not enable_detection else max_det)

    if local_ok:
        if progress: progress(0.05, desc="Starting extractor…")
        out_dir = extractor_run_pipeline(
            input_arg=video_path, lang=lang, vclass=vclass, fps=fps,
            device_choice=device, batch_size=batch_size,
            det_conf=det_conf, det_iou=det_iou, det_max=det_max
        )
        return str(out_dir)
    else:
        import subprocess, time
        cmd = [
            sys.executable, "video_audio_multitool.py", video_path,
            "--lang", lang, "--vclass", vclass, "--device", device,
            "--batch-size", str(batch_size), "--det-conf", str(det_conf),
            "--det-iou", str(det_iou), "--max-det", str(det_max)
        ]
        if fps: cmd += ["--fps", str(fps)]
        if not enable_detection:
            cmd += ["--no-detect"]   # <— this prevents loading the model entirely

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
        source_label = format_source_label(h.get("source"))
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        lines: List[str] = []
        for c in ctx_sorted:
            line_source = format_source_label(c.get("source"))
            lines.append(f"[{c['start_srt']} - {c['end_srt']}] ({line_source}) {c['text']}".strip())
        header = "### Passage {rank} | Source: {source} | Main: {start} - {end}".format(
            rank=h.get('rank'),
            source=source_label,
            start=h.get('start_srt'),
            end=h.get('end_srt'),
        )
        block = header + "\n" + "\n".join(lines)
        blocks.append(block)
    full_context = "\n\n".join(blocks)
    return enriched, full_context


def compile_context_blocks_multi(
    indexes: List[Tuple[object, str]],  # [(idx_obj, "SRT"/"VLM"), ...]
    query: str,
    top_k: int,
    method: str, alpha: float,
    rerank: str, rerank_model: str, overfetch: int,
    ctx_before: int, ctx_after: int,
    device: Optional[str], embed_model_override: Optional[str]
) -> Tuple[List[Dict], str]:
    """
    Split top_k across indexes (fair share), then merge; tag source in header.
    """
    if not indexes:
        return [], ""
    share = max(1, int(np.ceil(top_k / max(1, len(indexes)))))
    all_hits: List[Dict] = []
    blocks = []
    for idx_obj, label in indexes:
        hits, ctx_text = compile_context_blocks(
            idx=idx_obj, query=query, top_k=share, method=method, alpha=alpha,
            rerank=rerank, rerank_model=rerank_model, overfetch=overfetch,
            ctx_before=ctx_before, ctx_after=ctx_after,
            device=device, embed_model_override=embed_model_override
        )
        # Prefix block to separate sources
        if ctx_text.strip():
            blocks.append(f"#### Source: {label}\n{ctx_text}")
        # Stamp source label into hits for the timecode list
        for h in hits:
            h["source"] = label.lower() if label in ("VLM","SRT") else (h.get("source") or "")
        all_hits.extend(hits)
    # Keep at most top_k by current order (simple interleave/concat)
    all_hits = all_hits[:top_k]
    full_context = "\n\n".join(blocks)
    return all_hits, full_context


def openai_chat_stream(
    api_key: str,
    model: str,
    messages: List[Dict],
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
):
    """
    Stream tokens from OpenAI-compatible Chat Completions API.
    Yields text deltas as they arrive.
    """
    root = (base_url or "https://api.openai.com").rstrip("/")
    url = f"{root}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "" or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except Exception:
                continue


# ---------------------
# LLM clients (OpenAI, Anthropic)
# ---------------------
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


def anthropic_chat_stream(
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
):
    """
    Stream tokens from Anthropic Messages API.
    Yields text deltas as they arrive.
    """
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
        "stream": True,
    }
    if system_str:
        payload["system"] = system_str

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw.split("data:", 1)[1].strip()
            if data == "" or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {}).get("text", "")
                    if delta:
                        yield delta
            except Exception:
                continue


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


def llm_chat_stream(
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
):
    """
    Return a generator that yields text chunks from the selected provider.
    """
    p = (provider or "openai").lower()
    if p == "openai":
        return openai_chat_stream(
            api_key=cfg.get("oa_api_key",""),
            model=cfg.get("oa_model",""),
            messages=messages,
            base_url=cfg.get("oa_base_url") or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif p == "anthropic":
        return anthropic_chat_stream(
            api_key=cfg.get("an_api_key",""),
            model=cfg.get("an_model",""),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider for streaming: {provider}")


# Unified chat wrapper
def llm_chat(
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    provider = (provider or "anthropic").lower()
    if provider == "openai":
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


def llm_detect_intent_entities(
    provider: str,
    cfg: Dict,
    latest_user_msg: str,
    history_msgs: List[Dict],
) -> Dict:
    """
    Ask the LLM to (a) detect intent, (b) extract entity strings if intent=entity_search.
    Returns: {"intent": "entity_search"|"other", "entities": [str], "reason": str}
    """
    history_text = _format_history_as_text(history_msgs, max_turns=8, max_chars=3000)
    system = (
        "You are a routing classifier.\n"
        "Task: Decide if the latest user message is an ENTITY-SEARCH about a video, e.g., "
        "\"give me all scenes where Nike appeared\" / \"scenes with PSG logo\".\n"
        "If it IS entity-search: extract the entity mentions as a list of short strings.\n"
        "If it is NOT entity-search: intent='other' and entities=[].\n"
        "STRICT OUTPUT: JSON ONLY with keys: intent, entities, reason.\n"
        "• intent ∈ {'entity_search','other'}\n"
        "• entities = array of strings (may be 1+)\n"
        "• reason = short justification"
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message:\n{latest_user_msg}\n\n"
        "Output JSON ONLY like:\n"
        '{"intent":"entity_search","entities":["nike","psg"],"reason":"…"}'
    )
    try:
        out = llm_chat(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=250,
        )
        data = extract_json(out) or {}
    except Exception as e:
        data = {"intent":"other","entities":[],"reason":f"llm error: {e}"}
    intent = (data.get("intent") or "other").strip().lower()
    ents = [str(x).strip() for x in (data.get("entities") or []) if str(x).strip()]
    return {"intent": ("entity_search" if intent=="entity_search" and ents else "other"),
            "entities": ents, "reason": data.get("reason","")}


def llm_match_entities_to_keys(
    provider: str,
    cfg: Dict,
    query_entities: List[str],
    candidate_keys: List[str],
) -> Dict:
    """
    Ask the LLM to select which candidate_keys correspond to the query_entities.
    Returns: {"matches": [{"query":"nike","keys":["Nike","NIKE®"]}, ...], "flat_keys": ["Nike", ...]}
    """
    # Keep payload tight; if many keys, send top N (but entities_scenes keys are usually modest)
    system = (
        "You are matching user-specified entities to canonical keys extracted from a video analysis.\n"
        "Given: (a) query entity strings, (b) the exact list of candidate keys (canonical forms).\n"
        "Rules:\n"
        "• Be robust to casing, accents, punctuation, and common variants (e.g., 'PSG', 'Paris Saint-Germain').\n"
        "• ONLY choose from the provided candidate keys; do not invent.\n"
        "• Prefer exact/near-exact brand/org names over generic words.\n"
        "• If none match for a query entity, return an empty list for that entity.\n"
        "Output JSON ONLY with:\n"
        '{"matches":[{"query":"<q>","keys":["<k1>","<k2>"]},...], "flat_keys":["<k1>","<k2>",...]}'
    )
    user = json.dumps({
        "query_entities": query_entities,
        "candidate_keys": candidate_keys,
    }, ensure_ascii=False)
    try:
        out = llm_chat(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=600,
        )
        data = extract_json(out) or {}
    except Exception as e:
        data = {"matches": [], "flat_keys": [], "error": str(e)}
    # sanitize
    matches = data.get("matches") or []
    flat = data.get("flat_keys") or []
    flat = [str(k) for k in flat if str(k).strip()]
    # If LLM didn't provide flat_keys, derive from matches
    if not flat and matches:
        seen = set()
        for m in matches:
            for k in (m.get("keys") or []):
                k = str(k)
                if k and k not in seen:
                    seen.add(k); flat.append(k)
    return {"matches": matches, "flat_keys": flat}


def llm_decide_search(
    provider: str,
    cfg: Dict,
    question: str,
    transcript_context: str,
    explicit_flag: bool,
    history_text: str = "",
    *,                           # NEW: only keyword args after this
    recency_flag: bool = False,  # NEW
    years_mentioned: Optional[list[int]] = None,  # NEW
    current_year: Optional[int] = None,           # NEW
) -> Dict:
    """
    Returns dict: {need_search, query, answer, reason}
    """
    years_mentioned = years_mentioned or []
    system = (
        "You are a retrieval QA router.\n"
        "Inputs: (a) conversation so far, (b) a standalone user question, (c) transcript context,\n"
        "and (d) recency hints (flags + years mentioned).\n"
        "RULES (strict):\n"
        "• If the user explicitly asks to search the web → need_search=true.\n"
        "• If recency_flag=true OR the question asks for 'now', 'latest', 'most recent', 'today',\n"
        "  OR mentions a year >= (current_year-1) → need_search=true (regardless of transcript context).\n"
        "• Otherwise, if the answer is clearly in transcript context → need_search=false and answer with it.\n"
        "• Otherwise → need_search=true.\n\n"
        "Return JSON ONLY with schema:\n"
        '{"need_search": true|false, "query": string|null, "answer": string|null, "reason": string}\n'
        "If need_search=true: provide ONE precise web search string in 'query' (include any relevant years), 'answer' MUST be null.\n"
        "If need_search=false: 'answer' MUST contain the final answer grounded in transcript context, 'query' MUST be null."
    )

    user = (
        f"explicit_web_search_flag={explicit_flag}\n"
        f"recency_flag={recency_flag}\n"
        f"years_mentioned={years_mentioned}\n"
        f"current_year={current_year}\n\n"
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
# Helper: toggle provider panels visibility
# ---------------------
def toggle_provider_panels(provider: str):
    p = (provider or "").lower()
    return (
        gr.update(visible=(p == "openai")),
        gr.update(visible=(p == "anthropic")),
    )


def hard_clear():
    # Clears visible chat + related UI/state you use during a turn
    empty_msgs = []
    hide_radio = gr.update(choices=[], value=None, visible=False)
    empty_map = {}
    empty_ctx = ""   # context panel
    return empty_msgs, hide_radio, empty_map, empty_ctx, ""



import gradio as gr

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
                    # transcript extractor controls
                    lang, vclass, fps, device, batch, conf, iou, max_det,
                    # visual controls
                    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
                    # index settings
                    window, anchor, embed_model, embed_device,
                    use_detection
                    ):
    if not selected_outputs:
        raise gr.Error("No outputs folder selected.")
    out_dir = Path(selected_outputs).resolve()
    if not out_dir.exists():
        raise gr.Error(f"Selected outputs folder doesn't exist: {out_dir}")

    # Detect existing artifacts
    # Probe what we already have
    srt_path = find_any_srt(str(out_dir))
    have_srt = srt_path is not None
    have_vlm = has_visual_artifacts(str(out_dir))
    existing_idx = find_existing_indexes(str(out_dir))
    idx_dir_trans = existing_idx.get("transcript")
    idx_dir_vlm   = existing_idx.get("visual")

    # CASE matrix from your requirement:
    # - none → generate SRT + VLM → build both indexes
    # - only SRT → run VLM → build both indexes (reuse SRT index if exists; else build)
    # - only VLM → run SRT → build both indexes (reuse VLM index if exists; else build)
    # - both → reuse; if any index missing, build it

    # === Transcript path / index logic ===
    # 1) If an index already exists for transcript, we can reuse it (no SRT required).
    # 2) Else, if any .srt exists, build the index from it.
    # 3) Else, if an audio file exists, transcribe from that audio (counts as "something present"),
    #    then build the index.
    # 4) Else, do not auto-run heavy jobs here; ask user to click Generate.

    if not idx_dir_trans:
        if not have_srt:
            audio_path = find_audio_file_in(str(out_dir))
            if audio_path is not None:
                # We DO run extractor here because the user already put an audio file in outputs.
                out_dir_str = run_extractor(
                    video_path=str(audio_path),
                    lang=lang, vclass=vclass, fps=(None if not fps else int(fps)),
                    device=device, batch_size=int(batch),
                    conf=float(conf), iou=float(iou), max_det=int(max_det),
                    progress=None,
                    enable_detection=bool(use_detection),   # <--- important
                )
                # The extractor might choose/return a different outputs folder
                if Path(out_dir_str).resolve() != out_dir:
                    out_dir = Path(out_dir_str).resolve()
                srt_path = find_any_srt(str(out_dir))
                have_srt = srt_path is not None
        # Build transcript index if we now have an SRT and no existing index
        if have_srt and not idx_dir_trans:
            idx_dir_trans = build_index_from_srt(
                transcript_path=str(srt_path),
                window=int(window), anchor=anchor,
                embed_model=embed_model,
                device=(None if embed_device == "auto" else embed_device),
                progress=None,
            )

    # === Visuals: only use if all 3 files exist; do NOT run extraction here ===
    # If artifacts exist and no visual index, build it. Otherwise leave it for "Generate".
    if have_vlm and not idx_dir_vlm:
        idx_dir_vlm = build_visual_index(
            outputs_dir=str(out_dir),
            embed_model=embed_model,
            embed_device=embed_device,
        )

    # Refresh existing indexes after possibly creating artifacts
    if not idx_dir_trans or not idx_dir_vlm:
        existing_idx = find_existing_indexes(str(out_dir))
        idx_dir_trans = idx_dir_trans or existing_idx.get("transcript")
        idx_dir_vlm   = idx_dir_vlm   or existing_idx.get("visual")

    # Cache/load
    if idx_dir_trans: _ = load_index(idx_dir_trans)
    if idx_dir_vlm:   _ = load_index(idx_dir_vlm)

    # Persist in state
    sd = state_dict or {}
    rec = sd.get(str(video_path), {"out_dir": str(out_dir), "index_dirs": {}})
    rec["out_dir"] = str(out_dir)
    if idx_dir_trans: rec["index_dirs"]["transcript"] = idx_dir_trans
    if idx_dir_vlm:   rec["index_dirs"]["visual"] = idx_dir_vlm
    sd[str(video_path)] = rec

    parts = [f"Using outputs: {out_dir}"]
    parts.append(f"- Transcript (.srt found): {'✔' if have_srt else '✖'}")
    parts.append(f"- Transcript index: {idx_dir_trans or '(none — click Generate if needed)'}")
    parts.append(f"- Visual artifacts (csv/context/meta): {'✔' if have_vlm else '✖'}")
    parts.append(f"- Visual index: {idx_dir_vlm or '(none — click Generate if needed)'}")
    if not have_srt and not idx_dir_trans:
        parts.append("• No transcript SRT or index found. If you didn’t drop an audio file in this folder, click “Generate”.")
    parts.append(f"- transcript index: {idx_dir_trans or '(none)'}")
    parts.append(f"- visual index:     {idx_dir_vlm   or '(none)'}")
    return "\n".join(parts), sd


def do_generate(
    folder, video_path, outputs_root,
    lang, vclass, fps, device, batch, conf, iou, max_det,
    window, anchor, embed_model, embed_device, state_dict,
    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
    use_transcript: bool, use_detection: bool, use_visual: bool,   # <-- order aligned with new UI
    progress=gr.Progress(track_tqdm=True)):

    if not video_path:
        raise gr.Error("Please select a video.")
    progress(0.0, desc="Starting…")

    need_extractor = bool(use_transcript or use_detection)
    out_dir: Optional[str] = None
    idx_dir_trans = None
    idx_dir_vlm = None

    # Decide or create outputs dir
    existing = find_existing_outputs_for_video(video_path, outputs_root)
    out_dir = existing[0] if existing else None
    if not out_dir:
        stem = safe_stem(Path(video_path))
        out_dir = str(Path(outputs_root).expanduser() / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Run extractor only if SRT or detection is requested
    if need_extractor:
        srt_pre = find_any_srt(out_dir)
        # We still run extractor if detection is enabled (even if SRT already exists), so we can generate detections.* files
        must_run = (use_detection or not srt_pre)  # run if we need detections OR we need a transcript
        if must_run:
            out_dir = run_extractor(
                video_path=video_path, lang=lang, vclass=vclass, fps=(None if not fps else int(fps)),
                device=device, batch_size=int(batch), conf=float(conf), iou=float(iou), max_det=int(max_det),
                progress=progress, enable_detection=use_detection
            )
            progress(0.4, desc="Extractor finished.")

    # Build transcript index only if transcript requested and exists/was created
    if use_transcript:
        srt_path = find_any_srt(str(out_dir))
        if not srt_path:
            raise gr.Error("Transcript requested but no .srt was produced/found.")
        idx_dir_trans = build_index_from_srt(
            transcript_path=str(srt_path),
            window=int(window), anchor=anchor,
            embed_model=embed_model,
            device=(None if embed_device == "auto" else embed_device),
            progress=progress,
        )
        progress(0.7, desc="Transcript index ready.")

    # VLM pipeline (only if requested)
    if use_visual:
        have_vlm = has_visual_artifacts(out_dir)
        if not have_vlm:
            progress(0.75, desc="Visual extraction…")
            run_visual_strategy(
                video_path=video_path, outputs_dir=out_dir,
                base_url=vlm_base_url.strip(), api_key=vlm_api_key.strip(),
                model=vlm_model.strip(), profile=vlm_profile,
                max_concurrency=int(vlm_maxconc),
                stream=bool(vlm_stream), stream_mode=vlm_stream_mode,
                force=False,
            )
            progress(0.84, desc="Visual artifacts generated.")
        idx_dir_vlm = build_visual_index(
            outputs_dir=out_dir,
            embed_model=embed_model,
            embed_device=embed_device,
        )
        progress(0.9, desc="Visual index ready.")

    # Save to state
    sd = state_dict or {}
    rec = {"out_dir": out_dir, "index_dirs": {}}
    if idx_dir_trans: rec["index_dirs"]["transcript"] = idx_dir_trans
    if idx_dir_vlm:   rec["index_dirs"]["visual"] = idx_dir_vlm
    sd[str(video_path)] = rec
    progress(1.0, desc="Done.")

    status = [f"✅ Prepared in: {out_dir}"]
    status.append(f"- Transcript index: {idx_dir_trans or '(skipped)'}")
    status.append(f"- Visual index:     {idx_dir_vlm or '(skipped)'}")
    status.append(f"- Object detection: {'enabled' if use_detection else 'disabled'}")
    return "\n".join(status), sd


def _provider_cfg(provider,
                  oa_base_url, oa_api_key, oa_model,
                  an_api_key, an_model) -> Dict:
    p = (provider or "anthropic").lower()
    if p == "openai":
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
    p = (provider or "anthropic").lower()
    if p == "openai":
        if not cfg.get("oa_api_key") or not cfg.get("oa_model"):
            return "Please set OpenAI API Key and Model in the panel (Base URL optional)."
    elif p == "anthropic":
        if not cfg.get("an_api_key") or not cfg.get("an_model"):
            return "Please set Anthropic API Key and Model in the panel."
    return None


def _yield_stream_with_sanitize(
    stream_gen,
    msgs,
    radio_update,
    label_to_start,
    ctx_html_string,
    allowed_nums,
    times_by_num,
):
    """
    Consume a token stream, yield partial chat updates,
    then do a final sanitize pass and yield once more.
    """
    acc = ""
    msgs.append({"role": "assistant", "content": ""})
    for chunk in stream_gen:
        if not chunk:
            continue
        acc += chunk
        msgs[-1]["content"] = acc
        # live partial update
        yield msgs, radio_update, label_to_start, ctx_html_string

    # final sanitize + final update
    msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
    yield msgs, radio_update, label_to_start, ctx_html_string


def on_chat(
    user_msg, history,
    video_path, ctx_before, ctx_after, top_k, method, alpha,
    rerank, rerank_model, overfetch,
    provider,
    oa_base_url, oa_api_key, oa_model,
    an_api_key, an_model,
    embed_device, embed_model_override,
    enable_web, exa_api_key, exa_num_results,
    state_dict,
    source_mode
):
    """
    Generator that yields (messages, ts_radio_update, ts_map_state, ctx_html_string).
    Chatbot(type='messages'): messages must be [{'role','content'}, ...]
    Now **history-aware**: we use chat history to contextualize the query and in the final answer.
    """
    # The chat as displayed to the user
    msgs = list(history or [])

    # basic validations
    if not video_path:
        msgs.append({"role": "assistant", "content": "Please select a video first."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
        return

    cfg = _provider_cfg(provider, oa_base_url, oa_api_key, oa_model,
                        an_api_key, an_model)
    err = _validate_provider_inputs(provider, cfg)
    if err:
        msgs.append({"role": "assistant", "content": err})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
        return

    rec = (state_dict or {}).get(str(video_path))

    if not rec:
        msgs.append({"role": "assistant", "content": "No outputs mapped for this video. Click 'Use selected outputs' or 'Generate'."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
        return

    # Append the user's latest message to the visible chat
    latest_user = str(user_msg)
    if not (msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == latest_user):
        msgs.append({"role": "user", "content": latest_user})

    # Prepare a trimmed history window (EXCLUDES the latest user msg for clarity in prompts)
    history_window_for_rewrite = _trim_history_messages(msgs[:-1] or [], max_turns=10, max_chars=6000)
    history_text_for_router = _format_history_as_text(msgs[:-1] or [], max_turns=10, max_chars=6000)

    # === (1) Contextualize the question with history ===
    try:
        standalone_q = contextualize_question(
            provider=(provider or "anthropic"),
            cfg=cfg,
            history_msgs=history_window_for_rewrite,  # only prior messages
            latest_user_msg=latest_user,
        )
    except Exception:
        standalone_q = latest_user

    # === (2a) Entity-search intention? Prefer entities_scenes over RAG ===
    is_entity, entity_name = _detect_entity_intent(standalone_q)
    if is_entity and entity_name:
        # Load entities_scenes from *_context.json in this video's outputs
        out_dir = (state_dict or {}).get(str(video_path), {}).get("out_dir")
        entities_scenes, norm_map = _load_entities_scenes_from_context(out_dir) if out_dir else ({}, {})
        resolved_key = _fuzzy_pick_entity(entity_name, norm_map)

        if entities_scenes and resolved_key in entities_scenes:
            # Found: collect target scenes → build context from CSV rows
            scene_ids = sorted({int(s) for s in entities_scenes.get(resolved_key, [])})
            scene_rows = _load_scene_rows_from_csv(out_dir, scene_ids)
            hits = _hits_from_scene_rows(scene_rows)
            ctx_text = "\n\n".join(
                [f"[Scene {r['scene_id']}] {r['start_timecode']}–{r['end_timecode']}\n{r['generated_text']}" for r in scene_rows]
            )
            ctx_html_string = _ctx_md_from_hits_aggregated(hits, title=f"Scenes for entity: {resolved_key}")

            # Prepare timecode radio options
            labels = []
            label_to_start = {}
            for h in hits:
                c0 = h["context"][0]
                snippet = (c0.get("text","") or "").replace("\n"," ")
                if len(snippet) > 100: snippet = snippet[:97] + "..."
                label = f"[VLM] {c0['start_srt']} - {c0['end_srt']} | {snippet}"
                labels.append(label)
                label_to_start[label] = srt_to_seconds(c0["start_srt"])
            radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

            # Compose final answer with LLM (grounded in these scenes)
            system = (
                "Tu es un assistant vidéo. Règles STRICTES:\n"
                "• Pour chaque élément, préfixe le timecode EXACT avec **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}**.\n"
                "• {num} = scene_id s'il est fourni, sinon le numéro du passage (rank).\n"
                "• 1–2 lignes par scène. N'invente rien au-delà du contexte fourni."
                f"ALLOWED_SCENES = {json.dumps([{'num': int(r['scene_id']), 'start': r['start_timecode'], 'end': r['end_timecode'], 'src': 'VLM'} for r in scene_rows], ensure_ascii=False)}"
            )

            user_full = (
                f"Requête utilisateur: Donne toutes les scènes où «{entity_name}» apparaît.\n"
                f"Entité résolue (normalisée): {resolved_key}\n"
                f"Scenes trouvées: {scene_ids}\n\n"
                f"Contexte par scène (cartes VLM):\n{ctx_text}\n\n"
                "Tâche: Résume et liste les scènes sous forme de puces: [scene_id] HH:MM:SS,mmm–HH:MM:SS,mmm — 1 à 2 lignes utiles."
            )

            stream = llm_chat_stream(
                provider=provider, cfg=cfg,
                messages=[{"role":"system","content":system},{"role":"user","content":user_full}],
                temperature=0.1, max_tokens=2000
            )

            for out in _yield_stream_with_sanitize(
                stream,
                msgs,
                radio_update,
                label_to_start,
                ctx_html_string,
                allowed_nums,
                times_by_num
            ):
                yield out
            return

    # === (2a) LLM: detect intention & extract entities for entity_search ===
    try:
        intent_res = llm_detect_intent_entities(
            provider=(provider or "anthropic"),
            cfg=cfg,
            latest_user_msg=str(standalone_q),
            history_msgs=(history or []),
        )
    except Exception:
        intent_res = {"intent":"other","entities":[],"reason":"router failed"}

    if intent_res.get("intent") == "entity_search" and intent_res.get("entities"):
        # Load entities_scenes and delegate matching to the LLM
        out_dir = (state_dict or {}).get(str(video_path), {}).get("out_dir")
        entities_scenes, candidate_keys = _load_entities_scenes_from_context(out_dir) if out_dir else ({}, [])
        if entities_scenes and candidate_keys:
            try:
                match_res = llm_match_entities_to_keys(
                    provider=(provider or "anthropic"),
                    cfg=cfg,
                    query_entities=intent_res["entities"],
                    candidate_keys=candidate_keys,
                )
            except Exception:
                match_res = {"matches": [], "flat_keys": []}

            matched_keys: List[str] = match_res.get("flat_keys") or []
            if matched_keys:
                # Union all scene IDs for the selected keys
                scene_ids = sorted({int(s) for k in matched_keys for s in (entities_scenes.get(k) or [])})
                scene_rows = _load_scene_rows_from_csv(out_dir, scene_ids)
                hits = _hits_from_scene_rows(scene_rows)
                ctx_text = "\n\n".join(
                    [f"[Scene {r['scene_id']}] {r['start_timecode']}–{r['end_timecode']}\n{r['generated_text']}" for r in scene_rows]
                )
                ctx_html_string = _ctx_md_from_hits_aggregated(hits, title=f"Scenes for entities: {', '.join(matched_keys)}")

                # Prepare the timecode radio
                labels = []
                label_to_start = {}
                for h in hits:
                    c0 = h["context"][0]
                    snippet = (c0.get("text","") or "").replace("\n"," ")
                    if len(snippet) > 100: snippet = snippet[:97] + "..."
                    label = f"[VLM] {c0['start_srt']} - {c0['end_srt']} | {snippet}"
                    labels.append(label)
                    label_to_start[label] = srt_to_seconds(c0["start_srt"])
                radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

                # Build allow-list from the VLM scene rows
                allowed = []
                for r in scene_rows:
                    num = int(r["scene_id"])
                    allowed.append({
                        "num": num,
                        "start": r["start_timecode"],
                        "end": r["end_timecode"],
                        "src": "VLM",
                    })
                allowed_nums = {a["num"] for a in allowed}
                times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

                # Final answer grounded on those scenes
                system = (
                    "Tu es un assistant vidéo. Règles STRICTES:\n"
                    "• Pour chaque scène, affiche **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}** suivi d'une brève explication.\n"
                    "• {num} = scene_id si présent, sinon le numéro du passage (rank).\n"
                    "• N'invente rien."
                    f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
                )

                user_full = json.dumps({
                    "user_query": str(standalone_q),
                    "query_entities": intent_res["entities"],
                    "matched_keys": matched_keys,
                    "scenes": scene_ids,
                    "vlm_cards_text": ctx_text,
                }, ensure_ascii=False)

                try:
                    stream = llm_chat_stream(
                        provider=(provider or "anthropic"),
                        cfg=cfg,
                        messages=[{"role":"system","content":system},{"role":"user","content":user_full}],
                        temperature=0.1,
                        max_tokens=700,
                    )
                    for out in _yield_stream_with_sanitize(
                        stream,
                        msgs,
                        radio_update,
                        label_to_start,
                        ctx_html_string,
                        allowed_nums,
                        times_by_num
                    ):
                        yield out
                    return
                except Exception as e:
                    msgs.append({"role":"assistant","content": f"LLM error: {e}"})
                    yield msgs, radio_update, label_to_start, ctx_html_string
                    return

    # === (2) Decide which indexes to use ===
    available = (state_dict or {}).get(str(video_path), {}).get("index_dirs", {})
    have_trans = "transcript" in available
    have_visual = "visual" in available

    mode = (source_mode or "auto").lower()
    if mode == "both":
        want_transcript, want_visual = True, True
    elif mode == "transcript":
        want_transcript, want_visual = True, False
    elif mode == "visual":
        want_transcript, want_visual = False, True
    else:
        # AUTO: classify using the rewritten standalone question
        want_transcript, want_visual, _reason = auto_select_sources_from_query(standalone_q)

    # Fallbacks if selection not available
    if want_transcript and not have_trans:
        want_transcript = False
    if want_visual and not have_visual:
        want_visual = False
    if not want_transcript and not want_visual:
        # If nothing matches availability, fall back to whichever exists
        if have_trans:      want_transcript = True
        elif have_visual:   want_visual = True
        else:
            msgs.append({"role": "assistant", "content": "No RAG indexes are available for this video. Generate or load indexes first."})
            yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
            return

    idx_pairs = []
    if want_transcript and have_trans:
        idx_pairs.append((load_index(available["transcript"]), "SRT"))
    if want_visual and have_visual:
        idx_pairs.append((load_index(available["visual"]), "VLM"))

    hits, ctx_text = compile_context_blocks_multi(
        indexes=idx_pairs,
        query=str(standalone_q), top_k=int(top_k), method=method, alpha=float(alpha),
        rerank=rerank, rerank_model=rerank_model, overfetch=int(overfetch),
        ctx_before=int(ctx_before), ctx_after=int(ctx_after),
        device=(None if embed_device == "auto" else embed_device),
        embed_model_override=(None if not embed_model_override else embed_model_override)
    )

    # Build an allow-list of scene ids and times for the current answer turn
    allowed = []
    for h in hits:
        # prefer a real scene_id (VLM) else fall back to 'rank' as the scene number
        num = int(h.get("scene_id", h.get("rank", 0)) or h.get("rank", 0))
        ctx_sorted = sorted(h["context"], key=lambda c: (c.get("offset",0)!=0, c.get("offset",0)))
        main = next((c for c in ctx_sorted if c.get("offset",0)==0), ctx_sorted[0])
        allowed.append({
            "num": num,
            "start": main.get("start_srt","00:00:00,000"),
            "end":   main.get("end_srt","00:00:00,000"),
            "src":   h.get("source","srt").upper(),  # SRT/VLM
        })

    # Convenience maps for post-validation
    allowed_nums = {a["num"] for a in allowed}
    times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

    ctx_html_string = _ctx_md_from_hits_aggregated(hits, title="Retrieved passages")

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
            snippet = snippet[:97] + "..."
        source_label = format_source_label(h.get("source"))
        label = f"[{source_label}] {start_srt} - {end_srt} | {snippet}".strip()
        labels.append(label)
        label_to_start[label] = srt_to_seconds(start_srt)

    radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

    # === (3) Router: decide about web search (history-aware) ===
    explicit_flag = wants_web_search_explicit(latest_user)

    # NEW: recency detection on both the raw and rewritten question
    cy = datetime.now().year
    rec_flag1, yrs1 = wants_recent_info(latest_user, cy)
    rec_flag2, yrs2 = wants_recent_info(standalone_q, cy)
    recency_flag = rec_flag1 or rec_flag2
    years_mentioned = sorted(set(yrs1 + yrs2))

    if recency_flag and (not enable_web or not exa_api_key):
        msgs.append({"role":"assistant",
                    "content":"This looks time-sensitive (e.g., 'now/2025'). Enable web search (Exa) to fetch up-to-date info, otherwise I can only answer from the video’s content."})
        msgs[-1]["content"] = sanitize_scene_output(msgs[-1]["content"], allowed_nums, times_by_num)
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    try:
        decision = llm_decide_search(
            provider=(provider or "anthropic"),
            cfg=cfg,
            question=str(standalone_q),
            transcript_context=ctx_text,
            explicit_flag=explicit_flag,
            history_text=history_text_for_router,
            recency_flag=recency_flag,
            years_mentioned=years_mentioned,
            current_year=cy,
        )
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"Routing error: {e}"})
        msgs[-1]["content"] = sanitize_scene_output(msgs[-1]["content"], allowed_nums, times_by_num)
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    need_search = bool(decision.get("need_search"))
    search_query = decision.get("query")
    direct_answer = decision.get("answer")

    # === (4A) If no web search needed, answer using transcript + history ===
    if not need_search:
        history_for_model = _trim_history_messages(msgs[:-1], max_turns=10, max_chars=6000)
        system = (
            "Tu es un assistant qui répond en combinant:\n"
            "1) le transcript (fiable pour les propos et timestamps),\n"
            "2) l'historique de la conversation.\n"
            "RÈGLE DE FORMATAGE OBLIGATOIRE:\n"
            "• Chaque timecode cité doit être préfixé par **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}**.\n"
            "• {num} = scene_id si présent, sinon le numéro du passage (rank).\n"
            "N'invente rien au-delà du transcript."
            f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
        )
        user_full = (
            f"Dernière question (standalone):\n{standalone_q}\n\n"
            f"Transcript context (passages):\n{ctx_text}\n"
        )
        try:
            stream = llm_chat_stream(
                provider=(provider or "anthropic"),
                cfg=cfg,
                messages=[{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}],
                temperature=0.2,
                max_tokens=900,
            )
            for out in _yield_stream_with_sanitize(
                stream, msgs, radio_update, label_to_start, ctx_html_string, allowed_nums, times_by_num
            ):
                yield out
            return
        except Exception as e:
            msgs.append({"role": "assistant", "content": f"LLM error: {e}"})
            yield msgs, radio_update, label_to_start, ctx_html_string
            return


    # === (4B) Web search suggested ===
    if not search_query:
        msgs.append({"role": "assistant", "content": "La recherche web a été suggérée, mais aucune requête n'a été fournie."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    if not enable_web:
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (la recherche web est désactivée)."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    if not exa_api_key:
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (ajoutez une clé Exa pour effectuer la recherche)."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    # Show interim "searching" message
    msgs.append({"role": "assistant", "content": f"🔎 Web search query: \"{search_query}\" (running Exa…)"})
    yield msgs, radio_update, label_to_start, ctx_html_string

    # 3) Exa search
    try:
        web_hits = exa_search_with_contents(search_query, exa_api_key, num_results=int(exa_num_results))
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"Exa search failed: {e}"})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    if not web_hits:
        msgs.append({"role": "assistant", "content": "Aucun résultat web pertinent n'a été trouvé."})
        yield msgs, radio_update, label_to_start, ctx_html_string
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
        "1) l'historique de la conversation,\n"
        "2) le transcript (fiable pour propos/timestamps),\n"
        "3) des extraits web.\n"
        "RÈGLE DE FORMATAGE OBLIGATOIRE:\n"
        "• Chaque timecode cité doit être préfixé **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}** (scene_id si dispo, sinon passage rank).\n"
        "Pour le web, cite les sources en [1], [2], etc."
        f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
    )

    user_full = (
        f"Dernière question (standalone):\n{standalone_q}\n\n"
        f"Transcript context:\n{ctx_text}\n\n"
        f"Web results:\n{web_context}"
    )

    history_for_model = _trim_history_messages(msgs[:-1], max_turns=10, max_chars=6000)
    try:
        stream = llm_chat_stream(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=[{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}],
            temperature=0.2,
            max_tokens=900,
        )
        for out in _yield_stream_with_sanitize(
            stream,
            msgs,
            radio_update,
            label_to_start,
            ctx_html_string,
            allowed_nums,
            times_by_num
        ):
            yield out
        return
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"LLM error (final): {e}"})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return


import gradio as gr
from pathlib import Path

# ---------------------
# Gradio Layout
# ---------------------
with gr.Blocks(title="Agentic Video RAG Chat", fill_height=True) as demo:
    gr.Markdown("## Agentic Video RAG Chat\nScan → select video → reuse **existing outputs** or **Generate** → chat with transcript.\nOptional: let the assistant trigger **web search** via Exa when needed.\n\n**LLM Providers:** OpenAI • Anthropic\n\n**New:** answers use **RAG + chat history** for follow-ups and pronouns.")

    with gr.Row():
        with gr.Column(scale=3):
            upload_video = gr.File(
                label="Upload a video",
                file_count="single",
                file_types=["video"],
                type="filepath"
            )

            gr.HTML("""
            <style>
            /* Hide the built-in Clear button on the Chatbot header */
            #chat_box button[aria-label="Clear"] { display: none !important; }
            </style>
            """)

            folder_tb = gr.Textbox(label="Folder to scan for videos", value=str(Path.cwd() / "videos"))
            outputs_root_tb = gr.Textbox(label="Outputs folder", value=str(Path.cwd() / "outputs"))
            scan_btn = gr.Button("Scan videos")

            video_dd = gr.Dropdown(choices=[], label="Select a video")
            existing_outputs_dd = gr.Dropdown(choices=[], label="Existing outputs for this video")
            use_existing_btn = gr.Button("Use selected outputs")

            video_player = gr.Video(label="Preview", elem_id="video_preview")
            status_box = gr.Textbox(label="Status", lines=8)

            with gr.Accordion("Pipelines to run", open=True):
                use_srt_chk = gr.Checkbox(value=True, label="Transcript (SRT)")
                use_detect_chk = gr.Checkbox(value=False, label="Object detection (YOLO, etc.)")
                use_vlm_chk = gr.Checkbox(value=True, label="VLM visual cards")

            with gr.Accordion("Extraction settings (used only when you click Generate)", open=False):
                lang_dd = gr.Dropdown(choices=["EN","FR"], value="FR", label="Transcript language (Wit.ai key must exist)")
                vclass_dd = gr.Dropdown(choices=["sport","nature","screen","cctv"], value="sport", label="Detection pipeline")
                fps_tb = gr.Number(value=None, label="Override FPS (optional)")
                device_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Detector device")
                batch_tb = gr.Slider(8, 128, 64, step=8, label="Batch size")
                conf_tb = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Det conf")
                iou_tb = gr.Slider(0.1, 0.95, 0.7, step=0.05, label="NMS IoU")
                maxdet_tb = gr.Slider(50, 1000, 300, step=50, label="Max det/frame")

            with gr.Accordion("Visual extraction", open=False):
                vlm_base_url_tb = gr.Textbox(label="vLLM Base URL", value=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
                vlm_api_key_tb = gr.Textbox(label="vLLM API Key", type="password", value=os.getenv("VLLM_API_KEY", "apikey-1234"))
                vlm_model_tb = gr.Textbox(label="VLM Model", value=os.getenv("VLLM_MODEL", "OpenGVLab/InternVL3-8B"))
                vlm_profile_dd = gr.Dropdown(choices=["brand_heavy","sports","compliance","slides_docs","default"], value="brand_heavy", label="Profile")
                vlm_maxconc_tb = gr.Slider(2, 20, 8, step=1, label="Max concurrency")
                vlm_stream_chk = gr.Checkbox(value=False, label="Stream live to console (server logs)")
                vlm_stream_mode_dd = gr.Dropdown(choices=["aggregate","sequential","none"], value="none", label="Stream mode")

            with gr.Accordion("Index sources to use in Chat", open=True):
                source_mode_dd = gr.Dropdown(
                    choices=["auto", "both", "transcript", "visual"],
                    value="auto",
                    label="Index source mode"
                )


            with gr.Accordion("Index settings (also used if existing outputs lack an index)", open=False):
                window_tb = gr.Slider(3, 15, 10, step=1, label="Chunk window (seconds)")
                anchor_dd = gr.Dropdown(choices=["first","zero"], value="first", label="Window anchor")
                embed_model_tb = gr.Textbox(value="sentence-transformers/all-MiniLM-L6-v2", label="Embedding model")
                embed_device_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Embed device")

            with gr.Accordion("Web Search (Exa)", open=False):
                enable_web_chk = gr.Checkbox(value=True, label="Enable web search with Exa")
                exa_key_tb = gr.Textbox(label="Exa API Key", type="password", value=os.getenv("EXA_API_KEY", ""))
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
                    provider_dd = gr.Dropdown(choices=["openai","anthropic"], value="anthropic", label="Provider")

                    with gr.Group(visible=False) as oa_group:
                        gr.Markdown("#### OpenAI (Chat Completions)")
                        oa_base_url_tb = gr.Textbox(label="OpenAI Base URL (optional)", placeholder="https://api.openai.com")
                        oa_api_key_tb = gr.Textbox(label="OpenAI API Key", type="password")
                        oa_model_tb = gr.Textbox(label="OpenAI Model", placeholder="gpt-4o-mini or gpt-4o")

                    with gr.Group(visible=True) as an_group:
                        gr.Markdown("#### Anthropic (Messages API)")
                        an_api_key_tb = gr.Textbox(label="Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
                        an_model_tb = gr.Textbox(label="Anthropic Model", value="claude-3-5-haiku-20241022")

                with gr.Accordion("Timecodes (click to seek)", open=False):
                    ts_radio = gr.Radio(choices=[], label=None, interactive=True, visible=False)

                # Collapsible panel with the full retrieved context (all passages + windows)
                with gr.Accordion("Retrieved context (expand to view)", open=False):
                    ctx_panel = gr.HTML(value="", elem_id="ctx_md_full")

                ts_map_state = gr.State({})  # label -> start_seconds

                chat = gr.Chatbot(height=520, type="messages", elem_id="chat_box")
                chat_tb = gr.Textbox(placeholder="Ask about the video…", label="Message")
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear chat (hard)")

            state_dict = gr.State({})

    # Wiring
    scan_btn.click(fn=do_scan, inputs=[folder_tb], outputs=[video_dd, video_player])
    video_dd.change(fn=_on_select, inputs=[video_dd, outputs_root_tb], outputs=[video_player, existing_outputs_dd, status_box])

    provider_dd.change(toggle_provider_panels, inputs=[provider_dd], outputs=[oa_group, an_group])


    use_existing_btn.click(
        fn=on_use_existing,
        inputs=[
            existing_outputs_dd, video_dd, state_dict,
            # transcript extractor controls
            lang_dd, vclass_dd, fps_tb, device_dd, batch_tb, conf_tb, iou_tb, maxdet_tb,
            # visual controls
            vlm_base_url_tb, vlm_api_key_tb, vlm_model_tb, vlm_profile_dd, vlm_maxconc_tb, vlm_stream_chk, vlm_stream_mode_dd,
            # index settings
            window_tb, anchor_dd, embed_model_tb, embed_device_dd,
            use_detect_chk
        ],
        outputs=[status_box, state_dict]
    )

    upload_video.upload(
        fn=on_upload_video,
        inputs=[upload_video, folder_tb],
        outputs=[video_dd, video_player, status_box]
    )

    generate_btn.click(
        fn=do_generate,
        inputs=[
            folder_tb, video_dd, outputs_root_tb,
            lang_dd, vclass_dd, fps_tb, device_dd, batch_tb, conf_tb, iou_tb, maxdet_tb,
            window_tb, anchor_dd, embed_model_tb, embed_device_dd, state_dict,
            vlm_base_url_tb, vlm_api_key_tb, vlm_model_tb, vlm_profile_dd, vlm_maxconc_tb, vlm_stream_chk, vlm_stream_mode_dd,
            use_srt_chk, use_detect_chk, use_vlm_chk,   # <-- NEW
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
        oa_base_url, oa_api_key, oa_model,
        an_api_key, an_model,
        embed_device, embed_model_override,
        enable_web, exa_api_key, exa_num_results,
        state_dict,
        source_mode,
    ):
        for out in on_chat(
            user_msg, chat_history,
            video_path, ctx_before, ctx_after, top_k, method, alpha,
            rerank, rerank_model, overfetch,
            provider,
            oa_base_url, oa_api_key, oa_model,
            an_api_key, an_model,
            embed_device, embed_model_override,
            enable_web, exa_api_key, exa_num_results,
            state_dict,
            source_mode,
        ):
            yield out

    send_btn.click(
        _chat_send,
        inputs=[
            chat_tb, chat,
            video_dd, ctx_before_tb, ctx_after_tb, topk_tb, method_dd, alpha_tb,
            rerank_dd, rerank_model_tb, overfetch_tb,
            provider_dd,
            oa_base_url_tb, oa_api_key_tb, oa_model_tb,
            an_api_key_tb, an_model_tb,
            embed_device_q_dd, embed_model_override_tb,
            enable_web_chk, exa_key_tb, exa_num_tb,
            state_dict,
            source_mode_dd,
        ],
        outputs=[chat, ts_radio, ts_map_state, ctx_panel]
    ).then(lambda: "", None, [chat_tb])

    clear_btn.click(
        hard_clear,
        inputs=None,
        outputs=[chat, ts_radio, ts_map_state, ctx_panel, chat_tb]
    )


if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        share=True,
    )
