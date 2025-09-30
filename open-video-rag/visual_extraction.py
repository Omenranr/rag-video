#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strategy 2 pipeline (vLLM parallel async) — LIVE, Clean Streaming, and Profile system
- Global pass (sequential) with optional token streaming ("live")
- Scenes in parallel (bounded concurrency) with optional **clean** streaming that avoids interleaved token gibberish
- Context evolution modes:
   * EVOLVE_CONTEXT_MODE = "none"    -> all scenes share global context (fastest)
   * EVOLVE_CONTEXT_MODE = "batched" -> process scenes in waves; evolve between waves
- Flexible Profile management with validation, CLI/env overrides, and custom packs

USAGE (examples)
----------------
python strategy2_live_profiles.py --video video_3.mp4 --profile brand_heavy --stream --stream-mode aggregate
python strategy2_live_profiles.py --video video_3.mp4 --profile sports --packs products --exclude-packs ocr --max-concurrency 8 --stream
python strategy2_live_profiles.py --video video_3.mp4 --custom-profiles profiles.json --profile my_profile --stream --stream-mode sequential

Notes
-----
- **aggregate** streaming prints sentence-by-sentence updates per scene, prefixed by scene id. Output is readable even with high concurrency.
- **sequential** streaming prints only ONE scene per wave (the first), others run non-streaming in parallel → zero interleaving.
- You can also dump live text per scene to files via `--stream-dump-dir`.
"""

import os, re, csv, json, math, time, sys, base64, argparse
import contextlib, io
import datetime as dt
import logging
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from decord import VideoReader, cpu

import asyncio
from openai import AsyncOpenAI

# ----------------------------
# vLLM / OpenAI-compatible endpoint config (can be overridden by env/CLI)
# ----------------------------
DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY  = os.getenv("VLLM_API_KEY", "apikey-1234")
DEFAULT_MODEL    = os.getenv("VLLM_MODEL", "OpenGVLab/InternVL3-8B")

# Client is created later after parsing CLI so overrides apply
aclient: Optional[AsyncOpenAI] = None

# ----------------------------
# Defaults (overridable via CLI)
# ----------------------------
VIDEO_PATH = "video_3.mp4"
SCENE_SECONDS = 10

GLOBAL_SAMPLE_SEGMENTS = 16
GLOBAL_FRAMES_PER_SEGMENT = 1

SCENE_SEGMENTS = 10
SCENE_FRAMES_PER_SEGMENT = 1

INPUT_SIZE = 448
GRID_COLS = 4
JPEG_QUALITY = 90

# Generation params
GEN_MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9

# Parallelism
MAX_CONCURRENCY = 10          # number of concurrent vLLM requests
EVOLVE_CONTEXT_MODE = "batched"  # "none" or "batched"
BATCH_SIZE = 30               # used when EVOLVE_CONTEXT_MODE="batched"

# --- Profile system ---
# Concept Packs (dynamic extraction) — you can register more at runtime (JSON file or code)
PACK_CHECKLISTS = {
    "entities_basic": """- People (counts/roles), organizations, locations/landmarks, objects.\n- Keep names exact if on-screen; otherwise short generic labels.""",
    "ocr": """- List ALL legible text (case preserved); mention languages/scripts if mixed.""",
    "actions": """- Chronological 0–2s, 2–4s… terse verb phrases.""",
    "tags": """- 5–12 short keywords (topics, domains).""",
    "shot": """- camera_motion, angle, transitions (if any).""",
    "brands_logos": """- Brands/logos: name, where seen (jersey/billboard/UI/packaging/other), exact OCR if present.\n- If uncertain, give up to 3 visual guesses (icon/colors) with confidence 0–1.""",
    "products": """- Product category + model/series; packaging/claims (e.g., '4K', '0% sugar').""",
    "sports_core": """- Teams/colors, jerseys (#), scoreboard/time, play types or events.""",
    "pii_compliance": """- Flag license plates, badges, phone numbers, QR/barcodes (do NOT transcribe PII).""",
    "safety": """- Coarse safety: PPE present? risky acts? NSFW/violence (coarse labels only).""",
    "docs_slides_ui": """- Slides/Docs/UI: titles, charts/tables presence, app/tool names, code/terminal on screen.""",
}

@dataclass
class Profile:
    name: str
    packs: List[str] = field(default_factory=lambda: [
        "entities_basic", "ocr", "actions", "tags", "shot"
    ])
    # Optional generation overrides per profile
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None

# Built-in profiles
PROFILES: Dict[str, Profile] = {
    "default": Profile("default", ["entities_basic", "ocr", "actions", "tags", "shot"]),
    "brand_heavy": Profile("brand_heavy", [
        "entities_basic", "ocr", "actions", "tags", "shot",
        "brands_logos", "products"
    ], temperature=0.6),
    "sports": Profile("sports", [
        "entities_basic", "ocr", "actions", "tags", "shot",
        "sports_core"
    ]),
    "compliance": Profile("compliance", [
        "entities_basic", "ocr", "actions", "tags", "shot",
        "pii_compliance", "safety"
    ]),
    "slides_docs": Profile("slides_docs", [
        "entities_basic", "ocr", "actions", "tags", "shot",
        "docs_slides_ui"
    ], temperature=0.5, top_p=0.8),
}

# Allow env to pick default profile
DEFAULT_PROFILE_NAME = os.getenv("VIDEO_RAG_PROFILE", "brand_heavy")

# ----------------------------
# Prompts
# ----------------------------
GLOBAL_PROMPT = (
    "You are analyzing sampled frames from the WHOLE video. Provide:\n"
    "- Video type/genre (e.g., sports/soccer match highlight, vlog, screencast, ad, tutorial, etc.)\n"
    "- Accurate summary of the video (4-5 sentences)\n"
    "Return sections: [TYPE], [SUMMARY]"
)

def build_scene_prompt(global_context_text: str,
                       scene_id: int,
                       start_s: float,
                       end_s: float,
                       video_name: str,
                       start_tc: str,
                       end_tc: str,
                       profile: Profile) -> str:
    # Build checklist from selected concept packs
    checklist_lines = []
    for p in profile.packs:
        desc = PACK_CHECKLISTS.get(p)
        if desc:
            checklist_lines.append(f"[{p}]")
            checklist_lines.append(desc)
    checklist = "\n".join(checklist_lines) if checklist_lines else "(none)"

    # Concept extraction schema (text, simple to parse)
    extraction_schema = """
# Concept Extractions (fill per schema; use [] when none)
brands_logos:
  - name: <string>
    where: [jersey|billboard|sign|ui|packaging|other]
    ocr_text: <string|empty>
    candidates_if_uncertain: [<string>, <string>, <string>]
    confidence: <0.0-1.0>

products:
  - category: <string>
    model_or_series: <string|empty>
    attributes: [<short>]

pii:
  - type: <plate|badge|phone|qr|barcode|other>
    present: <true|false>

safety:
  - nsfw: <none|mild|explicit>
  - violence: <none|mild|graphic>
  - risky_act: <none|present>

sports:
  - teams: [<string>]
  - jerseys_visible: [<#>]
  - scoreboard_text: <string|empty>
  - phase_or_clock: <string|empty>

docs_ui:
  - has_slides_or_docs: <true|false>
  - has_charts_or_tables: <true|false>
  - app_or_tool_names: [<string>]
""".strip()

    return f"""
You will receive frames from a ~10-second SCENE of a video.
Use the GLOBAL CONTEXT to disambiguate when helpful, but do not contradict the visible frames.

GLOBAL CONTEXT (aggregated; may include inferred type/summary/entities/actions):
{global_context_text.strip()}

CONCEPT CHECKLIST (cover these precisely; be concise):
{checklist}

TASK:
Return a RAG SCENE CARD in plain text (no code fences). Keep it factual and concise.
---
Meta:
- video_name: {video_name}
- scene_id: {scene_id}
- start_sec: {start_s:.3f}
- end_sec: {end_s:.3f}
- start_timecode: {start_tc}
- end_timecode: {end_tc}

Description:
- description: detailed accurate and factual context capturing all relevant informations of the video <4-5 lines of who/what/where and context>
- actions_chronological:
  - <t≈ +0-2s: action phrase>
  - <t≈ +2-4s: action phrase>
  - <...>

OnScreenText:
- lines:
  - <exact OCR line 1>
  - <exact OCR line 2>
  - ...

Entities:
- items:
  - type: <person/team/logo/brand/location/object/number/other>
    name_or_value: <best guess or exact text>
    attributes: [<short attrs like jersey #, color, role, number>]
    confidence: <0.0-1.0>
  - ...

Tags:
- scene_tags: [<short keywords like teams, brands, numbers, jersey ids, location>, ...]

Shot:
- camera_motion: [<pan/tilt/handheld/static/zoom>, ...]
- camera_angle: [<wide/close-up/top-down>, ...]
- cuts_or_transitions: [<if any>, ...]

{extraction_schema}

Confidence:
- overall: <0.0-1.0>
- ocr: <0.0-1.0>
- entity_detection: <0.0-1.0>

NOTES:
- For OnScreenText, list ALL legible text exactly as written (preserve case).
- Prefer short bullets; keep chronology coarse (0–2s, 2–4s, etc.).
- If unknown or not visible, write 'unknown' or use empty lists [].
- Return ONLY the card content above (no extra commentary).
""".strip()

# ----------------------------
# Logging & Clean Live reporter
# ----------------------------

def _setup_logger(name: str = "video_rag", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    return logger

LOG_CONTEXT_CHARS = 1200       # max chars to show for context blocks
LOG_SCENE_PREVIEW_N = 0        # set >0 to preview per-scene generations (chars)

@contextlib.contextmanager
def _silence_stderr():
    # Jupyter-safe stderr silencer for ffmpeg/decord spew
    saved_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = saved_stderr

LOGGER = _setup_logger()

class LiveReporter:
    """Serialize prints across concurrent tasks and flush by sentence or size.

    Avoids interleaved token soup by buffering per-stream and printing whole
    sentences (or chunks) with a scene/global prefix.
    """
    SENTENCE_RE = re.compile(r"([\s\S]*?[\.!?](?:\s|$))")

    def __init__(self, *, enabled: bool = True, throttle_ms: int = 200,
                 min_chars: int = 160, dump_dir: Optional[str] = None):
        self.enabled = enabled
        self.queue: asyncio.Queue = asyncio.Queue()
        self.buffers: Dict[str, str] = {}
        self.last_print: Dict[str, float] = {}
        self.throttle = throttle_ms / 1000.0
        self.min_chars = max(1, int(min_chars))
        self.dump_dir = dump_dir
        self._printer_task: Optional[asyncio.Task] = None
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    async def start(self):
        if not self.enabled: return
        if self._printer_task: return
        self._printer_task = asyncio.create_task(self._printer())

    async def stop(self):
        if not self.enabled: return
        await self.queue.put(("__STOP__", None, None))
        if self._printer_task:
            await self._printer_task

    async def emit(self, stream_id: str, kind: str, payload: Optional[str] = None):
        if not self.enabled: return
        await self.queue.put((kind, stream_id, payload))

    def _extract_ready(self, sid: str) -> str:
        buf = self.buffers.get(sid, "")
        if not buf:
            return ""
        # Prefer full sentences
        ready = []
        consumed = 0
        while True:
            m = self.SENTENCE_RE.match(buf[consumed:])
            if not m:
                break
            chunk = m.group(1)
            consumed += len(chunk)
            ready.append(chunk)
            # stop at first sentence if buffer is short
            if len("".join(ready)) >= self.min_chars:
                break
        if not ready and len(buf) >= self.min_chars:
            # no sentence end yet; flush by size
            ready_text = buf[:self.min_chars]
            self.buffers[sid] = buf[self.min_chars:]
            return ready_text
        elif ready:
            ready_text = "".join(ready)
            self.buffers[sid] = buf[consumed:]
            return ready_text
        return ""

    def _dump(self, sid: str, text: str):
        if not self.dump_dir or not text:
            return
        path = os.path.join(self.dump_dir, f"{sid}.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    async def _printer(self):
        while True:
            kind, sid, payload = await self.queue.get()
            if kind == "__STOP__":
                # flush all remaining buffers
                for sid2, rest in list(self.buffers.items()):
                    txt = rest.strip()
                    if txt:
                        LOGGER.info(f"[live:{sid2}] {txt}")
                        self._dump(sid2, txt)
                        self.buffers[sid2] = ""
                break
            now = time.time()
            if kind == "start":
                LOGGER.info(f"[live:{sid}] ▶️  streaming started")
                self.buffers.setdefault(sid, "")
                self.last_print[sid] = 0.0
            elif kind == "update":
                if payload:
                    self.buffers[sid] = self.buffers.get(sid, "") + payload
                # flush by sentence or size, but throttle prints
                if (now - self.last_print.get(sid, 0.0)) >= self.throttle:
                    out = self._extract_ready(sid)
                    if out:
                        oneline = out.replace("\n", " ").strip()
                        if oneline:
                            LOGGER.info(f"[live:{sid}] {oneline}")
                            self._dump(sid, out)
                            self.last_print[sid] = now
            elif kind == "end":
                # print whatever remains for this stream
                tail = self.buffers.get(sid, "").strip()
                if tail:
                    LOGGER.info(f"[live:{sid}] {tail}")
                    self._dump(sid, tail)
                LOGGER.info(f"[live:{sid}] ✅ done")
                self.buffers[sid] = ""

# ----------------------------
# Utilities
# ----------------------------

def seconds_to_tc(t):
    hours = int(t // 3600); minutes = int((t % 3600) // 60); seconds = t % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=32, frames_per_segment=1):
    start, end = bound if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    if end_idx <= start_idx:
        end_idx = min(start_idx + max(1, int(fps)), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    idxs = []
    for idx in range(num_segments):
        seg_start = start_idx + seg_size * idx
        for j in range(frames_per_segment):
            pos = seg_start + (seg_size * (j + 0.5) / frames_per_segment)
            idxs.append(int(pos))
    return np.clip(idxs, start_idx, end_idx)

def load_video_frames(video_path, bound=None, image_size=448,
                      num_segments=32, frames_per_segment=1) -> Tuple[List[Image.Image], int, float, int, int]:
    with _silence_stderr():
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        idxs = get_frame_indices(bound, fps, max_frame, first_idx=0,
                                 num_segments=num_segments, frames_per_segment=frames_per_segment)
        frames = [Image.fromarray(vr[int(i)].asnumpy()).convert('RGB') for i in idxs]
        width, height = vr[0].shape[1], vr[0].shape[0]
    return frames, len(idxs), fps, width, height

def _square_letterbox(img: Image.Image, size=INPUT_SIZE, fill=(0,0,0)) -> Image.Image:
    img = img.copy()
    img.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    x = (size - img.width)//2
    y = (size - img.height)//2
    canvas.paste(img, (x, y))
    return canvas

def tile_frames_pil(frames: List[Image.Image], cols=GRID_COLS, size=INPUT_SIZE) -> Image.Image:
    if not frames:
        raise RuntimeError("No frames to tile.")
    rows = math.ceil(len(frames) / cols)
    canvas = Image.new("RGB", (cols*size, rows*size), (0,0,0))
    for idx, img in enumerate(frames[:rows*cols]):
        r, c = divmod(idx, cols)
        canvas.paste(_square_letterbox(img, size=size), (c*size, r*size))
    return canvas

def pil_to_data_url_jpeg(img: Image.Image, quality=JPEG_QUALITY) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def chunk_bounds(duration_sec, chunk_sec):
    n = math.ceil(duration_sec / chunk_sec)
    return [(i * chunk_sec, min((i + 1) * chunk_sec, duration_sec))
            for i in range(n) if min((i + 1) * chunk_sec, duration_sec) > i * chunk_sec]

def _filesize(path: str) -> str:
    try:
        sz = os.path.getsize(path)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if sz < 1024.0: return f"{sz:,.2f} {unit}"
            sz /= 1024.0
    except Exception:
        return "n/a"
    return "n/a"

# ----------------------------
# Context state & parsing
# ----------------------------

def init_context_state() -> Dict[str, Any]:
    return {
        "video_type": None,
        "summary": None,
        "style": None,
        "entities_freq": {},
        "entity_types": {},
        "actions_freq": {},
        "tags_freq": {},
        "ocr_vocab": {},
        "scenes_seen": 0,
        "brand_hits_freq": {},
    }

def normalize_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def parse_scene_card(card: str) -> Dict[str, Any]:
    sections = {
        "Meta": r"Meta:\s*(.*?)\n\n",
        "Description": r"Description:\s*(.*?)\n\n",
        "OnScreenText": r"OnScreenText:\s*(.*?)\n\n",
        "Entities": r"Entities:\s*(.*?)\n\n",
        "Tags": r"Tags:\s*(.*?)\n\n",
        "Shot": r"Shot:\s*(.*?)\n\n",
        "Confidence": r"Confidence:\s*(.*)$",
    }
    out = {}
    for key, pat in sections.items():
        m = re.search(pat, card, re.DOTALL | re.IGNORECASE)
        out[key] = m.group(1).strip() if m else ""
    return out

def extract_entity_names(card_text: str) -> List[str]:
    """
    Parse the Entities block of a scene card and return a list of normalized
    entity names/values found in that scene.
    """
    ent_block = parse_scene_card(card_text).get("Entities", "")
    ent_items = []
    current = {}

    for raw_line in ent_block.splitlines():
        line = raw_line.strip()
        if line.startswith("- ") and "type:" in line:
            if current:
                ent_items.append(current); current = {}
            current = {"type": line.split("type:", 1)[1].strip()}
        elif "name_or_value:" in line:
            current["name_or_value"] = line.split("name_or_value:", 1)[1].strip()
        elif line.startswith("- ") and current:
            ent_items.append(current); current = {}

    if current:
        ent_items.append(current)

    names = []
    for ent in ent_items:
        name = ent.get("name_or_value", "").strip()
        if name:
            names.append(normalize_token(name))
    return names

def parse_bullets(block: str) -> List[str]:
    items = []
    for line in block.splitlines():
        line = line.strip()
        if re.match(r"^[-•] ", line) or re.match(r"^\d+\.", line):
            items.append(re.sub(r"^[-•]\s*", "", line))
        elif line.startswith("  - "):
            items.append(line[4:])
        elif line.startswith("- "):
            items.append(line[2:])
    return [s.strip() for s in items if s.strip()]

def update_context_from_scene(state: Dict[str, Any], card_text: str, global_text_hint: str = ""):
    parsed = parse_scene_card(card_text)
    # OCR vocab
    ocr_block = parsed.get("OnScreenText", "")
    ocr_lines = [re.sub(r"^-/\s*", "", l.strip()) for l in ocr_block.splitlines() if l.strip().startswith("-")]
    for ln in ocr_lines:
        for tok in re.findall(r"[A-Za-z0-9:#@\-_/]+", ln):
            nt = normalize_token(tok)
            state["ocr_vocab"][nt] = state["ocr_vocab"].get(nt, 0) + 1
    # Actions
    desc_block = parsed.get("Description", "")
    act_match = re.search(r"actions_chronological:\s*(.*)", desc_block, re.DOTALL | re.IGNORECASE)
    if act_match:
        for a in parse_bullets(act_match.group(1)):
            na = normalize_token(re.sub(r"^t≈\s*[^:]+:\s*", "", a))
            if na:
                state["actions_freq"][na] = state["actions_freq"].get(na, 0) + 1
    # Entities
    ent_block = parsed.get("Entities", ""); ent_items = []; current = {}
    for line in ent_block.splitlines():
        l = line.strip()
        if l.startswith("- ") and "type:" in l:
            if current: ent_items.append(current); current = {}
            current = {"type": l.split("type:", 1)[1].strip()}
        elif "name_or_value:" in l:
            current["name_or_value"] = l.split("name_or_value:", 1)[1].strip()
        elif "attributes:" in l:
            attrs = re.findall(r"\[(.*)\]", l)
            current["attributes"] = attrs[0].split(",") if attrs else []
        elif l.startswith("- ") and current:
            ent_items.append(current); current = {}
    if current: ent_items.append(current)
    for ent in ent_items:
        name = normalize_token(ent.get("name_or_value", ""))
        typ = normalize_token(ent.get("type", ""))
        if name:
            state["entities_freq"][name] = state["entities_freq"].get(name, 0) + 1
            if typ: state["entity_types"][name] = typ
    # Tags
    tags_block = parsed.get("Tags", "")
    for line in tags_block.splitlines():
        l = line.strip()
        m = re.search(r"scene_tags:\s*\[(.*)\]", l, re.IGNORECASE)
        if m:
            for t in [normalize_token(x).strip() for x in m.group(1).split(",")]:
                if t: state["tags_freq"][t] = state["tags_freq"].get(t, 0) + 1
    state["scenes_seen"] += 1

def confront_facts(state: Dict[str, Any], global_text: str) -> Dict[str, Any]:
    def top_k(d, k=10):
        return sorted(d.items(), key=lambda x: (-x[1], x[0]))[:k]
    top_entities = top_k(state["entities_freq"], 15)
    top_actions = top_k(state["actions_freq"], 15)
    top_tags = top_k(state["tags_freq"], 15)
    top_brands = sorted(state.get("brand_hits_freq", {}).items(), key=lambda x: (-x[1], x[0]))[:15]
    likely_type = None
    m = re.search(r"\[TYPE\]\s*(.*)", global_text, re.IGNORECASE)
    if m: likely_type = m.group(1).strip()
    ctx_lines = []
    if likely_type: ctx_lines.append(f"Likely video type: {likely_type}")
    if state["summary"]:
        ctx_lines.append(f"Global summary: {state['summary']}")
    else:
        m2 = re.search(r"\[SUMMARY\]\s*(.*?)(?:\n\[|$)", global_text, re.IGNORECASE | re.DOTALL)
        if m2: ctx_lines.append("Global summary: " + " ".join(m2.group(1).split()))
    if top_entities:
        ctx_lines.append("Frequent entities: " + ", ".join([f"{k} (x{v})" for k, v in top_entities]))
    if top_actions:
        ctx_lines.append("Frequent actions: " + ", ".join([f"{k} (x{v})" for k, v in top_actions]))
    if top_tags:
        ctx_lines.append("Frequent tags: " + ", ".join([f"{k} (x{v})" for k, v in top_tags]))
    if top_brands:
        ctx_lines.append("Frequent brands: " + ", ".join([f"{k} (x{v})" for k, v in top_brands]))
    return {"likely_type": likely_type, "context_text": "\n".join(ctx_lines) or global_text}

# ----------------------------
# Async vLLM call (with optional clean streaming)
# ----------------------------
async def async_chat_vllm_with_images(prompt: str, data_urls: List[str], *,
                                      stream_id: str,
                                      reporter: LiveReporter,
                                      temperature=GEN_TEMPERATURE,
                                      max_tokens=GEN_MAX_NEW_TOKENS,
                                      top_p=GEN_TOP_P,
                                      stream: bool = False) -> str:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    if stream:
        await reporter.emit(stream_id, "start")
        final_text_parts: List[str] = []
        stream_resp = await aclient.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream_resp:  # type: ignore[func-returns-value]
            # Try OpenAI-compatible delta
            chs = getattr(chunk, 'choices', [])
            if chs:
                delta = getattr(chs[0], 'delta', None)
                piece = getattr(delta, 'content', None) if delta is not None else None
                if piece is None:
                    # fallback to provider variations
                    piece = getattr(chs[0], 'text', None) or getattr(chs[0], 'content', None)
                if piece:
                    final_text_parts.append(piece)
                    await reporter.emit(stream_id, "update", piece)
        await reporter.emit(stream_id, "end")
        return ("".join(final_text_parts)).strip()

    # Non-streaming path (simple)
    resp = await aclient.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    text = (resp.choices[0].message.content or "").strip()
    if LOG_SCENE_PREVIEW_N > 0:
        LOGGER.info(f"[live:{stream_id}] preview: {text[:LOG_SCENE_PREVIEW_N]}{' …' if len(text) > LOG_SCENE_PREVIEW_N else ''}")
    return text

# --- Parse concept blocks out of the card text (robust-ish, text-based) ---
_SECTION_ORDER = ["brands_logos", "products", "pii", "safety", "sports", "docs_ui", "Confidence"]

def _extract_section(card: str, name: str) -> str:
    # Grab text from "name:" line down to next known section header or end
    pattern = rf"(?mi)^{name}\s*:\s*\n(.*?)(?=^(?:{'|'.join(map(re.escape, _SECTION_ORDER))})\s*:|\Z)"
    m = re.search(pattern, card, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def parse_concept_extractions(card: str) -> Dict[str, Any]:
    return {
        "brands_logos_raw": _extract_section(card, "brands_logos"),
        "products_raw": _extract_section(card, "products"),
        "pii_raw": _extract_section(card, "pii"),
        "safety_raw": _extract_section(card, "safety"),
        "sports_raw": _extract_section(card, "sports"),
        "docs_ui_raw": _extract_section(card, "docs_ui"),
    }

_BRAND_NAME_LINE = re.compile(r"(?mi)^\s*-\s*name:\s*(.+)$")
_ENTITY_BRAND_LINE = re.compile(r"(?mi)^\s*-\s*type:\s*(brand|logo)\s*$")
_ENTITY_NAME_LINE  = re.compile(r"(?mi)^\s*name_or_value:\s*(.+)$")

def derive_brand_hits(card: str, concepts: Dict[str, Any]) -> List[str]:
    hits = set()

    # 1) From concept section 'brands_logos'
    raw = concepts.get("brands_logos_raw", "")
    for m in _BRAND_NAME_LINE.finditer(raw):
        name = m.group(1).strip()
        if name and name.lower() != "<string>":
            hits.add(name)

    # 2) From Entities section when type is brand or logo
    ent_block = parse_scene_card(card).get("Entities", "")
    last_was_brand = False
    for line in ent_block.splitlines():
        if _ENTITY_BRAND_LINE.search(line):
            last_was_brand = True
        elif last_was_brand:
            m = _ENTITY_NAME_LINE.search(line)
            if m:
                name = m.group(1).strip()
                if name:
                    hits.add(name)
                last_was_brand = False
        else:
            last_was_brand = False

    # Basic normalization
    normed = []
    for h in hits:
        t = re.sub(r"\s+", " ", h).strip()
        if t:
            normed.append(t)
    # Dedup, keep stable order
    seen = set(); out = []
    for t in normed:
        k = t.lower()
        if k not in seen:
            seen.add(k); out.append(t)
    return out

# ----------------------------
# Profile helpers
# ----------------------------

def load_custom_profiles(path: Optional[str]) -> Dict[str, Profile]:
    if not path: return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Custom profiles JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Profile] = {}
    for name, spec in data.items():
        packs = spec.get("packs", [])
        out[name] = Profile(
            name=name,
            packs=packs,
            temperature=spec.get("temperature"),
            top_p=spec.get("top_p"),
            max_new_tokens=spec.get("max_new_tokens"),
        )
    return out

def resolve_profile(name: str,
                    extra_packs: List[str],
                    exclude_packs: List[str],
                    custom_profiles: Dict[str, Profile]) -> Profile:
    catalog = {**PROFILES, **custom_profiles}
    if name not in catalog:
        raise KeyError(f"Unknown profile '{name}'. Available: {', '.join(sorted(catalog.keys()))}")
    base = catalog[name]
    packs = list(dict.fromkeys(base.packs + extra_packs))  # keep order, dedupe
    packs = [p for p in packs if p not in exclude_packs]
    # Validate
    unknown = [p for p in packs if p not in PACK_CHECKLISTS]
    if unknown:
        LOGGER.warning(f"Profile '{name}' includes unknown packs: {unknown} (they will be ignored)")
        packs = [p for p in packs if p in PACK_CHECKLISTS]
    return Profile(
        name=base.name,
        packs=packs,
        temperature=base.temperature,
        top_p=base.top_p,
        max_new_tokens=base.max_new_tokens,
    )

# ----------------------------
# Scene task (one scene → frames → tile → call → result)
# ----------------------------
async def process_one_scene(scene_idx: int,
                            start_s: float,
                            end_s: float,
                            fps_fallback: float,
                            video_name: str,
                            video_path: str,
                            global_context_text: str,
                            profile: Profile,
                            reporter: LiveReporter,
                            stream: bool) -> Dict[str, Any]:
    # 1) Extract frames (sync CPU-bound) in a thread; avoids blocking event loop
    frames, sampled, fps_local, w, h = await asyncio.to_thread(
        load_video_frames,
        video_path,
        (start_s, end_s),
        INPUT_SIZE,
        SCENE_SEGMENTS,
        SCENE_FRAMES_PER_SEGMENT
    )
    # 2) Tile & encode (also off-thread)
    scene_grid = await asyncio.to_thread(tile_frames_pil, frames, GRID_COLS, INPUT_SIZE)
    scene_url = await asyncio.to_thread(pil_to_data_url_jpeg, scene_grid, JPEG_QUALITY)
    # 3) Prompt
    prompt = build_scene_prompt(
        global_context_text=global_context_text,
        scene_id=scene_idx,
        start_s=start_s,
        end_s=end_s,
        video_name=video_name,
        start_tc=seconds_to_tc(start_s),
        end_tc=seconds_to_tc(end_s),
        profile=profile,
    )
    # 4) vLLM call (maybe streaming)
    temp = profile.temperature if profile.temperature is not None else GEN_TEMPERATURE
    topp = profile.top_p if profile.top_p is not None else GEN_TOP_P
    max_new = profile.max_new_tokens if profile.max_new_tokens is not None else GEN_MAX_NEW_TOKENS

    text = await async_chat_vllm_with_images(
        prompt, [scene_url],
        stream_id=f"scene-{scene_idx}",
        reporter=reporter,
        temperature=temp,
        max_tokens=max_new,
        top_p=topp,
        stream=stream,
    )

    # --- Parse concept blocks & brand hits ---
    concepts = parse_concept_extractions(text)
    brand_hits = derive_brand_hits(text, concepts)

    # Update brand hit counts (global context)
    for b in brand_hits:
        k = normalize_token(b)
        context_state["brand_hits_freq"][k] = context_state["brand_hits_freq"].get(k, 0) + 1

    return {
        "video_name": video_name,
        "video_path": os.path.abspath(video_path),
        "scene_id": scene_idx,
        "start_sec": round(float(start_s), 3),
        "end_sec": round(float(end_s), 3),
        "start_timecode": seconds_to_tc(start_s),
        "end_timecode": seconds_to_tc(end_s),
        "scene_duration_sec": round(float(end_s - start_s), 3),
        "fps": fps_local or fps_fallback,
        "width": w,
        "height": h,
        "sampled_frames_for_scene": sampled,
        "generated_text": text,

        # new fields (JSON-serializable for CSV/JSON)
        "brand_hits": brand_hits,
        "brands_logos_raw": concepts["brands_logos_raw"],
        "products_raw": concepts["products_raw"],
        "pii_raw": concepts["pii_raw"],
        "safety_raw": concepts["safety_raw"],
        "sports_raw": concepts["sports_raw"],
        "docs_ui_raw": concepts["docs_ui_raw"],
    }

# ----------------------------
# Main pipeline (async)
# ----------------------------
async def process_strategy2_async(video_path: str,
                                  *,
                                  profile: Profile,
                                  stream: bool,
                                  stream_mode: str,
                                  max_concurrency: int,
                                  evolve_mode: str,
                                  batch_size: int,
                                  stream_throttle_ms: int,
                                  stream_min_chars: int,
                                  stream_dump_dir: Optional[str]) -> Tuple[str, str, str]:
    global aclient, MODEL, context_state
    t0 = time.perf_counter()
    assert os.path.isfile(video_path), f"Video not found: {video_path}"
    LOGGER.info(f"Opening video: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps()); total_frames = len(vr)
    duration = total_frames / fps if fps > 0 else 0.0
    width, height = vr[0].shape[1], vr[0].shape[0]
    LOGGER.info(f"Video | fps={fps:.3f} frames={total_frames} dur={duration:.3f}s res={width}x{height}")

    base = os.path.splitext(os.path.basename(video_path))[0]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{base}_{stamp}.csv"
    meta_name = f"{base}_{stamp}_metadata.txt"
    ctx_name = f"{base}_{stamp}_context.json"

    reporter = LiveReporter(
        enabled=(stream_mode != "none" and stream),
        throttle_ms=stream_throttle_ms,
        min_chars=stream_min_chars,
        dump_dir=stream_dump_dir,
    )
    await reporter.start()

    # GLOBAL pass (sequential)
    LOGGER.info("Sampling frames for GLOBAL description...")
    global_frames, global_n, _, _, _ = load_video_frames(
        video_path,
        (0, duration),
        INPUT_SIZE,
        GLOBAL_SAMPLE_SEGMENTS,
        GLOBAL_FRAMES_PER_SEGMENT
    )
    global_grid = tile_frames_pil(global_frames, GRID_COLS, INPUT_SIZE)
    global_url = pil_to_data_url_jpeg(global_grid, JPEG_QUALITY)
    LOGGER.info("Generating GLOBAL description via vLLM...")

    # Global generation (maybe streaming)
    global_text = await async_chat_vllm_with_images(
        GLOBAL_PROMPT, [global_url],
        stream_id="global",
        reporter=reporter,
        temperature=profile.temperature or GEN_TEMPERATURE,
        max_tokens=profile.max_new_tokens or GEN_MAX_NEW_TOKENS,
        top_p=profile.top_p or GEN_TOP_P,
        stream=stream and (stream_mode != "none"),
    )

    # Extract & log global TYPE/SUMMARY
    m_type = re.search(r"\[TYPE\]\s*(.*)", global_text, re.IGNORECASE)
    m_sum  = re.search(r"\[SUMMARY\]\s*(.*?)(?:\n\[|$)", global_text, re.IGNORECASE | re.DOTALL)
    type_text = (m_type.group(1).strip() if m_type else "").strip()
    sum_text  = (" ".join(m_sum.group(1).split()) if m_sum else global_text).strip()

    LOGGER.info("\n" + _box("GLOBAL TYPE", type_text or "(unknown)"))
    LOGGER.info("\n" + _box("GLOBAL SUMMARY", sum_text or "(none)"))

    # Seed context from global text
    context_state = init_context_state()
    if m_type: context_state["video_type"] = " ".join(m_type.group(1).split())
    if m_sum: context_state["summary"] = " ".join(m_sum.group(1).split())

    # Scene bounds
    bounds = chunk_bounds(duration, SCENE_SECONDS)
    total_scenes = len(bounds)
    LOGGER.info(f"{total_scenes} scene(s) planned). Streaming mode: {stream_mode}")

    rows: List[Dict[str, Any]] = []

    # Helper to run a wave of scenes concurrently with a shared context_text
    async def run_wave(scene_range: List[Tuple[int, float, float]], context_text: str):
        sem = asyncio.Semaphore(max_concurrency)
        first_scene_id = scene_range[0][0] if scene_range else -1

        async def wrapped(scene_id, s, e):
            async with sem:
                LOGGER.info(f"→ submit scene {scene_id}/{total_scenes} [{seconds_to_tc(s)}–{seconds_to_tc(e)}]")
                # Decide streaming policy per scene
                if stream and stream_mode == "sequential":
                    stream_this = (scene_id == first_scene_id)
                elif stream and stream_mode == "aggregate":
                    stream_this = True
                else:
                    stream_this = False
                res = await process_one_scene(scene_id, s, e, fps, base, video_path, context_text, profile, reporter, stream_this)
                LOGGER.info(f"✓ done   scene {scene_id}/{total_scenes}")
                return res
        tasks = [asyncio.create_task(wrapped(i, s, e)) for (i, s, e) in scene_range]
        return await asyncio.gather(*tasks, return_exceptions=False)

    if evolve_mode == "none":
        # One big wave, shared context (fastest)
        confronted = confront_facts(context_state, global_text)
        wave = [(i, s, e) for i, (s, e) in enumerate(bounds, 1)]
        wave_rows = await run_wave(wave, confronted["context_text"])
        rows.extend(wave_rows)
        # Post-hoc aggregation (update context based on all rows)
        for r in rows:
            update_context_from_scene(context_state, r["generated_text"], global_text_hint=global_text)

    elif evolve_mode == "batched":
        # Process in waves (batch_size) and evolve context in between
        idx_bounds = [(i, b[0], b[1]) for i, b in enumerate(bounds, 1)]
        for k in range(0, len(idx_bounds), batch_size):
            batch = idx_bounds[k:k+batch_size]
            # Context BEFORE the batch
            confronted = confront_facts(context_state, global_text)
            context_before = confronted["context_text"]
            batch_range = f"{batch[0][0]}–{batch[-1][0]}"
            LOGGER.info("\n" + _box(f"BATCH {batch_range} CONTEXT (BEFORE)",
                                         context_before, max_chars=LOG_CONTEXT_CHARS))
            batch_rows = await run_wave(batch, confronted["context_text"])
            rows.extend(batch_rows)
            # evolve context with the just-finished batch
            for r in batch_rows:
                update_context_from_scene(context_state, r["generated_text"], global_text_hint=global_text)
            # Context AFTER the batch (so you can see what changed)
            confronted_after = confront_facts(context_state, global_text)
            LOGGER.info("\n" + _box(f"BATCH {batch_range} CONTEXT (AFTER)",
                                         confronted_after["context_text"], max_chars=LOG_CONTEXT_CHARS))
    else:
        raise ValueError("EVOLVE_CONTEXT_MODE must be 'none' or 'batched'")

    # -------------- Save results --------------
    # CSV
    LOGGER.info("Writing CSV...")
    rows.sort(key=lambda x: x["scene_id"])  # keep order stable
    fieldnames = list(rows[0].keys()) if rows else [
        "video_name","video_path","scene_id","start_sec","end_sec",
        "start_timecode","end_timecode","scene_duration_sec","fps","width","height",
        "sampled_frames_for_scene","generated_text",
        "brand_hits","brands_logos_raw","products_raw","pii_raw","safety_raw","sports_raw","docs_ui_raw"
    ]
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)
    LOGGER.info(f"CSV saved: {csv_name} ({_filesize(csv_name)})")

    # Build entity -> scenes index (scene IDs where each entity appears)
    entities_scenes: Dict[str, List[int]] = {}
    for r in rows:
        scene_id = r["scene_id"]
        for name in extract_entity_names(r["generated_text"]):
            lst = entities_scenes.setdefault(name, [])
            if scene_id not in lst:
                lst.append(scene_id)
    # keep scene lists sorted for readability
    for k in entities_scenes:
        entities_scenes[k].sort()

    # Context JSON (final confronted facts)
    confronted_final = confront_facts(context_state, global_text)
    ctx_payload = {
        "video_name": base,
        "created_at": stamp,
        "scenes_seen": context_state["scenes_seen"],
        "video_type": context_state["video_type"],
        "summary": context_state["summary"],
        "confronted": confronted_final,
        "entities_freq": context_state["entities_freq"],
        "entity_types": context_state["entity_types"],
        "actions_freq": context_state["actions_freq"],
        "tags_freq": context_state["tags_freq"],
        "ocr_vocab_top": sorted(context_state["ocr_vocab"].items(), key=lambda x: (-x[1], x[0]))[:100],
        "entities_scenes": entities_scenes,
    }
    with open(ctx_name, "w", encoding="utf-8") as f:
        json.dump(ctx_payload, f, ensure_ascii=False, indent=2)
    LOGGER.info(f"Context JSON saved: {ctx_name} ({_filesize(ctx_name)})")

    # Metadata TXT
    technical_meta = {
        "file_name": os.path.basename(video_path),
        "abs_path": os.path.abspath(video_path),
        "created_at": stamp,
        "video_length_sec": round(duration, 3),
        "fps": fps,
        "width": width,
        "height": height,
        "num_scenes_10s": len(bounds),
        "scene_seconds": SCENE_SECONDS,
        "global_sample_frames": global_n
    }
    top_ents_str = ", ".join([f"{k} (x{v})" for k, v in sorted(context_state["entities_freq"].items(), key=lambda x: (-x[1], x[0]))[:20]])
    top_acts_str = ", ".join([f"{k} (x{v})" for k, v in sorted(context_state["actions_freq"].items(), key=lambda x: (-x[1], x[0]))[:20]])
    top_tags_str = ", ".join([f"{k} (x{v})" for k, v in sorted(context_state["tags_freq"].items(), key=lambda x: (-x[1], x[0]))[:20]])

    meta_text = [
        "# GENERAL VIDEO METADATA",
        "",
        "## Technical",
        json.dumps(technical_meta, indent=2),
        "",
        "## Model-Inferred (Global Sampling)",
        global_text.strip(),
        "",
        "## Aggregated Context (Confronted Facts)",
        f"Likely type: {confronted_final.get('likely_type') or context_state['video_type']}",
        confronted_final.get("context_text", ""),
        "",
        "## Top Entities",
        top_ents_str or "(none)",
        "",
        "## Top Actions",
        top_acts_str or "(none)",
        "",
        "## Top Tags",
        top_tags_str or "(none)",
        ""
    ]
    with open(meta_name, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_text))
    LOGGER.info(f"Metadata TXT saved: {meta_name} ({_filesize(meta_name)})")

    await reporter.stop()

    LOGGER.info(f"Done in {time.perf_counter() - t0:.2f}s | CSV: {csv_name} | TXT: {meta_name} | JSON: {ctx_name}")
    return csv_name, meta_name, ctx_name

# ----------------------------
# CLI and entry point
# ----------------------------

def _box(title: str, body: str, max_chars: int = None) -> str:
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars].rstrip() + " …"
    lines = (body or "").splitlines() or [""]
    width = max(len(title), *(len(l) for l in lines))
    bar = "═" * (width + 2)
    out = [f"╔{bar}╗", f"║ {title.ljust(width)} ║", f"╠{bar}╣"]
    for l in lines:
        out.append(f"║ {l.ljust(width)} ║")
    out.append(f"╚{bar}╝")
    return "\n".join(out)

async def main_async(args):
    global aclient, MODEL
    # Build client from CLI/env
    base_url = args.base_url or DEFAULT_BASE_URL
    api_key = args.api_key or DEFAULT_API_KEY
    MODEL = args.model or DEFAULT_MODEL
    LOGGER.info(f"Endpoint: {base_url} | Model: {MODEL}")
    # Instantiate client
    globals()['aclient'] = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Profiles
    custom_profiles = load_custom_profiles(args.custom_profiles)
    # Also allow env overrides for packs
    env_extra = os.getenv("VIDEO_RAG_EXTRA_PACKS", "").strip()
    env_exclude = os.getenv("VIDEO_RAG_EXCLUDE_PACKS", "").strip()
    extra_packs = (args.packs or []) + ([p.strip() for p in env_extra.split(",") if p.strip()] if env_extra else [])
    exclude_packs = (args.exclude_packs or []) + ([p.strip() for p in env_exclude.split(",") if p.strip()] if env_exclude else [])

    prof = resolve_profile(args.profile, extra_packs, exclude_packs, custom_profiles)
    LOGGER.info(f"Active profile: {prof.name} | packs={prof.packs}")

    # Global tunables (override module-level for convenience)
    global SCENE_SECONDS, GLOBAL_SAMPLE_SEGMENTS, GLOBAL_FRAMES_PER_SEGMENT
    global SCENE_SEGMENTS, SCENE_FRAMES_PER_SEGMENT, INPUT_SIZE, GRID_COLS, JPEG_QUALITY
    global GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P
    global MAX_CONCURRENCY, EVOLVE_CONTEXT_MODE, BATCH_SIZE

    SCENE_SECONDS = args.scene_seconds
    GLOBAL_SAMPLE_SEGMENTS = args.global_segments
    GLOBAL_FRAMES_PER_SEGMENT = args.global_fps
    SCENE_SEGMENTS = args.scene_segments
    SCENE_FRAMES_PER_SEGMENT = args.scene_fps
    INPUT_SIZE = args.input_size
    GRID_COLS = args.grid_cols
    JPEG_QUALITY = args.jpeg_quality

    # Generation defaults (profile may override per-call)
    GEN_MAX_NEW_TOKENS = args.max_new_tokens
    GEN_TEMPERATURE = args.temperature
    GEN_TOP_P = args.top_p

    MAX_CONCURRENCY = args.max_concurrency
    EVOLVE_CONTEXT_MODE = args.evolve_context
    BATCH_SIZE = args.batch_size

    stream_mode = args.stream_mode

    await process_strategy2_async(
        args.video,
        profile=prof,
        stream=args.stream,
        stream_mode=stream_mode,
        max_concurrency=args.max_concurrency,
        evolve_mode=args.evolve_context,
        batch_size=args.batch_size,
        stream_throttle_ms=args.stream_throttle_ms,
        stream_min_chars=args.stream_min_chars,
        stream_dump_dir=args.stream_dump_dir,
    )


def build_arg_parser():
    p = argparse.ArgumentParser(description="Video RAG Strategy 2 (clean live + profiles)")
    p.add_argument("--video", default=VIDEO_PATH, help="Path to input video")

    # Endpoint/model
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default from env or localhost)")
    p.add_argument("--api-key", default=None, help="API key (default from env or dummy)")
    p.add_argument("--model", default=None, help="Model name (default from env)")

    # Profiles
    p.add_argument("--profile", default=DEFAULT_PROFILE_NAME, help=f"Profile name (default: {DEFAULT_PROFILE_NAME})")
    p.add_argument("--packs", nargs="*", default=None, help="Additional packs to include (space-separated)")
    p.add_argument("--exclude-packs", nargs="*", default=None, help="Packs to exclude (space-separated)")
    p.add_argument("--custom-profiles", default=None, help="Path to JSON file defining custom profiles")

    # Streaming config
    p.add_argument("--stream", action="store_true", help="Enable live streaming to console")
    p.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming (default)")
    p.set_defaults(stream=False)

    p.add_argument("--stream-mode", choices=["none", "aggregate", "sequential"],
                   default="none", help="'aggregate' streams all scenes with clean sentence chunks; 'sequential' streams only the first scene per wave (others non-stream)")
    p.add_argument("--stream-throttle-ms", type=int, default=200, help="Min milliseconds between console prints per stream")
    p.add_argument("--stream-min-chars", type=int, default=160, help="Flush at least this many chars per print if no sentence end yet")
    p.add_argument("--stream-dump-dir", default=None, help="Optional dir to write per-stream live text (global.txt, scene-#.txt)")

    # Scene/global sampling
    p.add_argument("--scene-seconds", type=int, default=SCENE_SECONDS, help="Seconds per scene chunk")
    p.add_argument("--global-segments", type=int, default=GLOBAL_SAMPLE_SEGMENTS, help="Global pass segments")
    p.add_argument("--global-fps", type=int, default=GLOBAL_FRAMES_PER_SEGMENT, help="Frames per global segment")
    p.add_argument("--scene-segments", type=int, default=SCENE_SEGMENTS, help="Segments per scene")
    p.add_argument("--scene-fps", type=int, default=SCENE_FRAMES_PER_SEGMENT, help="Frames per scene segment")

    # Image tiling
    p.add_argument("--input-size", type=int, default=INPUT_SIZE, help="Square tile size (px)")
    p.add_argument("--grid-cols", type=int, default=GRID_COLS, help="Columns in grid tile")
    p.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY, help="JPEG quality for data URLs")

    # Generation params (defaults; profiles may override)
    p.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=GEN_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=GEN_TOP_P)

    # Parallelism & context evolution
    p.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    p.add_argument("--evolve-context", choices=["none", "batched"], default=EVOLVE_CONTEXT_MODE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    return p

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    # If user enabled --stream but forgot a mode, default to aggregate for nice UX
    if args.stream and args.stream_mode == "none":
        args.stream_mode = "aggregate"
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
