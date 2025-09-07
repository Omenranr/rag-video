#!/usr/bin/env python3
"""
video_audio_multitool.py  (improved detections & outputs)
========================================================

- Captures ALL detections per frame (fixed relative time bug).
- Adds per-frame aggregated CSV in addition to per-object CSV.
- Configurable detection params: conf / iou / max_det.
- Consistent SRT-style timestamps + frame_index in outputs.

Usage:
------
pip install av==10.* torch torchvision torchaudio paddleocr==2.7 \
            ultralytics>=8.2.0 pandas tqdm soundfile silero-vad numpy \
            python-dotenv tafrigh

# Ensure ffmpeg and yt-dlp are installed on your PATH.

# .env next to this script:
#   WIT_API_KEY_ENGLISH=...
#   WIT_API_KEY_FRENCH=...

python video_audio_multitool.py "<video_or_audio_path_or_url>" \
    --lang EN \
    --vclass sport \
    --fps 15 \
    --device auto \
    --batch-size 64 \
    --det-conf 0.25 \
    --det-iou 0.7 \
    --max-det 300
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Tuple, Optional, Dict

from dotenv import load_dotenv

import av
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# --- Optional heavy deps ---
try:
    from paddleocr import PaddleOCR
    import paddle
except Exception:
    PaddleOCR = None  # type: ignore
    paddle = None     # type: ignore

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore

try:
    from tafrigh import Config, TranscriptType, farrigh
except Exception:
    Config = None           # type: ignore
    TranscriptType = None   # type: ignore
    farrigh = None          # type: ignore


# -----------------------------------------------------------------------------
# Environment & logging
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video_audio_multitool")


# -----------------------------------------------------------------------------
# Wit.ai API keys per language
# -----------------------------------------------------------------------------
LANGUAGE_API_KEYS: Dict[str, Optional[str]] = {
    'EN': os.getenv('WIT_API_KEY_ENGLISH'),
    'FR': os.getenv('WIT_API_KEY_FRENCH'),
    # 'AR': os.getenv('WIT_API_KEY_ARABIC'),
    # 'JA': os.getenv('WIT_API_KEY_JAPANESE'),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_cmd_exists(cmd: str, friendly_name: Optional[str] = None):
    name = friendly_name or cmd
    if shutil.which(cmd) is None:
        raise RuntimeError(
            f"Required tool '{name}' is not installed or not on PATH."
        )

def is_url(s: str) -> bool:
    return bool(re.match(r'^https?://', s.strip(), re.IGNORECASE))

def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def seconds_to_srt_time(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    hours, rem_ms = divmod(total_ms, 3600000)
    minutes, rem_ms = divmod(rem_ms, 60000)
    seconds, millis = divmod(rem_ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def safe_stem(path: Path) -> str:
    s = path.stem.strip()
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s)
    return s or "video"

def make_output_dir(input_stem: str) -> Path:
    out = Path("outputs") / f"{input_stem}_{now_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------------------------------------------------------
# Download / Convert
# -----------------------------------------------------------------------------
def download_video(url: str, out_dir: Path) -> Path:
    ensure_cmd_exists("yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(title).80s-%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-o", template,
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        url
    ]
    logger.info("Downloading video…")
    subprocess.run(cmd, check=True)
    vids = list(out_dir.glob("*.mp4"))
    if not vids:
        raise RuntimeError("Download succeeded but no MP4 found.")
    vid = max(vids, key=lambda p: p.stat().st_mtime)
    logger.info("Downloaded → %s", vid)
    return vid

def convert_video_to_wav(video_path: Path, wav_out: Path) -> Path:
    ensure_cmd_exists("ffmpeg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(wav_out)
    ]
    logger.info("Extracting audio → %s", wav_out.name)
    subprocess.run(cmd, check=True)
    return wav_out

def convert_mp3_to_wav(mp3_path: Path, wav_out: Path) -> Path:
    ensure_cmd_exists("ffmpeg")
    cmd = ["ffmpeg", "-y", "-i", str(mp3_path), str(wav_out)]
    logger.info("Converting MP3 → WAV: %s → %s", mp3_path.name, wav_out.name)
    subprocess.run(cmd, check=True)
    return wav_out


# -----------------------------------------------------------------------------
# Transcription (tafrigh + Wit.ai)
# -----------------------------------------------------------------------------
def transcribe_wit(wav_path: Path, lang_sign: str, out_dir: Path):
    if farrigh is None or Config is None or TranscriptType is None:
        raise RuntimeError("`tafrigh` is not installed. `pip install tafrigh` to enable transcription.")

    wit_api_key = LANGUAGE_API_KEYS.get(lang_sign.upper())
    if not wit_api_key:
        raise RuntimeError(
            f"No Wit.ai API key for language '{lang_sign}'. "
            f"Set it in your .env (e.g., WIT_API_KEY_ENGLISH)."
        )

    logger.info("Transcribing (Wit.ai via tafrigh)…")
    config = Config(
        urls_or_paths=[str(wav_path)],
        skip_if_output_exist=False,
        playlist_items="",
        verbose=False,
        model_name_or_path="",
        task="",
        language="",
        use_faster_whisper=False,
        beam_size=0,
        ct2_compute_type="",
        wit_client_access_tokens=[wit_api_key],
        max_cutting_duration=5,
        min_words_per_segment=1,
        save_files_before_compact=False,
        save_yt_dlp_responses=False,
        output_sample=0,
        output_formats=[TranscriptType.SRT],
        output_dir=str(out_dir),
    )
    _ = list(farrigh(config))
    srts = sorted(out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not srts:
        raise RuntimeError("Transcription finished but no SRT produced.")
    srt_path = srts[0]
    logger.info("Transcription saved → %s", srt_path.name)
    return srt_path


# -----------------------------------------------------------------------------
# Detection pipeline
# -----------------------------------------------------------------------------
@dataclass
class Detection:
    frame_idx: int        # sampled frame index (integer)
    frame_ts: float       # seconds from video start
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    extra: str            # json string (e.g., OCR text)

@dataclass
class ClipSpec:
    fps: int
    detector_fn: Callable[[List[np.ndarray], float, str], List[Detection]]

_det_cache: Dict[str, object] = {}

# Global detection options (set from CLI)
DETECT_OPTS = {
    "conf": 0.25,
    "iou": 0.7,
    "max_det": 300,
}

def _load_ultra(model_id: str, device: str = "cpu"):
    key = f"ultra:{model_id}:{device}"
    if key not in _det_cache:
        if YOLO is None:
            raise RuntimeError("ultralytics not installed; `pip install ultralytics`")
        logger.info("Loading Ultralytics model '%s' on '%s'…", model_id, device)
        _det_cache[key] = YOLO(model_id, task="detect").to(device).eval()
    return _det_cache[key]

def _load_paddleocr(device: str):
    if "ocr" in _det_cache:
        return _det_cache["ocr"]
    if PaddleOCR is None or paddle is None:
        raise RuntimeError("PaddleOCR not available. Install `paddlepaddle(-gpu)` and `paddleocr`.")
    dev_str = "gpu:0" if device == "cuda" else "cpu"
    logger.info("Setting Paddle device: %s", dev_str)
    paddle.device.set_device(dev_str)
    try:
        _det_cache["ocr"] = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            rec_batch_num=32,
        )
    except OSError as e:
        logger.warning("PaddleOCR init failed on '%s': %s. Falling back to CPU.", dev_str, e)
        paddle.device.set_device("cpu")
        _det_cache["ocr"] = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            rec_batch_num=16,
        )
    return _det_cache["ocr"]

def _iter_paddle_lines(res):
    if isinstance(res, dict):
        data = res.get("res", res)
        polys = data.get("rec_polys") or data.get("dt_polys")
        texts = data.get("rec_texts")
        scores = data.get("rec_scores")
        if polys is not None and texts is not None and scores is not None:
            for poly, text, score in zip(polys, texts, scores):
                yield poly, text, float(score)
        return
    if isinstance(res, list) and res and isinstance(res[0], list):
        for box, (text, conf) in res[0]:
            yield box, text, float(conf)
        return
    return

def detect_screen(frames: List[np.ndarray], fps: float, device: str = "cpu") -> List[Detection]:
    if not frames:
        return []
    ocr = _load_paddleocr(device)
    batch_res = ocr.predict(frames)
    outs: List[Detection] = []
    for f_idx, res in enumerate(batch_res):
        ts_rel = f_idx / float(fps)
        for poly, text, conf in _iter_paddle_lines(res):
            x1, y1 = map(float, poly[0])
            x2, y2 = map(float, poly[2])
            outs.append(Detection(f_idx, ts_rel, "text", conf, x1, y1, x2, y2, json.dumps({"text": text})))
    return outs

ULTRA_ID_SPORT = "yolov10l.pt"   # swap to a permissive HF ckpt if licensing matters
ULTRA_ID_NATURE = "yolov10l.pt"

def _ultra_predict(frames: List[np.ndarray], fps: float, device: str, model_id: str) -> List[Detection]:
    model = _load_ultra(model_id, device)
    if not frames:
        return []
    results = model.predict(
        frames,
        device=device,
        verbose=False,
        conf=DETECT_OPTS["conf"],
        iou=DETECT_OPTS["iou"],
        max_det=DETECT_OPTS["max_det"],
        agnostic_nms=False,  # keep per-class NMS
    )
    outs: List[Detection] = []
    for f_idx, res in enumerate(results):
        # precise per-frame relative seconds (so multiple detections share the same ts)
        ts_rel = f_idx / float(fps)
        boxes = res.boxes
        # iterate explicitly to avoid any zip-mismatch
        n = 0 if boxes is None else boxes.shape[0]
        for i in range(n):
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
            label = model.names[cls_id]
            outs.append(Detection(f_idx, ts_rel, label, conf, x1, y1, x2, y2, ""))
    return outs

def detect_sport(frames: List[np.ndarray], fps: float, device: str) -> List[Detection]:
    return _ultra_predict(frames, fps, device, ULTRA_ID_SPORT)

def detect_nature(frames: List[np.ndarray], fps: float, device: str) -> List[Detection]:
    return _ultra_predict(frames, fps, device, ULTRA_ID_NATURE)

def detect_cctv(frames: List[np.ndarray], fps: float, device: str) -> List[Detection]:
    return detect_nature(frames, fps, device)

PIPELINES = {
    "screen": ClipSpec(fps=3,  detector_fn=detect_screen),
    "sport":  ClipSpec(fps=15, detector_fn=detect_sport),
    "nature": ClipSpec(fps=7,  detector_fn=detect_nature),
    "cctv":   ClipSpec(fps=7,  detector_fn=detect_cctv),
}

def frame_sampler(path: str, target_fps: int) -> Iterator[Tuple[float, np.ndarray]]:
    container = av.open(path)
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate) if stream.average_rate else 30.0
    step = max(1, round(native_fps / target_fps))
    frame_idx = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame_idx % step == 0:
                ts = float(frame.pts * frame.time_base) if frame.pts is not None else frame.time
                if ts is None:
                    ts = frame_idx / native_fps
                yield ts, frame.to_ndarray(format="rgb24")
            frame_idx += 1

def _process_batch(
    frames: List[np.ndarray],
    ts_list: List[float],
    cfg: ClipSpec,
    device: str,
    out_rows: List[Detection],
    base_idx: int,
):
    dets = cfg.detector_fn(frames, cfg.fps, device)
    ts0 = ts_list[0] if ts_list else 0.0
    for d in dets:
        d.frame_ts = ts0 + d.frame_ts   # align to absolute video time
        d.frame_idx = base_idx + d.frame_idx
    out_rows.extend(dets)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run_pipeline(
    input_arg: str,
    lang: str,
    vclass: str,
    fps: Optional[int],
    device_choice: str,
    batch_size: int,
    det_conf: float,
    det_iou: float,
    det_max: int,
) -> Path:
    downloads_dir = Path("downloads"); downloads_dir.mkdir(exist_ok=True)
    local_video_path: Optional[Path] = None
    local_audio_wav: Optional[Path] = None

    # 1) Input resolve
    if is_url(input_arg):
        local_video_path = download_video(input_arg, downloads_dir)
        input_stem = safe_stem(local_video_path)
    else:
        p = Path(input_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {p}")
        if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"}:
            local_video_path = p
        elif p.suffix.lower() == ".wav":
            local_audio_wav = p
        elif p.suffix.lower() == ".mp3":
            out_wav = downloads_dir / (safe_stem(p) + ".wav")
            local_audio_wav = convert_mp3_to_wav(p, out_wav)
        else:
            raise ValueError(f"Unsupported input format: {p.suffix.lower()}")
        input_stem = safe_stem(p)

    # 2) Output dir
    out_dir = make_output_dir(input_stem)
    logger.info("Output folder: %s", out_dir)

    # 3) Transcription (need WAV; extract from video if needed)
    if local_audio_wav is None:
        if local_video_path is None:
            raise RuntimeError("No media found for transcription.")
        local_audio_wav = convert_video_to_wav(local_video_path, out_dir / "audio.wav")
    srt_path = transcribe_wit(local_audio_wav, lang, out_dir)

    # 4) Detections (only if we have a video)
    if local_video_path is None:
        logger.warning("No video available — skipping visual detections.")
        return out_dir

    device = "cuda" if device_choice == "auto" and torch.cuda.is_available() else device_choice if device_choice != "auto" else "cpu"
    cfg = PIPELINES[vclass]
    if fps:
        cfg.fps = int(fps)

    # Set global detection options
    DETECT_OPTS["conf"] = float(det_conf)
    DETECT_OPTS["iou"] = float(det_iou)
    DETECT_OPTS["max_det"] = int(det_max)

    # Warmup (except OCR)
    if vclass != "screen":
        try:
            cfg.detector_fn([], cfg.fps, device)
        except Exception as e:
            logger.warning("Model warmup issue: %s", e)

    rows: List[Detection] = []
    batch_frames: List[np.ndarray] = []
    batch_ts: List[float] = []
    BATCH_SZ = int(batch_size)

    logger.info("Running '%s' detector | fps=%d | device=%s | conf=%.3f | iou=%.2f | max_det=%d",
                vclass, cfg.fps, device, DETECT_OPTS["conf"], DETECT_OPTS["iou"], DETECT_OPTS["max_det"])

    frame_counter = 0
    batch_start_idx = 0  # global index of first frame in current batch

    with logging_redirect_tqdm():
        for ts, frame in tqdm(frame_sampler(str(local_video_path), cfg.fps), desc="Frames"):
            if not batch_frames:
                batch_start_idx = frame_counter
            batch_frames.append(frame)
            batch_ts.append(ts)
            frame_counter += 1

            if len(batch_frames) == BATCH_SZ:
                _process_batch(batch_frames, batch_ts, cfg, device, rows, base_idx=batch_start_idx)
                batch_frames, batch_ts = [], []

        if batch_frames:
            _process_batch(batch_frames, batch_ts, cfg, device, rows, base_idx=batch_start_idx)

    # 5) Save per-object CSV (tall)
    det_csv = out_dir / "detections.csv"
    df_rows = []
    for r in rows:
        df_rows.append({
            "frame_index": int(r.frame_idx),
            "time": seconds_to_srt_time(r.frame_ts),     # SRT timestamp
            "time_seconds": round(float(r.frame_ts), 3),
            "label": r.label,
            "conf": round(float(r.conf), 6),
            "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
            "extra": r.extra
        })
    df_tall = pd.DataFrame(df_rows)
    if not df_tall.empty:
        df_tall.sort_values(["frame_index", "label", "conf"], ascending=[True, True, False], inplace=True)
    df_tall.to_csv(det_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info("Detections (per-object) → %s (%d rows)", det_csv.name, len(df_tall))

    # 6) Save per-frame aggregated CSV (wide-ish)
    by_frame_csv = out_dir / "detections_by_frame.csv"
    if not df_tall.empty:
        groups = []
        for frame_idx, g in df_tall.groupby("frame_index"):
            time_srt = g["time"].iloc[0]
            time_sec = float(g["time_seconds"].iloc[0])
            labels = g["label"].tolist()
            class_counts: Dict[str, int] = {}
            for lb in labels:
                class_counts[lb] = class_counts.get(lb, 0) + 1
            objects = []
            for _, row in g.iterrows():
                objects.append({
                    "label": row["label"],
                    "conf": float(row["conf"]),
                    "bbox_xyxy": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                    "extra": row["extra"],
                })
            groups.append({
                "frame_index": int(frame_idx),
                "time": time_srt,
                "time_seconds": round(time_sec, 3),
                "n_detections": int(len(g)),
                "labels": ";".join(labels),
                "class_counts_json": json.dumps(class_counts, ensure_ascii=False),
                "objects_json": json.dumps(objects, ensure_ascii=False),
            })
        df_by = pd.DataFrame(groups).sort_values("frame_index")
        df_by.to_csv(by_frame_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info("Detections (per-frame aggregated) → %s (%d frames)", by_frame_csv.name, len(df_by))
    else:
        # create empty file with headers
        pd.DataFrame(columns=[
            "frame_index","time","time_seconds","n_detections","labels","class_counts_json","objects_json"
        ]).to_csv(by_frame_csv, index=False)

    # 7) Make transcript alias predictable
    transcript_alias = out_dir / "transcript.srt"
    try:
        if Path(srt_path) != transcript_alias:
            shutil.copy2(srt_path, transcript_alias)
            logger.info("Transcript copied → %s", transcript_alias.name)
    except Exception as e:
        logger.warning("Could not copy transcript: %s", e)

    return out_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Transcribe and detect content from a video URL or local file.")
    parser.add_argument("input", help="Video URL or local media path (.mp4/.mkv/.avi/.mov/.wav/.mp3)")
    parser.add_argument("--lang", default="EN", help="Language for Wit.ai (e.g., EN, FR)")
    parser.add_argument("--vclass", default="nature", choices=list(PIPELINES.keys()), help="Detection pipeline")
    parser.add_argument("--fps", type=int, help="Override sampling FPS")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Detection device")
    parser.add_argument("--batch-size", dest="batch_size", default=64, type=int, choices=[8,16,32,64,128], help="Frame batch size")

    # NEW: detection knobs
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--det-iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per frame")

    args = parser.parse_args()

    if not any(LANGUAGE_API_KEYS.values()):
        logger.error("At least one Wit.ai API key must be provided in the .env file.")
        sys.exit(1)

    try:
        out_dir = run_pipeline(
            input_arg=args.input,
            lang=args.lang,
            vclass=args.vclass,
            fps=args.fps,
            device_choice=args.device,
            batch_size=args.batch_size,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            det_max=args.max_det,
        )
        print("\n✅ Done.")
        print(f"   Results: {out_dir}")
        print(f"   - transcript.srt")
        print(f"   - detections.csv             (one row per object)")
        print(f"   - detections_by_frame.csv    (one row per frame, aggregated)")
    except Exception as e:
        logger.exception("Failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
