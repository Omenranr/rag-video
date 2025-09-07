"""
multiclass_video_pipeline.py  (rev‑2)
====================================
**What changed vs rev‑1?**

* **RT‑DETR is now loaded directly from `ultralytics`** – the same library that
  ships YOLO‑v10; so the extra `rtdetr` pip package is no longer required.
* Added a generic `_load_ultra(model_id)` helper that caches *any* Ultralytics
  checkpoint (`yolov10m`, `rtdetr‑l`, etc.).
* Updated pip one‑liner accordingly.

```bash
pip install av==10.* torch torchvision torchaudio paddleocr==2.7 \
            ultralytics>=8.2.0 pandas tqdm soundfile silero-vad numpy

python multiclass_video_pipeline.py video.mp4 --class nature --csv nature.csv
```

Licence status stays the same: Ultralytics weights default to AGPL‑3; pick
Apache/CC checkpoints on Hugging Face if you need permissive terms.

---  code begins  ---
"""
from __future__ import annotations
import time
import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Tuple
import logging
from logging import getLogger
import av
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


# ---------------------------------------------------------------------------
#  OPTIONAL DEPENDENCIES
# ---------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR
    import paddle
except ImportError:
    PaddleOCR = None  # type: ignore

try:
    from ultralytics import YOLO  # same class loads YOLOv10 and RT‑DETR
except ImportError:
    YOLO = None  # type: ignore



# ---------------------------------------------------------------------------
@dataclass
class Detection:
    frame_ts: float
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    extra: str  # json (OCR text, pose, etc.)


@dataclass
class ClipSpec:
    fps: int
    detector_fn: Callable[[List[np.ndarray], float, str], List[Detection]]


_det_cache = {}
logger = getLogger(__name__)

# ---------------------------------------------------------------------------
#  ULTRALYTICS MODEL LOADER  -----------------------------------------------

def _load_ultra(model_id: str, device: str = "cpu"):
    """Lazy‑load and cache any Ultralytics checkpoint."""
    key = f"ultra:{model_id}:{device}"
    if key not in _det_cache:
        if YOLO is None:
            raise RuntimeError("ultralytics not installed; pip install ultralytics")
        logger.info("Loading Ultralytics model '%s' on device '%s'...", model_id, device)
        _det_cache[key] = YOLO(model_id, task="detect").to(device).eval()
        logger.debug("Model '%s' loaded and cached.", key)
    else:
        logger.debug("Using cached Ultralytics model '%s'.", key)
    return _det_cache[key]

# ---------------------------------------------------------------------------
#  PER‑CLASS DETECTORS  -----------------------------------------------------

def detect_screen(frames: List[np.ndarray], fps: float, device: str = "cpu") -> List[Detection]:
    """
    Run PaddleOCR in batch on a list of frames to speed up inference.

    Parameters
    ----------
    frames : List[np.ndarray]
        RGB frames (H, W, 3) as numpy arrays.
    fps : float
        Sampling FPS used to compute timestamps.
    device : str
        'cpu' or 'cuda' (mapped internally by _load_paddleocr).

    Returns
    -------
    List[Detection]
        One Detection per text line found.
    """
    logger = logging.getLogger(__name__)
    if not frames:
        logger.debug("detect_screen called with 0 frames.")
        return []

    start = time.time()
    ocr = _load_paddleocr(device)

    batch_res = ocr.predict(frames)

    outs: List[Detection] = []
    for f_idx, res in enumerate(batch_res):
        ts = f_idx / fps
        for poly, text, conf in _iter_paddle_lines(res):
            # poly is 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x1, y1 = map(float, poly[0])
            x2, y2 = map(float, poly[2])
            outs.append(
                Detection(ts, "text", conf, x1, y1, x2, y2, json.dumps({"text": text}))
            )

    elapsed = time.time() - start
    logger.debug(
        "detect_screen processed %d frames -> %d detections in %.3fs (%.2f fps)",
        len(frames), len(outs), elapsed,
        (len(frames) / elapsed) if elapsed > 0 else 0.0
    )
    return outs




def _load_paddleocr(device: str):
    """
    Initialize PaddleOCR without the deprecated 'use_gpu' flag.
    Device is controlled globally via paddle.set_device().
    """
    if "ocr" in _det_cache:
        logger.debug("Using cached PaddleOCR instance.")
        return _det_cache["ocr"]

    if PaddleOCR is None:
        raise RuntimeError("No OCR backend found. Install paddleocr.")
    if paddle is None:
        raise RuntimeError("PaddlePaddle is required. Install paddlepaddle or paddlepaddle-gpu.")

    # Decide and set Paddle device
    if device == "cuda":
        dev_str = "gpu:0"
    else:
        dev_str = "cpu"
    logger.info("Setting Paddle device to '%s' (replaces deprecated use_gpu).", dev_str)
    paddle.device.set_device(dev_str)

    # Instantiate OCR
    try:
        _det_cache["ocr"] = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            rec_batch_num=32,  # tune as needed
        )
        logger.debug("PaddleOCR initialized with rec_batch_num=%s", 32)
    except OSError as e:
        logger.warning("PaddleOCR init failed on '%s': %s. Falling back to CPU.", dev_str, e)
        paddle.device.set_device("cpu")
        _det_cache["ocr"] = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            rec_batch_num=16,
        )
        logger.debug("PaddleOCR CPU fallback with rec_batch_num=%s", 16)

    return _det_cache["ocr"]



# Ultralytics IDs: you can swap in custom HF checkpoints
ULTRA_ID_SPORT = "yolov10l.pt"   # AGPL‑3 by default
ULTRA_ID_NATURE = "yolov10l.pt"


def detect_sport(frames: List[np.ndarray], fps: float, device: str) -> List[Detection]:
    model = _load_ultra(ULTRA_ID_SPORT, device)
    if not frames:
        return []
    results = model.predict(frames, verbose=False, device=device)
    dets: List[Detection] = []
    for f_idx, res in enumerate(results):
        ts = f_idx / fps
        for cls_id, conf, xyxy in zip(
            res.boxes.cls.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.xyxy.cpu().numpy(),
        ):
            label = model.names[int(cls_id)]
            x1, y1, x2, y2 = map(float, xyxy)
            dets.append(Detection(ts, label, float(conf), x1, y1, x2, y2, ""))
    return dets


def detect_nature(frames: List[np.ndarray], fps: float, device: str) -> List[Detection]:
    # print(f"Starting Nature Detection for frames length: {len(frames)} with fps {fps}")
    model = _load_ultra(ULTRA_ID_NATURE, device)
    if not frames:
        return []
    results = model.predict(frames, verbose=False, device=device)
    dets: List[Detection] = []
    for idx, res in enumerate(results):
        ts = idx / fps
        for cls_id, conf, xyxy in zip(
            res.boxes.cls.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.xyxy.cpu().numpy(),
        ):
            label = model.names[int(cls_id)]
            x1, y1, x2, y2 = map(float, xyxy)
            dets.append(Detection(ts, label, float(conf), x1, y1, x2, y2, ""))
    return dets


def detect_cctv(frames: List[np.ndarray], fps: float, device: str) -> List[Detection]:
    # Reuse nature detector; optionally add tracking later
    return detect_nature(frames, fps, device)

# ---------------------------------------------------------------------------
PIPELINES = {
    "screen": ClipSpec(fps=3, detector_fn=detect_screen),
    "sport": ClipSpec(fps=15, detector_fn=detect_sport),
    "nature": ClipSpec(fps=7, detector_fn=detect_nature),
    "cctv": ClipSpec(fps=7, detector_fn=detect_cctv),
}

# ---------------------------------------------------------------------------
#  VIDEO SAMPLER  -----------------------------------------------------------

def frame_sampler(path: str, target_fps: int) -> Iterator[Tuple[float, np.ndarray]]:
    container = av.open(path)
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate) if stream.average_rate else 30.0
    step = max(1, round(native_fps / target_fps))
    frame_idx = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame_idx % step == 0:
                ts = frame.pts * frame.time_base
                yield ts, frame.to_ndarray(format="rgb24")
            frame_idx += 1


# ---------------------------------------------------------------------------
#  MAIN  --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Multi‑class video metadata extractor")
    p.add_argument("video", help="input video path")
    p.add_argument("--class", dest="vclass", required=True, choices=PIPELINES.keys())
    p.add_argument("--fps", dest="fps")
    p.add_argument("--csv", default="meta.csv", help="output CSV")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch_size", dest="batch_size", default='64', choices=['8', '16', '32', '64', '128'])
    args = p.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    cfg = PIPELINES[args.vclass]

    if args.fps:
        cfg.fps = int(args.fps)

    print(f"▶  Using '{args.vclass}' pipeline | fps={cfg.fps} | device={device}")

    # prime model cache (skip for screen)
    if args.vclass != "screen":
        cfg.detector_fn([], cfg.fps, device)

    rows: List[Detection] = []
    batch_frames: List[np.ndarray] = []
    batch_ts: List[float] = []
    BATCH_SZ = int(args.batch_size)

    for ts, frame in tqdm(frame_sampler(args.video, cfg.fps), desc="Frames"):
        batch_frames.append(frame)
        batch_ts.append(ts)
        if len(batch_frames) == BATCH_SZ:
            print(f"{ts}_{len(batch_frames)}")
            _process_batch(batch_frames, batch_ts, cfg, device, rows)
            batch_frames, batch_ts = [], []

    if batch_frames:
        _process_batch(batch_frames, batch_ts, cfg, device, rows)

    pd.DataFrame([asdict(r) for r in rows]).to_csv(Path(args.csv), index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅  saved {len(rows)} rows → {args.csv}")


def _iter_paddle_lines(res):
    """
    Yields (poly, text, score) for each detected line.
    Works with both PaddleOCR <2.7 and >=2.7 outputs.
    """
    # Newer PaddleOCR (>=2.7): dict with 'res' or directly the arrays
    if isinstance(res, dict):
        data = res.get("res", res)
        polys = data.get("rec_polys") or data.get("dt_polys")
        texts = data.get("rec_texts")
        scores = data.get("rec_scores")
        if polys is not None and texts is not None and scores is not None:
            for poly, text, score in zip(polys, texts, scores):
                yield poly, text, float(score)
        return

    # Older PaddleOCR (<2.7): list format: res[0] -> [ [poly], [text, score] ]
    if isinstance(res, list) and res and isinstance(res[0], list):
        for box, (text, conf) in res[0]:
            yield box, text, float(conf)
        return

    # If nothing matched, silently yield nothing
    return


def _process_batch(frames: List[np.ndarray], ts_list: List[float], cfg: ClipSpec, device: str, out_rows: List[Detection]):
    dets = cfg.detector_fn(frames, cfg.fps, device)
    ts0 = ts_list[0]
    for d in dets:
        d.frame_ts = ts0 + d.frame_ts
    out_rows.extend(dets)


if __name__ == "__main__":
    main()
