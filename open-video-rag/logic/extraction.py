from typing import Optional, Dict
from pathlib import Path
import sys

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

