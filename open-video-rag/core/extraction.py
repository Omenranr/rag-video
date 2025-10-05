from typing import Optional, Dict
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_parallel_pipelines(
    *,
    want_transcript: bool,
    want_detection: bool,
    want_vlm: bool,
    video_path: str,
    out_dir: str,
    # extractor args
    lang, vclass, fps, device, batch, conf, iou, max_det,
    # vlm args
    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
):
    """
    Run audio transcription/detections and VLM extraction in parallel (if requested).
    Returns a dict:
      {
        "extractor": {"ok": bool, "error": str|None, "out_dir": str|None, "srt_path": str|None},
        "vlm":       {"ok": bool, "error": str|None}
      }
    """

    results = {
        "extractor": {"ok": False, "error": None, "out_dir": None, "srt_path": None},
        "vlm": {"ok": False, "error": None},
    }

    def _run_extractor_job():
        # Only run if we need an SRT (transcript) or detections
        if not (want_transcript or want_detection):
            return None
        out_dir_final = run_extractor(
            video_path=video_path,
            lang=lang, vclass=vclass,
            fps=(None if not fps else int(fps)),
            device=device, batch_size=int(batch),
            conf=float(conf), iou=float(iou), max_det=int(max_det),
            progress=None,
            enable_detection=bool(want_detection),
        )
        srt = find_any_srt(str(out_dir_final))
        return {"out_dir": out_dir_final, "srt_path": (str(srt) if srt else None)}

    def _run_vlm_job():
        if not want_vlm:
            return None
        run_visual_strategy(
            video_path=video_path, outputs_dir=out_dir,
            base_url=vlm_base_url.strip(), api_key=vlm_api_key.strip(),
            model=vlm_model.strip(), profile=vlm_profile,
            max_concurrency=int(vlm_maxconc),
            stream=bool(vlm_stream), stream_mode=vlm_stream_mode,
            force=False,
        )
        return {"ok": True}

    # Launch in parallel when applicable
    futures = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        if want_transcript or want_detection:
            futures.append(("extractor", ex.submit(_run_extractor_job)))
        if want_vlm:
            futures.append(("vlm", ex.submit(_run_vlm_job)))

        for label, fut in futures:
            try:
                val = fut.result()
                if label == "extractor":
                    if val is None:
                        results["extractor"]["ok"] = True
                    else:
                        results["extractor"]["ok"] = True
                        results["extractor"]["out_dir"] = val.get("out_dir")
                        results["extractor"]["srt_path"] = val.get("srt_path")
                else:
                    results["vlm"]["ok"] = True
            except Exception as e:
                if label == "extractor":
                    results["extractor"]["error"] = str(e)
                else:
                    results["vlm"]["error"] = str(e)

    return results
