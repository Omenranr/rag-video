from pathlib import Path
from typing import Optional
import re
import sys
from resources.consts import *

def srt_to_seconds(s: str) -> int:
    # "HH:MM:SS,mmm" → seconds (floor)
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", s.strip())
    if not m:
        return 0
    hh, mm, ss, _ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss


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


def has_transcript(outputs_dir: str) -> bool:
    return find_any_srt(outputs_dir) is not None