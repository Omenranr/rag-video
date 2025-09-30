from __future__ import annotations


import re
from pathlib import Path
from typing import List, Optional


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}

def srt_to_seconds(s: str) -> int:
    """"HH:MM:SS,mmm" → seconds (floor)."""
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$", s.strip())
    if not m:
        return 0
    hh, mm, ss, _ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss


def list_videos(folder: str) -> List[str]:
    p = Path(folder).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    hits: List[str] = []
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