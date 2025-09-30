import unicodedata
import re
import json
from pathlib import Path
from typing import Optional, List
from resources.consts import *

# Cache of loaded indexes: {index_dir: TranscriptRAGIndex}
_INDEX_CACHE: Dict[str, object] = {}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize_dash(s: str) -> str:
    return s.replace("–", "-").replace("—", "-")


def normalize_entity_key(s: str) -> str:
    s = (s or "").strip()
    s = strip_accents(s).lower()
    # drop punctuation → keep letters/digits/space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # optional: remove leading articles
    s = re.sub(r"^(the|la|le|les|l|une|un)\s+", "", s)
    return s


def safe_stem(path: Path) -> str:
    s = path.stem.strip()
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s)
    return s or "video"


def format_source_label(src: str | None) -> str:
    s = (src or "transcript").lower()
    if s.startswith("vlm"): return "VLM"
    if s == "transcript":  return "SRT"
    return s.upper()


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


def latest_rag_index_dir(outputs_dir: str) -> Optional[str]:
    base = Path(outputs_dir)
    idx_dirs = [p for p in base.glob("rag_index_*") if p.is_dir()]
    if not idx_dirs:
        return None
    idx_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(idx_dirs[0])