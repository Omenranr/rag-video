# visual_rag_index.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

# Reuse the transcript index builder
from transcript_hybrid_rag import TranscriptRAGIndex

def _seconds_to_srt_time(sec: float) -> str:
    if sec < 0: sec = 0.0
    total_ms = int(round(sec * 1000))
    hh, rem = divmod(total_ms, 3600000)
    mm, rem = divmod(rem, 60000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

def _read_strategy2_visual_files(
    outputs_dir: str
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Returns (csv_path, ctx_json_path, meta_txt_path) if found in outputs_dir; else Nones.
    Picks the newest CSV that matches the base*stamp pattern produced by strategy2.
    """
    base = Path(outputs_dir)
    if not base.exists(): return None, None, None
    csvs = sorted(base.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    ctxs = sorted(base.glob("*_context.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    metas = sorted(base.glob("*_metadata.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (csvs[0] if csvs else None,
            ctxs[0] if ctxs else None,
            metas[0] if metas else None)

def _make_visual_chunks_df(csv_path: Path, ctx_json_path: Optional[Path]) -> pd.DataFrame:
    """
    Build a chunks dataframe compatible with TranscriptRAGIndex.build_from_dataframe.
    One row per scene (10s) from strategy2 CSV. Text = generated_text + small structured append.
    Required columns: chunk_id, start, end, start_srt, end_srt, text, source
    """
    df_raw = pd.read_csv(csv_path, encoding="utf-8")
    # Defensive defaults
    for col in ["scene_id","start_sec","end_sec","generated_text"]:
        if col not in df_raw.columns:
            raise ValueError(f"Visual CSV missing column: {col}")
    # Build text payload (add light lexical juice from structured fields)
    def _augment(row):
        parts = [str(row.get("generated_text","")).strip()]
        # Add brands/tags to help BM25
        brands = row.get("brand_hits")
        if isinstance(brands, str) and brands.strip().startswith("["):
            try:
                arr = json.loads(brands)
            except Exception:
                arr = []
        else:
            arr = brands if isinstance(brands, list) else []
        if arr:
            parts.append("Brands: " + ", ".join(map(str, arr)))
        for raw_key, label in [
            ("products_raw","Products"),
            ("sports_raw","Sports"),
            ("docs_ui_raw","DocsUI"),
            ("pii_raw","PII"),
            ("safety_raw","Safety"),
            ("brands_logos_raw","BrandsLogosRaw"),
        ]:
            val = row.get(raw_key, "")
            if isinstance(val, str) and val.strip():
                parts.append(f"{label}: {val.strip()}")
        return "\n".join(p for p in parts if p)

    start = df_raw["start_sec"].astype(float)
    end   = df_raw["end_sec"].astype(float)
    scene = df_raw["scene_id"].astype(int)

    text = df_raw.apply(_augment, axis=1)

    chunk_id = [
        f"vlm_{int(sid):06d}_{int(round(s*1000)):010d}"
        for sid, s in zip(scene.tolist(), start.tolist())
    ]

    df = pd.DataFrame({
        "chunk_id": chunk_id,
        "start": start,
        "end": end,
        "start_srt": [_seconds_to_srt_time(float(s)) for s in start.tolist()],
        "end_srt":   [_seconds_to_srt_time(float(e)) for e in end.tolist()],
        "text": text,
        "source": "vlm_scene",
        "scene_id": scene,
    })

    # Keep some useful extras (if present) – they’ll be preserved and surfaced
    for extra in ["video_name","video_path","fps","width","height","scene_duration_sec"]:
        if extra in df_raw.columns:
            df[extra] = df_raw[extra]

    # Attach a hint to the global context file so UIs can reference it later
    if ctx_json_path:
        df["vlm_context_json"] = str(ctx_json_path)

    return df

def build_visual_index_from_outputs(
    outputs_dir: str,
    outdir_name: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None
) -> str:
    """
    Locate CSV + *_context.json inside outputs_dir, then build a Visual RAG index:
      outputs_dir / (outdir_name or 'rag_index_vlm_<stamp>')
    Returns the index directory path.
    """
    csv_path, ctx_json, meta_txt = _read_strategy2_visual_files(outputs_dir)
    if not csv_path:
        raise FileNotFoundError(f"No visual CSV (*.csv) found in: {outputs_dir}")
    df = _make_visual_chunks_df(csv_path, ctx_json)

    # Name index folder
    base = Path(outputs_dir)
    outdir = base / (outdir_name or f"rag_index_vlm_{Path(csv_path).stem}")
    # Keep a short name if the CSV has a long timestamped name
    if len(outdir.name) > 48:
        outdir = base / "rag_index_vlm"

    meta_overrides: Dict[str, Any] = {
        "window_sec": 0,
        "srt_source": "",
        "anchor": "zero",
        "sources": ("vlm_scene",),
        "extra_files": {
            "visual_csv": str(csv_path),
            "visual_context_json": (str(ctx_json) if ctx_json else ""),
            "visual_metadata_txt": (str(meta_txt) if meta_txt else "")
        }
    }
    idx = TranscriptRAGIndex.build_from_dataframe(
        df=df,
        outdir=outdir,
        model_name=model_name,
        device=device,
        meta_overrides=meta_overrides
    )
    return str(idx.root)
