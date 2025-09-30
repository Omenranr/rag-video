from __future__ import annotations


import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from .helpers import latest_rag_index_dir


# Cache of loaded indexes: {index_dir: TranscriptRAGIndex}
_INDEX_CACHE: Dict[str, object] = {}


def run_extractor(
    video_path: str,
    lang: str,
    vclass: str,
    fps: Optional[int],
    device: str,
    batch_size: int,
    conf: float,
    iou: float,
    max_det: int,
    progress=None,
) -> str:
    """
    Returns output directory containing: transcript.srt, detections.csv, detections_by_frame.csv
    Lazy-imports heavy code; falls back to subprocess if not importable.
    """
    local_ok = False
    try:
        from video_audio_multitool import run_pipeline as extractor_run_pipeline # type: ignore
        local_ok = True
    except Exception:
        local_ok = False

    if local_ok:
        if progress:
            progress(0.05, desc="Starting extractor…")
        out_dir = extractor_run_pipeline(
            input_arg=video_path, lang=lang, vclass=vclass, fps=fps,
            device_choice=device, batch_size=batch_size,
            det_conf=conf, det_iou=iou, det_max=max_det
        )
        return str(out_dir)
    else:
        import subprocess
        cmd = [
        sys.executable, "video_audio_multitool.py", video_path,
        "--lang", lang, "--vclass", vclass, "--device", device,
        "--batch-size", str(batch_size), "--det-conf", str(conf),
        "--det-iou", str(iou), "--max-det", str(max_det)
        ]
        if fps:
            cmd += ["--fps", str(fps)]
        if progress:
            progress(0.05, desc="Starting extractor (subprocess)…")
        subprocess.run(cmd, check=True)
        outs = sorted((Path.cwd() / "outputs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not outs:
            raise RuntimeError("Extractor finished but no outputs/ folder found.")
        return str(outs[0])


def build_index_from_srt(
    transcript_path: str,
    window: int,
    anchor: str,
    embed_model: str,
    device: Optional[str],
    progress=None,
) -> str:
    """
    Returns path to built index folder (rag_index_*)
    Lazy-imports heavy code; falls back to subprocess if not importable.
    """
    srt = Path(transcript_path)
    if not srt.exists():
        raise FileNotFoundError(f"Transcript not found: {srt}")


    local_ok = False
    try:
        from transcript_hybrid_rag import TranscriptRAGIndex # type: ignore
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
        from transcript_hybrid_rag import TranscriptRAGIndex # type: ignore
        idx = TranscriptRAGIndex.load(Path(key))
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location("transcript_hybrid_rag", Path("transcript_hybrid_rag.py"))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod) # type: ignore
        idx = mod.TranscriptRAGIndex.load(Path(key)) # type: ignore
    _INDEX_CACHE[key] = idx
    return idx


def compile_context_blocks(
    idx,
    query: str,
    top_k: int,
    method: str,
    alpha: float,
    rerank: str,
    rerank_model: str,
    overfetch: int,
    ctx_before: int,
    ctx_after: int,
    device: Optional[str],
    embed_model_override: Optional[str],
):
    if method in ("bm25", "embed"):
        base_hits = (
            idx.query_bm25(query, k=max(top_k, overfetch))
            if method == "bm25"
            else idx.query_embed(
                query, k=max(top_k, overfetch), model_name=embed_model_override, device=device
            )
        )
    else:
        base_hits = idx.query_hybrid(
            query, k=top_k, method=method, alpha=alpha, overfetch=overfetch,
            model_name=embed_model_override, device=device
        )


    if rerank == "cross":
        hits = idx.rerank_cross(query, base_hits, model_name=rerank_model, device=device, top_k=top_k)
    elif rerank == "mmr":
        hits = idx.rerank_mmr(
            query, base_hits, top_k=top_k, lambda_mult=0.7,
            model_name=embed_model_override, device=device
        )
    else:
        hits = base_hits[: top_k]


    enriched = idx.attach_context(hits, before=ctx_before, after=ctx_after)


    blocks = []
    for h in enriched:
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        lines = [f"[{c['start_srt']}–{c['end_srt']}] {c['text']}".strip() for c in ctx_sorted]
        block = f"### Passage {h['rank']} | Main: {h['start_srt']}–{h['end_srt']}\n" + "\n".join(lines)
        blocks.append(block)
    full_context = "\n\n".join(blocks)
    return enriched, full_context