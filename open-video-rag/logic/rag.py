from typing import Optional, Tuple, List, Dict
from utils.common import format_source_label, normalize_entity_key
from utils.visual import find_visual_artifacts_in
from resources.consts import *
import numpy as np
import json
from pathlib import Path


def compile_context_blocks(idx, query: str, top_k: int, method: str, alpha: float,
                           rerank: str, rerank_model: str, overfetch: int,
                           ctx_before: int, ctx_after: int,
                           device: Optional[str], embed_model_override: Optional[str]) -> Tuple[List[Dict], str]:
    if method in ("bm25", "embed"):
        base_hits = (idx.query_bm25(query, k=max(top_k, overfetch))
                     if method == "bm25"
                     else idx.query_embed(query, k=max(top_k, overfetch),
                                          model_name=embed_model_override, device=device))
    else:
        base_hits = idx.query_hybrid(query, k=top_k, method=method, alpha=alpha,
                                     overfetch=overfetch, model_name=embed_model_override, device=device)

    if rerank == "cross":
        hits = idx.rerank_cross(query, base_hits, model_name=rerank_model, device=device, top_k=top_k)
    elif rerank == "mmr":
        hits = idx.rerank_mmr(query, base_hits, top_k=top_k, lambda_mult=0.7,
                              model_name=embed_model_override, device=device)
    else:
        hits = base_hits[:top_k]

    enriched = idx.attach_context(hits, before=ctx_before, after=ctx_after)


    blocks = []
    for h in enriched:
        source_label = format_source_label(h.get("source"))
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        lines: List[str] = []
        for c in ctx_sorted:
            line_source = format_source_label(c.get("source"))
            lines.append(f"[{c['start_srt']} - {c['end_srt']}] ({line_source}) {c['text']}".strip())
        header = "### Passage {rank} | Source: {source} | Main: {start} - {end}".format(
            rank=h.get('rank'),
            source=source_label,
            start=h.get('start_srt'),
            end=h.get('end_srt'),
        )
        block = header + "\n" + "\n".join(lines)
        blocks.append(block)
    full_context = "\n\n".join(blocks)
    return enriched, full_context


def compile_context_blocks_multi(
    indexes: List[Tuple[object, str]],  # [(idx_obj, "SRT"/"VLM"), ...]
    query: str,
    top_k: int,
    method: str, alpha: float,
    rerank: str, rerank_model: str, overfetch: int,
    ctx_before: int, ctx_after: int,
    device: Optional[str], embed_model_override: Optional[str]
) -> Tuple[List[Dict], str]:
    """
    Split top_k across indexes (fair share), then merge; tag source in header.
    """
    if not indexes:
        return [], ""
    share = max(1, int(np.ceil(top_k / max(1, len(indexes)))))
    all_hits: List[Dict] = []
    blocks = []
    for idx_obj, label in indexes:
        hits, ctx_text = compile_context_blocks(
            idx=idx_obj, query=query, top_k=share, method=method, alpha=alpha,
            rerank=rerank, rerank_model=rerank_model, overfetch=overfetch,
            ctx_before=ctx_before, ctx_after=ctx_after,
            device=device, embed_model_override=embed_model_override
        )
        # Prefix block to separate sources
        if ctx_text.strip():
            blocks.append(f"#### Source: {label}\n{ctx_text}")
        # Stamp source label into hits for the timecode list
        for h in hits:
            h["source"] = label.lower() if label in ("VLM","SRT") else (h.get("source") or "")
        all_hits.extend(hits)
    # Keep at most top_k by current order (simple interleave/concat)
    all_hits = all_hits[:top_k]
    full_context = "\n\n".join(blocks)
    return all_hits, full_context



def auto_select_sources_from_query(q: str) -> tuple[bool, bool, str]:
    """
    Returns (want_transcript, want_visual, reason)
    Strategy:
      - Only visual terms -> visual
      - Only transcript terms -> transcript
      - Both (or none) -> both (be generous)
    """
    s = (q or "").lower()
    v_hits = sum(1 for pat in VISUAL_PATTERNS if re.search(pat, s))
    t_hits = sum(1 for pat in TRANSCRIPT_PATTERNS if re.search(pat, s))

    if v_hits > 0 and t_hits == 0:
        return (False, True, "auto: visual cues only")
    if t_hits > 0 and v_hits == 0:
        return (True, False, "auto: transcription cues only")
    # ambiguous or none -> both
    return (True, True, "auto: ambiguous or mixed → both")



def load_entities_scenes_from_context(outputs_dir: str) -> tuple[dict, dict]:
    """
    Returns (entities_scenes, original_key_by_norm).
    entities_scenes: {original_key: [scene_ids]}
    original_key_by_norm: {normalized_key: original_key}
    """
    art = find_visual_artifacts_in(outputs_dir)
    cpath = art.get("context")
    if not cpath or not Path(cpath).exists():
        return {}, {}
    try:
        data = json.loads(Path(cpath).read_text(encoding="utf-8"))
        es = data.get("entities_scenes") or {}
        # build normalized map → pick longest/original repr deterministically
        norm_map = {}
        for k in es.keys():
            nk = normalize_entity_key(k)
            if nk and nk not in norm_map:
                norm_map[nk] = k
        return es, norm_map
    except Exception:
        return {}, {}
