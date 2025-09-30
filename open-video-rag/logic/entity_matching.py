from typing import Dict, List, Optional
import json
from utils.common import extract_json, normalize_entity_key
from chat.llm import llm_chat
import difflib


def llm_match_entities_to_keys(
    provider: str,
    cfg: Dict,
    query_entities: List[str],
    candidate_keys: List[str],
) -> Dict:
    """
    Ask the LLM to select which candidate_keys correspond to the query_entities.
    Returns: {"matches": [{"query":"nike","keys":["Nike","NIKE®"]}, ...], "flat_keys": ["Nike", ...]}
    """
    # Keep payload tight; if many keys, send top N (but entities_scenes keys are usually modest)
    system = (
        "You are matching user-specified entities to canonical keys extracted from a video analysis.\n"
        "Given: (a) query entity strings, (b) the exact list of candidate keys (canonical forms).\n"
        "Rules:\n"
        "• Be robust to casing, accents, punctuation, and common variants (e.g., 'PSG', 'Paris Saint-Germain').\n"
        "• ONLY choose from the provided candidate keys; do not invent.\n"
        "• Prefer exact/near-exact brand/org names over generic words.\n"
        "• If none match for a query entity, return an empty list for that entity.\n"
        "Output JSON ONLY with:\n"
        '{"matches":[{"query":"<q>","keys":["<k1>","<k2>"]},...], "flat_keys":["<k1>","<k2>",...]}'
    )
    user = json.dumps({
        "query_entities": query_entities,
        "candidate_keys": candidate_keys,
    }, ensure_ascii=False)
    try:
        out = llm_chat(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=600,
        )
        data = extract_json(out) or {}
    except Exception as e:
        data = {"matches": [], "flat_keys": [], "error": str(e)}
    # sanitize
    matches = data.get("matches") or []
    flat = data.get("flat_keys") or []
    flat = [str(k) for k in flat if str(k).strip()]
    # If LLM didn't provide flat_keys, derive from matches
    if not flat and matches:
        seen = set()
        for m in matches:
            for k in (m.get("keys") or []):
                k = str(k)
                if k and k not in seen:
                    seen.add(k); flat.append(k)
    return {"matches": matches, "flat_keys": flat}


def fuzzy_pick_entity(name: str, norm_map: dict, threshold: float = 0.82) -> Optional[str]:
    """
    Return the ORIGINAL key from entities_scenes that best matches the user's entity name.
    Strategy:
      1) exact normalized key
      2) difflib closest
      3) token-set containment heuristic
    """
    if not name or not norm_map:
        return None
    nk = normalize_entity_key(name)
    if nk in norm_map:
        return norm_map[nk]

    candidates = list(norm_map.keys())
    # difflib
    close = difflib.get_close_matches(nk, candidates, n=1, cutoff=threshold)
    if close:
        return norm_map[close[0]]

    # token-set overlap
    toks = set(nk.split())
    best, best_j = None, 0.0
    for cand in candidates:
        ctoks = set(cand.split())
        j = len(toks & ctoks) / max(1, len(toks | ctoks))
        if j > best_j:
            best, best_j = cand, j
    if best and best_j >= 0.6:
        return norm_map[best]
    return None
