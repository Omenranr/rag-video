from typing import Optional, List, Dict
import re
from resources.consts import ENTITY_INTENT_PATTERNS, RECENCY_PATTERNS, YEAR_RE
from utils.common import extract_json
from utils.chat import format_history_as_text
from chat.llm import llm_chat


def detect_entity_intent(question: str) -> tuple[bool, Optional[str]]:
    q = (question or "").strip()
    for pat in ENTITY_INTENT_PATTERNS:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            name = (m.group("name") or "").strip().strip("'\"“”‘’.,:;!?")
            # prune trailing generic words
            name = re.sub(r"\b(?:appears?|appara[iî]t|figure|present[e]?|se voit)\b.*$", "", name, flags=re.I).strip()
            return True, name
    # Very short pattern: "nike scenes", "scenes nike"
    if re.search(r"\bscenes?\b", q, flags=re.I):
        # grab a plausible last token phrase
        tail = re.sub(r".*\bscenes?\b", "", q, flags=re.I).strip()
        if tail:
            return True, tail.strip(" .,:;!?")
    return False, None


def wants_recent_info(msg: str, current_year: Optional[int] = None) -> tuple[bool, list[int]]:
    """
    Return (needs_recent, years_mentioned).
    needs_recent is True if message asks for 'now/latest/recent' or mentions a year
    that is close to the present (>= current_year-1).
    """
    s = (msg or "").lower()
    if any(re.search(p, s) for p in RECENCY_PATTERNS):
        years = [int(y) for y in YEAR_RE.findall(s)]
        return True, years

    years = [int(y) for y in YEAR_RE.findall(s)]
    if years and (current_year is not None) and any(y >= current_year - 1 for y in years):
        return True, years
    return False, years


# ---------------------
# Search decision helper (history-aware, NEW)
# ---------------------
def wants_web_search_explicit(user_msg: str) -> bool:
    """
    Heuristic to flag explicit web search intent (English + French).
    """
    s = (user_msg or "").lower()
    patterns = [
        r"\b(search|google|web|internet|online|look\s*up|check\s*online|find on the web)\b",
        r"cherche( r)? sur (le|la|les)?\s*web|internet",
        r"recherche en ligne",
        r"regarde sur internet",
        r"sur le web",
    ]
    return any(re.search(p, s) for p in patterns)



def llm_detect_intent_entities(
    provider: str,
    cfg: Dict,
    latest_user_msg: str,
    history_msgs: List[Dict],
) -> Dict:
    """
    Ask the LLM to (a) detect intent, (b) extract entity strings if intent=entity_search.
    Returns: {"intent": "entity_search"|"other", "entities": [str], "reason": str}
    """
    history_text = format_history_as_text(history_msgs, max_turns=8, max_chars=3000)
    system = (
        "You are a routing classifier.\n"
        "Task: Decide if the latest user message is an ENTITY-SEARCH about a video, e.g., "
        "\"give me all scenes where Nike appeared\" / \"scenes with PSG logo\".\n"
        "If it IS entity-search: extract the entity mentions as a list of short strings.\n"
        "If it is NOT entity-search: intent='other' and entities=[].\n"
        "STRICT OUTPUT: JSON ONLY with keys: intent, entities, reason.\n"
        "• intent ∈ {'entity_search','other'}\n"
        "• entities = array of strings (may be 1+)\n"
        "• reason = short justification"
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message:\n{latest_user_msg}\n\n"
        "Output JSON ONLY like:\n"
        '{"intent":"entity_search","entities":["nike","psg"],"reason":"…"}'
    )
    try:
        out = llm_chat(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=250,
        )
        data = extract_json(out) or {}
    except Exception as e:
        data = {"intent":"other","entities":[],"reason":f"llm error: {e}"}
    intent = (data.get("intent") or "other").strip().lower()
    ents = [str(x).strip() for x in (data.get("entities") or []) if str(x).strip()]
    return {"intent": ("entity_search" if intent=="entity_search" and ents else "other"),
            "entities": ents, "reason": data.get("reason","")}


def llm_decide_search(
    provider: str,
    cfg: Dict,
    question: str,
    transcript_context: str,
    explicit_flag: bool,
    history_text: str = "",
    *,                           # NEW: only keyword args after this
    recency_flag: bool = False,  # NEW
    years_mentioned: Optional[list[int]] = None,  # NEW
    current_year: Optional[int] = None,           # NEW
) -> Dict:
    """
    Returns dict: {need_search, query, answer, reason}
    """
    years_mentioned = years_mentioned or []
    system = (
        "You are a retrieval QA router.\n"
        "Inputs: (a) conversation so far, (b) a standalone user question, (c) transcript context,\n"
        "and (d) recency hints (flags + years mentioned).\n"
        "RULES (strict):\n"
        "• If the user explicitly asks to search the web → need_search=true.\n"
        "• If recency_flag=true OR the question asks for 'now', 'latest', 'most recent', 'today',\n"
        "  OR mentions a year >= (current_year-1) → need_search=true (regardless of transcript context).\n"
        "• Otherwise, if the answer is clearly in transcript context → need_search=false and answer with it.\n"
        "• Otherwise → need_search=true.\n\n"
        "Return JSON ONLY with schema:\n"
        '{"need_search": true|false, "query": string|null, "answer": string|null, "reason": string}\n'
        "If need_search=true: provide ONE precise web search string in 'query' (include any relevant years), 'answer' MUST be null.\n"
        "If need_search=false: 'answer' MUST contain the final answer grounded in transcript context, 'query' MUST be null."
    )

    user = (
        f"explicit_web_search_flag={explicit_flag}\n"
        f"recency_flag={recency_flag}\n"
        f"years_mentioned={years_mentioned}\n"
        f"current_year={current_year}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Standalone Question:\n{question}\n\n"
        f"Transcript context:\n{transcript_context}\n"
    )

    out = llm_chat(
        provider=provider,
        cfg=cfg,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=500,
    )
    data = extract_json(out) or {}
    need = bool(data.get("need_search"))
    query = (data.get("query") or "").strip() if need else None
    answer = (data.get("answer") or "").strip() if not need else None
    reason = (data.get("reason") or "").strip()
    return {"need_search": need, "query": query, "answer": answer, "reason": reason}
