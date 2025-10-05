from typing import List, Dict, Optional, Tuple

from resources.consts import *
from chat.llm import llm_chat
from utils.common import format_source_label, normalize_dash

def trim_history_messages(history_msgs: List[Dict], max_turns: int = 8, max_chars: int = 6000) -> List[Dict]:
    """
    Return recent history as a list of chat messages for the model,
    limited by turns and approx characters.
    """
    if not history_msgs:
        return []
    recent = history_msgs[-max_turns:]
    out: List[Dict] = []
    total = 0
    for m in recent:
        content = str(m.get("content", "")).strip()
        line_len = len(content)
        if total + line_len > max_chars:
            # If adding would exceed cap, try to add a truncated version
            content = content[: max(0, max_chars - total)]
        out.append({"role": m.get("role", "user"), "content": content})
        total += len(content)
        if total >= max_chars:
            break
    return out


def contextualize_question(
    provider: str,
    cfg: Dict,
    history_msgs: List[Dict],
    latest_user_msg: str,
) -> str:
    """
    Use an LLM to rewrite the latest user message into a standalone question,
    leveraging the recent chat history.
    """
    history_text = format_history_as_text(history_msgs, max_turns=10, max_chars=6000)
    system = (
        "You rewrite user questions into a single, clear standalone question.\n"
        "Use the conversation so far to resolve pronouns, references, and ellipses.\n"
        "Output ONLY the rewritten question, with no commentary or quotes."
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message:\n{latest_user_msg}\n\n"
        f"Rewritten standalone question:"
    )
    try:
        out = llm_chat(
            provider=provider,
            cfg=cfg,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
        )
        # sanitize trivial outputs
        cleaned = (out or "").strip()
        # avoid empty or overly short fallback
        return cleaned if len(cleaned) >= 2 else str(latest_user_msg)
    except Exception:
        return str(latest_user_msg)


def messages_preview_text(messages: List[Dict]) -> Tuple[str, int, int]:
    """
    Build a readable preview of what will be sent to the LLM and compute lengths.
    Returns (preview_text, total_chars, approx_tokens).
    Token approximation uses 4 chars/token heuristic.
    """
    parts = []
    total_chars = 0

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # content may be a list (Anthropic content blocks); normalize to text
        if isinstance(content, list):
            text = "\n".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content])
        else:
            text = str(content)
        header = f"--- {role.upper()} ---"
        parts.append(header)
        parts.append(text)
        total_chars += len(text)

    preview = "\n".join(parts)
    approx_tokens = max(1, total_chars // 4)
    return preview, total_chars, approx_tokens


def redact_cfg_for_preview(cfg: Dict) -> Dict:
    """
    Return a cfg copy suitable for preview (mask keys).
    """
    safe = dict(cfg)
    for k in list(safe.keys()):
        if "key" in k.lower():
            v = safe.get(k) or ""
            if v:
                safe[k] = v[:4] + "…" + v[-2:]
    return safe


# ---------------------
# History helpers (NEW)
# ---------------------
def format_history_as_text(history_msgs: List[Dict], max_turns: int = 8, max_chars: int = 6000) -> str:
    """
    Convert recent chat history into a compact text transcript for prompting.
    Keeps the last `max_turns` messages (user+assistant counts as 2).
    Also enforces an approximate char cap.
    """
    if not history_msgs:
        return ""
    # keep last N
    recent = history_msgs[-max_turns:]
    lines: List[str] = []
    total = 0
    for m in recent:
        role = m.get("role", "user")
        content = str(m.get("content", "")).strip()
        pref = "User" if role == "user" else "Assistant"
        line = f"{pref}: {content}"
        total += len(line)
        lines.append(line)
        if total >= max_chars:
            break
    return "\n".join(lines)



def sanitize_scene_output(text: str, allowed_nums: set[int], times_by_num: dict[int, tuple[str,str]]) -> str:
    lines = text.splitlines()
    out = []
    for ln in lines:
        m = SC_LINE_RE.search(ln)
        if not m:
            out.append(ln); continue
        try:
            num = int(m.group(1))
        except Exception:
            # skip malformed scene line
            continue
        if num not in allowed_nums:
            # Drop hallucinated scene
            continue
        start, end = times_by_num.get(num, ("00:00:00,000","00:00:00,000"))
        # Rebuild canonical prefix
        prefix = f"SCENE N°{num}: {start}–{end}"
        # Replace everything up to dash and keep the rest
        # Find " — " or " - " separator; if missing, add an em dash
        parts = re.split(r"\s+—\s+|\s+-\s+", normalize_dash(ln), maxsplit=1)
        suffix = parts[1] if len(parts) > 1 else ln[m.end():].lstrip(" :-—–")
        clean = f"{prefix} — {suffix.strip()}" if suffix.strip() else prefix
        out.append(clean)
    return "\n".join(out)
