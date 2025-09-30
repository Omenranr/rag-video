import requests
import json
from typing import List, Dict, Tuple


def _anthropic_convert_messages(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Split out system content and convert messages to Anthropic Messages API format."""
    system_parts: List[str] = []
    converted: List[Dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Collect system content separately
        if role == "system":
            if isinstance(content, list):
                # concatenate any text parts
                txt = "\n\n".join(
                    [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                )
                system_parts.append(txt)
            else:
                system_parts.append(str(content))
            continue

        def to_text_list(c):
            if isinstance(c, list):
                # already a list of content blocks
                blocks = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        blocks.append(p)
                    else:
                        blocks.append({"type": "text", "text": str(p)})
                return blocks
            return [{"type": "text", "text": str(c)}]

        if role in ("user", "assistant"):
            converted.append({"role": role, "content": to_text_list(content)})
        else:
            # default to user
            converted.append({"role": "user", "content": to_text_list(content)})

    system_str = "\n\n".join([s for s in system_parts if s])
    return system_str, converted


def anthropic_chat_stream(
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
):
    """
    Stream tokens from Anthropic Messages API.
    Yields text deltas as they arrive.
    """
    system_str, conv = _anthropic_convert_messages(messages)
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: Dict = {
        "model": model,
        "messages": conv,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if system_str:
        payload["system"] = system_str

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw.split("data:", 1)[1].strip()
            if data == "" or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {}).get("text", "")
                    if delta:
                        yield delta
            except Exception:
                continue


def anthropic_chat(
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    """Call Anthropic Messages API and return assistant text."""
    system_str, conv = _anthropic_convert_messages(messages)
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: Dict = {
        "model": model,
        "messages": conv,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_str:
        payload["system"] = system_str

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Anthropic error {resp.status_code}: {resp.text}") from e
    data = resp.json()
    # content is a list of blocks; return concatenated text
    blocks = data.get("content", [])
    text = "\n".join([b.get("text", "") for b in blocks if isinstance(b, dict)])
    return text
