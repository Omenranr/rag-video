import requests
import json
from typing import List, Dict, Optional


def openai_chat_stream(
    api_key: str,
    model: str,
    messages: List[Dict],
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
):
    """
    Stream tokens from OpenAI-compatible Chat Completions API.
    Yields text deltas as they arrive.
    """
    root = (base_url or "https://api.openai.com").rstrip("/")
    url = f"{root}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "" or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except Exception:
                continue


# ---------------------
# LLM clients (OpenAI, Anthropic)
# ---------------------
def openai_chat(
    api_key: str,
    model: str,
    messages: List[Dict],
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    """Call OpenAI-compatible Chat Completions API and return assistant text."""
    root = (base_url or "https://api.openai.com").rstrip("/")
    url = f"{root}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}") from e
    data = resp.json()
    return data["choices"][0]["message"]["content"]

