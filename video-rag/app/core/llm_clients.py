from __future__ import annotations
import json
import requests
from typing import Dict, List, Tuple, Optional


# ---------------------
# IBM watsonx.ai
# ---------------------
def watsonx_chat(
    base_url: str,
    api_key: str,
    model: str,
    project_id: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
    version: str = "2024-10-10",
) -> str:
    # IAM token
    iam_resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
        timeout=timeout,
    )
    iam_resp.raise_for_status()
    iam_token = iam_resp.json()["access_token"]


    def _wx_content(role: str, content):
        if isinstance(content, list):
            return content
        if role == "user":
            return [{"type": "text", "text": str(content)}]
        if role == "system":
            return str(content)
        if role == "assistant":
            return str(content)
        return [{"type": "text", "text": str(content)}]


    wx_messages = [
        {"role": m.get("role", "user"), "content": _wx_content(m.get("role", "user"), m.get("content", ""))}
        for m in messages
    ]


    url = f"{base_url.rstrip('/')}/ml/v1/text/chat?version={version}"
    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model_id": model,
        "project_id": project_id,
        "messages": wx_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Bad LLM response: {data}")
    return data["choices"][0]["message"]["content"]


# ---------------------
# OpenAI-compatible
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
    root = (base_url or "https://api.openai.com").rstrip("/")
    url = f"{root}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------
# Anthropic Messages API
# ---------------------
def _anthropic_convert_messages(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    system_parts: List[str] = []
    converted: List[Dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, list):
                txt = "\n\n".join([p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"])
                system_parts.append(txt)
            else:
                system_parts.append(str(content))
        continue

    def to_text_list(c):
        if isinstance(c, list):
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
        converted.append({"role": "user", "content": to_text_list(content)})

    system_str = "\n\n".join([s for s in system_parts if s])
    return system_str, converted


def anthropic_chat(
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    system_str, conv = _anthropic_convert_messages(messages)
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload: Dict = {"model": model, "messages": conv, "max_tokens": max_tokens, "temperature": temperature}
    if system_str:
        payload["system"] = system_str

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    blocks = data.get("content", [])
    text = "\n".join([b.get("text", "") for b in blocks if isinstance(b, dict)])
    return text


# ---------------------
# Unified wrapper
# ---------------------
def llm_chat(
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    provider = (provider or "watsonx").lower()
    if provider == "watsonx":
        return watsonx_chat(
            base_url=cfg.get("wx_base_url", ""),
            api_key=cfg.get("wx_api_key", ""),
            model=cfg.get("wx_model", ""),
            project_id=cfg.get("wx_project_id", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "openai":
        return openai_chat(
            api_key=cfg.get("oa_api_key", ""),
            model=cfg.get("oa_model", ""),
            messages=messages,
            base_url=cfg.get("oa_base_url") or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "anthropic":
        return anthropic_chat(
            api_key=cfg.get("an_api_key", ""),
            model=cfg.get("an_model", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")