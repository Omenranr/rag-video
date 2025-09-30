from typing import Dict, List
from chat.openai import *
from chat.anthropic import *


def llm_chat_stream(
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
):
    """
    Return a generator that yields text chunks from the selected provider.
    """
    p = (provider or "openai").lower()
    if p == "openai":
        return openai_chat_stream(
            api_key=cfg.get("oa_api_key",""),
            model=cfg.get("oa_model",""),
            messages=messages,
            base_url=cfg.get("oa_base_url") or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif p == "anthropic":
        return anthropic_chat_stream(
            api_key=cfg.get("an_api_key",""),
            model=cfg.get("an_model",""),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider for streaming: {provider}")


# Unified chat wrapper
def llm_chat(
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    timeout: int = 120,
) -> str:
    provider = (provider or "anthropic").lower()
    if provider == "openai":
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
