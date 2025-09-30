from typing import Dict, List
import requests

# ---------------------
# Exa web search
# ---------------------
def exa_search_with_contents(query: str, api_key: str, num_results: int = 5, timeout: int = 60) -> List[Dict]:
    """
    Returns list of {title, url, snippet} using Exa /search + /contents.
    """
    base = "https://api.exa.ai"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 1) Search
    sr = requests.post(
        f"{base}/search",
        headers=headers,
        json={
            "query": query,
            "numResults": int(num_results),
            "type": "neural",
            "useAutoprompt": False,
        },
        timeout=timeout,
    )
    sr.raise_for_status()
    sdata = sr.json()
    results = sdata.get("results", []) or []

    if not results:
        return []

    # 2) Contents for each result (if not already present)
    ids = [r.get("id") for r in results if r.get("id")]
    content_by_id: Dict[str, str] = {}
    if ids:
        cr = requests.post(
            f"{base}/contents",
            headers=headers,
            json={"ids": ids},
            timeout=timeout,
        )
        if cr.ok:
            cdata = cr.json()
            for item in cdata.get("results", []):
                content_by_id[item.get("id", "")] = (item.get("text") or "").strip()

    out: List[Dict] = []
    for r in results:
        rid = r.get("id", "")
        title = r.get("title") or r.get("url") or "(untitled)"
        url = r.get("url") or ""
        text = (r.get("text") or "").strip()
        text = text or content_by_id.get(rid, "")
        if len(text) > 800:
            text = text[:790].rstrip() + "…"
        out.append({"title": title, "url": url, "snippet": text})
    return out

