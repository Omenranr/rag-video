from utils.common import html_escape, safe_stem, format_source_label
from typing import List, Optional, Dict
from pathlib import Path
import csv as _csv

from resources.consts import *

def find_existing_outputs_for_video(video_path: str, outputs_root: str) -> List[str]:
    """
    Look under outputs_root for subfolders starting with safe_stem(video) + '_'.
    We NO LONGER require transcript.srt to list them.
    """
    root = Path(outputs_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    stem = safe_stem(Path(video_path))
    pattern = f"{stem}_*"
    matches = [p for p in root.glob(pattern) if p.is_dir()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in matches]


def find_visual_artifacts_in(outputs_dir: str) -> Dict[str, Optional[Path]]:
    p = Path(outputs_dir)
    csvs = sorted(p.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    ctxs = sorted(p.glob("*_context.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    metas = sorted(p.glob("*_metadata.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return {
        "csv": (csvs[0] if csvs else None),
        "context": (ctxs[0] if ctxs else None),
        "meta": (metas[0] if metas else None),
    }


def has_visual_artifacts(outputs_dir: str) -> bool:
    a = find_visual_artifacts_in(outputs_dir)
    return bool(a["csv"] and a["context"] and a["meta"])


def list_videos(folder: str) -> List[str]:
    p = Path(folder).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    hits = []
    for ext in VIDEO_EXTS:
        hits.extend(str(x) for x in p.rglob(f"*{ext}"))
    hits.sort()
    return hits


def build_visual_index(outputs_dir: str, embed_model: str, embed_device: Optional[str]):
    from visual_rag_index import build_visual_index_from_outputs  # new module
    return build_visual_index_from_outputs(
        outputs_dir=outputs_dir,
        model_name=embed_model,
        device=(None if embed_device == "auto" else embed_device)
    )


def load_scene_rows_from_csv(outputs_dir: str, scene_ids: list[int]) -> list[dict]:
    """
    Read *.csv produced by visual_extraction to get generated_text + timecodes for given scenes.
    Returns list of rows: {scene_id:int, start_timecode:str, end_timecode:str, generated_text:str}
    """
    art = find_visual_artifacts_in(outputs_dir)
    csv_path = art.get("csv")
    rows = []
    if not csv_path or not Path(csv_path).exists():
        return rows
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for rec in r:
                try:
                    sid = int(rec.get("scene_id", "0"))
                except Exception:
                    continue
                if sid in scene_ids:
                    rows.append({
                        "scene_id": sid,
                        "start_timecode": rec.get("start_timecode") or rec.get("start_tc") or "00:00:00,000",
                        "end_timecode": rec.get("end_timecode") or rec.get("end_tc") or "00:00:00,000",
                        "generated_text": rec.get("generated_text", "") or "",
                    })
    except Exception:
        return []
    # keep scene order
    rows.sort(key=lambda x: x["scene_id"])
    return rows


def hits_from_scene_rows(scene_rows: list[dict]) -> list[dict]:
    """
    Convert scene rows into 'hits' compatible with the context panel + timecode radio.
    """
    hits = []
    for i, r in enumerate(scene_rows, start=1):
        start, end = r["start_timecode"], r["end_timecode"]
        txt = r["generated_text"].strip()
        hits.append({
            "rank": i,
            "start_srt": start,
            "end_srt": end,
            "source": "vlm",
            "method": "entities_scenes",
            "hretr_score": None,
            "rerank_score": None,
            "context": [{
                "offset": 0,
                "start_srt": start,
                "end_srt": end,
                "text": txt,
                "source": "vlm",
            }],
        })
    return hits



def ctx_md_from_hits_aggregated(enriched_hits: list[dict], title: str = "Retrieved passages") -> str:
    if not enriched_hits:
        return f"<h3>{html_escape(title)}</h3><em>(no passages retrieved)</em>"

    html = [
        "<style>",
        "details > summary { cursor: pointer; }",
        ".ctx-block { margin: 6px 0 12px 0; padding: 8px 10px; border: 1px solid #333;"
        " border-radius: 8px; background: #0b0b0b; }",
        ".ctx-line { display: block; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        ".ctx-meta { opacity: .8; font-size: .9em; }",
        "</style>",
        f"<h3>{html_escape(title)}</h3>"
    ]

    for h in enriched_hits:
        parts = [h.get("method","")]
        if "hretr_score" in h:
            try: parts.append(f"hyb={float(h['hretr_score']):.4f}")
            except Exception: pass
        if h.get("rerank_score") is not None:
            try: parts.append(f"rerank={float(h['rerank_score']):.4f}")
            except Exception: pass
        method_lbl = ", ".join([p for p in parts if p]) or "retrieval"
        src_lbl = format_source_label(h.get("source"))
        header = f"[{int(h.get('rank',0)):02}] {h.get('start_srt','??')} → {h.get('end_srt','??')}  ({src_lbl} • {method_lbl})"

        ctx_sorted = sorted(h.get("context", []), key=lambda c: (c.get("offset",0) != 0, c.get("offset",0)))
        body_lines = []
        for c in ctx_sorted:
            bullet = "•" if c.get("offset",0) == 0 else "·"
            tc = f"{c.get('start_srt','??')}–{c.get('end_srt','??')}"
            txt = html_escape((c.get("text","") or "").strip())
            body_lines.append(f"<span class='ctx-line'><span class='ctx-meta'>{bullet} [{tc}]</span> {txt}</span>")

        html.append(
            f"<details><summary>{html_escape(header)}</summary>"
            f"<div class='ctx-block'>{''.join(body_lines)}</div>"
            f"</details>"
        )
    return "\n".join(html)
