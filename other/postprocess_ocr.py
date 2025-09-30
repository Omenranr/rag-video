"""
postprocess_ocr_words.py

Now the main output (default --out) is a WORD-CENTRIC JSONL where each line groups a single word and all its occurrences:

Example line:
{"type": "word", "word": "agents", "timestamps": [0.43, 1.27, 5.02], "occurrences": [{"ts":0.43,"bbox":[...],"conf":0.91},{...}], "count": 3}

Changes vs original:
- Keeps normalize/filter pipeline.
- (Optional) still lets you export merged time segments with --segments-out.
- By default, --out writes the grouped words; no more duplicated rows for the same token.

Usage:
    python postprocess_ocr_words.py input.csv \
        --out word_index.jsonl \
        --segments-out ocr_segments.jsonl \
        --dict frequency_dictionary_en_82_765.txt

If you don't need segments, omit --segments-out.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Iterable, Optional

import pandas as pd
from rapidfuzz import fuzz

try:
    from symspellpy import SymSpell, Verbosity  # optional
except ImportError:  # pragma: no cover
    SymSpell = None  # type: ignore
    Verbosity = None  # type: ignore

# -------------------- Tunables -------------------- #
CONF_THR = 0.85          # discard OCR rows below this confidence
SIM_THR = 90             # RapidFuzz similarity threshold for segment merging
MAX_GAP_S = 1.0          # seconds; max gap to extend a segment

# -------------------- Helpers -------------------- #

def load_symspell(dict_path: str) -> Optional[SymSpell]:
    if not dict_path or SymSpell is None:
        return None
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym


def normalize(txt: str, sym: Optional[SymSpell]) -> str:
    t = " ".join(txt.strip().lower().split())
    if sym:
        sug = sym.lookup(t, Verbosity.TOP, max_edit_distance=2)
        if sug and sug[0].distance <= 2:
            t = sug[0].term
    return t


@dataclass
class Segment:
    text: str
    start: float
    end: float
    avg_conf: float
    count: int
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return {
            "type": "ocr_segment",
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "avg_conf": round(self.avg_conf, 3),
            "frames": self.count,
            "bboxes": self.bboxes,
        }


def merge_segments(rows: Iterable[Dict[str, Any]], sim_thr: int = SIM_THR, max_gap: float = MAX_GAP_S) -> List[Segment]:
    segs: List[Segment] = []
    cur: Optional[Segment] = None
    for r in rows:
        ts, text, conf, box = r["frame_ts"], r["text"], r["conf"], r["bbox"]
        if cur is None:
            cur = Segment(text=text, start=ts, end=ts, avg_conf=conf, count=1, bboxes=[box])
            continue
        same_time = ts - cur.end <= max_gap
        similar = fuzz.token_set_ratio(text, cur.text) >= sim_thr
        if same_time and similar:
            cur.end = ts
            cur.avg_conf = (cur.avg_conf * cur.count + conf) / (cur.count + 1)
            cur.count += 1
            cur.bboxes.append(box)
        else:
            segs.append(cur)
            cur = Segment(text=text, start=ts, end=ts, avg_conf=conf, count=1, bboxes=[box])
    if cur:
        segs.append(cur)
    return segs

# -------------------- Word Grouping -------------------- #

def group_words(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return a dict: word -> {timestamps: [...], occurrences: [...], count: int}
    occurrences: each is {ts, bbox, conf, raw_text_row(optional)}
    """
    store: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"timestamps": [], "occurrences": [], "count": 0})
    for _, row in df.iterrows():
        ts = float(row["frame_ts"])
        conf = float(row["conf"])
        bbox = tuple(row["bbox"])
        # split already-normalized text
        for token in row["text"].split():
            entry = store[token]
            entry["timestamps"].append(ts)
            entry["occurrences"].append({"ts": ts, "bbox": bbox, "conf": conf})
            entry["count"] += 1
    # sort timestamps & occurrences by ts
    for w, d in store.items():
        order = sorted(range(len(d["timestamps"])), key=lambda i: d["timestamps"][i])
        d["timestamps"] = [d["timestamps"][i] for i in order]
        d["occurrences"] = [d["occurrences"][i] for i in order]
    return store

# -------------------- Main -------------------- #

def main(csv_in: str, out_words: str, segments_out: Optional[str], dict_path: Optional[str]):
    df = pd.read_csv(csv_in)
    df["text"] = df["extra"].apply(lambda s: json.loads(s)["text"])
    df = df[df["conf"] >= CONF_THR].copy()

    sym = load_symspell(dict_path) if dict_path else None
    df["text"] = df["text"].map(lambda t: normalize(t, sym))
    df = df[df["text"].str.len() > 1]
    df["bbox"] = df[["x1", "y1", "x2", "y2"]].apply(tuple, axis=1)
    df = df.sort_values("frame_ts")

    # 1. WORD-CENTRIC OUTPUT
    word_dict = group_words(df)
    with open(out_words, "w", encoding="utf-8") as f:
        for word, payload in word_dict.items():
            rec = {
                "type": "word",
                "word": word,
                **payload
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "")

    # 2. OPTIONAL SEGMENTS FILE
    if segments_out:
        segments = merge_segments(df.to_dict("records"))
        with open(segments_out, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg.to_record(), ensure_ascii=False) + "")
        print(f"Saved {len(word_dict)} words to {out_words} and {len(segments)} segments to {segments_out}")
    else:
        print(f"Saved {len(word_dict)} words to {out_words}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Group OCR words and (optionally) export merged segments.")
    p.add_argument("csv_in")
    p.add_argument("--out", dest="out_words", default="word_index.jsonl", help="JSONL file with one word per line")
    p.add_argument("--segments-out", help="Optional JSONL for merged segments")
    p.add_argument("--dict", dest="dict_path", help="SymSpell frequency dictionary path")
    args = p.parse_args()

    main(args.csv_in, args.out_words, args.segments_out, args.dict_path)
