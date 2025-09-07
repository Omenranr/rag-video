"""
video_rag_ingest_v2.py
=======================
Unify SRT (audio) + detection CSV (vision/OCR) into a clean, queryable store.
This version handles the two detection shapes you described:

1. **"screen" videos (OCR):** `label == 'text'` and the recognized string is in `extra.text`.
2. **"nature"/"sport"/"cctv" videos (objects):** regular object classes in `label`, `extra` is usually empty.

It creates:
- `visual_event` table: ALL detections (OCR + objects) with unified columns.
- `ocr_event` table: Only OCR rows, making FTS on OCR text easy.
- `utterance` table: Transcript blocks from SRT.

Persistence layers:
- Parquet files (canonical storage)
- DuckDB views over Parquet (fast SQL analytics)
- SQLite FTS5 indices for transcript **and** OCR text

CLI
----
python video_rag_ingest_v2.py \
  --video-id demo1 \
  --srt results/demo1.srt \
  --detections results/demo1_detections.csv \
  --out-dir data \
  --duckdb data/store.duckdb \
  --sqlite data/fts.sqlite

Optional:
  --det-kind screen|nature|sport|cctv   (only used for logging / sanity checks)

"""
from __future__ import annotations
import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------
SRT_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")

def _ts_to_seconds(ts: str) -> float:
    m = SRT_TS.match(ts.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {ts}")
    h, m_, s, ms = map(int, m.groups())
    return h * 3600 + m_ * 60 + s + ms / 1000.0

def parse_srt(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        block: List[str] = []
        for line in fh:
            if line.strip():
                block.append(line.rstrip("\n"))
            else:
                if block:
                    _flush_block(block, rows)
                    block = []
        if block:
            _flush_block(block, rows)
    return pd.DataFrame(rows, columns=["utt_id", "start", "end", "speaker", "text"])

def _flush_block(lines: List[str], rows: List[dict]):
    if len(lines) < 2:
        return
    # 1st line may be index
    try:
        utt_id = int(lines[0].strip())
        ts_line = lines[1]
        text_lines = lines[2:]
    except ValueError:
        utt_id = len(rows) + 1
        ts_line = lines[0]
        text_lines = lines[1:]
    start_s, end_s = [t.strip() for t in ts_line.split("-->")]
    start, end = _ts_to_seconds(start_s), _ts_to_seconds(end_s)
    speaker = "UNK"
    if text_lines:
        m = re.match(r"^\[(.*?)\]\s*(.*)$", text_lines[0])
        if m:
            speaker = m.group(1)
            text_lines[0] = m.group(2)
    text = " ".join(t.strip() for t in text_lines).strip()
    rows.append({"utt_id": utt_id, "start": start, "end": end, "speaker": speaker, "text": text})

# ---------------------------------------------------------------------------
# Detection CSV loader & normalization
# ---------------------------------------------------------------------------

def load_detections(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"frame_ts", "label", "conf", "x1", "y1", "x2", "y2", "extra"}
    miss = expected - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in detections CSV: {miss}")

    # Parse extra JSON safely
    def _parse_extra(val: str):
        if pd.isna(val) or val == "":
            return {}
        try:
            return json.loads(val)
        except Exception:
            return {}

    extra = df["extra"].apply(_parse_extra)
    extra_df = pd.json_normalize(extra).add_prefix("extra.")

    out = pd.concat([df.drop(columns=["extra"]).rename(columns={"frame_ts": "ts"}), extra_df], axis=1)

    # Normalize OCR vs object rows
    # If label=='text' and extra.text exists => OCR row
    out["ocr_text"] = out.get("extra.text")
    out["is_ocr"] = (out["label"].str.lower() == "text") & out["ocr_text"].notna()

    # Insert surrogate key
    out.insert(0, "event_id", range(1, len(out) + 1))
    return out

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def init_duckdb(db_path: Path, parquet_map: dict[str, Path]):
    con = duckdb.connect(str(db_path))
    for table, pq in parquet_map.items():
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{pq.as_posix()}');")
    con.close()

def init_sqlite_fts(sqlite_path: Path, utter_df: pd.DataFrame, ocr_df: pd.DataFrame | None = None):
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")

    # Drop old
    cur.executescript(
        """
        DROP TABLE IF EXISTS utterance;
        DROP TABLE IF EXISTS utterance_fts;
        DROP TABLE IF EXISTS ocr_event;
        DROP TABLE IF EXISTS ocr_event_fts;
        """
    )

    # Plain utterance table
    cur.execute(
        """
        CREATE TABLE utterance (
            utt_id INTEGER PRIMARY KEY,
            video_id TEXT,
            start REAL,
            end REAL,
            speaker TEXT,
            text TEXT
        );
        """
    )
    utter_df.to_sql("utterance", con, if_exists="append", index=False)

    cur.execute(
        "CREATE VIRTUAL TABLE utterance_fts USING fts5(text, content='utterance', content_rowid='utt_id');"
    )
    cur.execute("INSERT INTO utterance_fts(rowid, text) SELECT utt_id, text FROM utterance;")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_utter_start ON utterance(start);")

    # OCR FTS (optional)
    if ocr_df is not None and len(ocr_df):
        cur.execute(
            """
            CREATE TABLE ocr_event (
                event_id INTEGER PRIMARY KEY,
                video_id TEXT,
                ts REAL,
                ocr_text TEXT
            );
            """
        )
        ocr_df[["event_id", "video_id", "ts", "ocr_text"]].to_sql("ocr_event", con, if_exists="append", index=False)
        cur.execute(
            "CREATE VIRTUAL TABLE ocr_event_fts USING fts5(ocr_text, content='ocr_event', content_rowid='event_id');"
        )
        cur.execute("INSERT INTO ocr_event_fts(rowid, ocr_text) SELECT event_id, ocr_text FROM ocr_event;")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_ts ON ocr_event(ts);")

    con.commit()
    con.close()

# ---------------------------------------------------------------------------
# Args & main
# ---------------------------------------------------------------------------
@dataclass
class Args:
    video_id: str
    srt: Path
    detections: Path
    out_dir: Path
    duckdb: Optional[Path]
    sqlite: Optional[Path]
    det_kind: Optional[str]

def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Unify SRT + detections (OCR/objects) into Parquet, DuckDB & SQLite FTS")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--srt", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--duckdb", type=Path, default=None)
    ap.add_argument("--sqlite", type=Path, default=None)
    ap.add_argument("--det-kind", choices=["screen", "nature", "sport", "cctv"], default=None)
    ns = ap.parse_args()
    return Args(ns.video_id, ns.srt, ns.detections, ns.out_dir, ns.duckdb, ns.sqlite, ns.det_kind)

def main():
    args = parse_args()

    # 1. Audio transcript
    utter_df = parse_srt(args.srt)
    utter_df.insert(0, "video_id", args.video_id)

    # 2. Visual detections
    det_df = load_detections(args.detections)
    det_df.insert(1, "video_id", args.video_id)

    # 3. Split OCR subset for convenience
    ocr_df = det_df[det_df["is_ocr"]].copy()

    # 4. Write Parquet
    pqdir = args.out_dir / "parquet"
    write_parquet(pd.DataFrame([[args.video_id]], columns=["video_id"]), pqdir / "video.parquet")
    write_parquet(utter_df, pqdir / "utterance.parquet")
    write_parquet(det_df, pqdir / "visual_event.parquet")
    if len(ocr_df):
        write_parquet(ocr_df[["event_id", "video_id", "ts", "ocr_text"]], pqdir / "ocr_event.parquet")

    # 5. DBs
    if args.duckdb:
        parquet_map = {
            "video": pqdir / "video.parquet",
            "utterance": pqdir / "utterance.parquet",
            "visual_event": pqdir / "visual_event.parquet",
        }
        if len(ocr_df):
            parquet_map["ocr_event"] = pqdir / "ocr_event.parquet"
        init_duckdb(args.duckdb, parquet_map)
        print(f"DuckDB views created at {args.duckdb}")

    if args.sqlite:
        init_sqlite_fts(args.sqlite, utter_df, ocr_df if len(ocr_df) else None)
        print(f"SQLite FTS indices built at {args.sqlite}")

    kind_msg = f" ({args.det_kind})" if args.det_kind else ""
    print(f"✅ Done for video_id='{args.video_id}'{kind_msg}")

if __name__ == "__main__":
    main()
