#!/usr/bin/env python3
"""
transcript_hybrid_rag.py  (FR-friendly + anchored windows + RERANK + CONTEXT WINDOWS)
====================================================================================

Adds parameterizable context windows around each retrieved chunk:
  --ctx-before N   include N previous chunks (default 3)
  --ctx-after M    include M following chunks (default 3)
  --show-context   pretty print the window for each hit in CLI
  --json path.json dump full JSON results (hits + contexts)

Build (same as before)
----------------------
python transcript_hybrid_rag.py build transcript.srt \
  --window 5 --anchor first \
  --model sentence-transformers/all-MiniLM-L6-v2

Query with rerank + context
---------------------------
python transcript_hybrid_rag.py query /path/to/rag_index_* \
  --q "hors-jeu en panne ?" --k 8 --method rrf \
  --rerank cross --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 --overfetch 50 \
  --ctx-before 3 --ctx-after 3 --show-context --json results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import pickle
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- BM25 ---
from rank_bm25 import BM25Okapi
# --- Embeddings ---
from sentence_transformers import SentenceTransformer
# --- Cross-encoder (optional) ---
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


# ------------------------------
# Time helpers
# ------------------------------
_SRT_TIME_RE = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)")

def srt_time_to_seconds(s: str) -> float:
    m = _SRT_TIME_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"Bad SRT time: {s}")
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0

def seconds_to_srt_time(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    hh, rem = divmod(total_ms, 3600000)
    mm, rem = divmod(rem, 60000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# ------------------------------
# SRT Parsing
# ------------------------------
@dataclass
class SRTEntry:
    idx: int
    start: float
    end: float
    text: str

def parse_srt(srt_path: Path) -> List[SRTEntry]:
    content = srt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    entries: List[SRTEntry] = []
    for block in blocks:
        lines = [l.strip("\ufeff") for l in block.splitlines() if l.strip() != ""]
        if not lines:
            continue
        try:
            idx = int(lines[0].strip())
            time_line = lines[1]
            text_lines = lines[2:]
        except ValueError:
            idx = len(entries) + 1
            time_line = lines[0]
            text_lines = lines[1:]
        if "-->" not in time_line:
            if len(lines) >= 2 and "-->" in lines[1]:
                time_line = lines[1]; text_lines = lines[2:]
            else:
                continue
        start_s, end_s = [t.strip() for t in time_line.split("-->")]
        start = srt_time_to_seconds(start_s)
        end = srt_time_to_seconds(end_s)
        text = " ".join(t.strip() for t in text_lines).strip()
        if text:
            entries.append(SRTEntry(idx=idx, start=start, end=end, text=text))
    return entries


# ------------------------------
# Windowing
# ------------------------------
@dataclass
class Chunk:
    chunk_id: str
    start: float
    end: float
    text: str

def window_chunks(entries: List[SRTEntry], window_sec: int = 5, anchor: str = "first") -> List[Chunk]:
    if not entries:
        return []
    max_end = max(e.end for e in entries)
    t = entries[0].start if anchor == "first" else 0.0
    chunks: List[Chunk] = []
    win_idx = 0
    while t < max_end + 1e-6:
        t0, t1 = t, min(t + window_sec, max_end)
        texts = [e.text for e in entries if e.end > t0 and e.start < t1]
        if texts:
            chunk_id = f"win_{win_idx:06d}_{int(round(t0*1000)):010d}"
            chunks.append(Chunk(chunk_id=chunk_id, start=t0, end=t1, text=" ".join(texts).strip()))
        t += window_sec
        win_idx += 1
    return chunks


# ------------------------------
# Tokenization for BM25 (FR-aware)
# ------------------------------
def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _fr_stopwords() -> set:
    words = """
    le la les de des du un une et à a au aux que qui en dans pour sur par pas plus ne est c est ce cette ces ça
    donc mais ou où alors avec sans se sa son ses leur leurs y là être avoir faire aller pouvoir devoir comme
    on il ils elle elles je tu nous vous te me se moi toi lui leur eux ce cet cette ça cela ceci
    d l j t qu n s m y
    euh bah hein voilà genre ben ouais ouai oui non
    """
    return set(w for w in words.split() if w)

def simple_tokenize_fr(s: str, remove_stopwords: bool = True) -> List[str]:
    s = s.lower()
    s = _strip_accents(s)
    s = s.replace("’", "'").replace("-", " ").replace("'", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split() if s else []
    if remove_stopwords:
        stops = _fr_stopwords()
        toks = [t for t in toks if t not in stops and len(t) > 1]
    return toks

def build_bm25(corpus_tokens: List[List[str]]) -> BM25Okapi:
    return BM25Okapi(corpus_tokens)


# ------------------------------
# Embeddings
# ------------------------------
class EmbeddingBackend:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = True) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vecs.astype(np.float32, copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text], show_progress_bar=False)[0]


# ------------------------------
# Rerankers
# ------------------------------
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None):
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder not available. Install sentence-transformers.")
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: List[str]) -> np.ndarray:
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return scores.astype(np.float32, copy=False)

def mmr_rerank(query_vec: np.ndarray, doc_vecs: np.ndarray, candidates: List[int], k: int, lambda_mult: float = 0.7) -> List[int]:
    if len(candidates) <= k:
        return candidates
    selected: List[int] = []
    cand_set = set(candidates)
    sims_q = (doc_vecs[candidates] @ query_vec).astype(np.float32)
    while len(selected) < k and cand_set:
        best_i, best_score = None, -1e9
        for pos, idx in enumerate(candidates):
            if idx not in cand_set:
                continue
            div = 0.0
            if selected:
                sims_to_sel = doc_vecs[idx] @ doc_vecs[selected].T
                if np.ndim(sims_to_sel) == 0:
                    sims_to_sel = np.array([float(sims_to_sel)])
                div = float(np.max(sims_to_sel))
            score = lambda_mult * float(sims_q[pos]) - (1.0 - lambda_mult) * div
            if score > best_score:
                best_score, best_i = score, idx
        selected.append(best_i)
        cand_set.remove(best_i)
    return selected


# ------------------------------
# Index object (build / load / query + rerank + context)
# ------------------------------
@dataclass
class Meta:
    model_name: str
    window_sec: int
    built_at: str
    num_chunks: int
    srt_source: str
    anchor: str

class TranscriptRAGIndex:
    def __init__(self, root: Path):
        self.root = root
        self.meta: Optional[Meta] = None
        self.df: Optional[pd.DataFrame] = None
        self.corpus_tokens: Optional[List[List[str]]] = None
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None

    # ---------- Build ----------
    @classmethod
    def build_from_srt(cls, srt_path: Path, outdir: Optional[Path] = None, window_sec: int = 5,
                       model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None,
                       anchor: str = "first") -> "TranscriptRAGIndex":
        srt_path = Path(srt_path).resolve()
        if not srt_path.exists():
            raise FileNotFoundError(srt_path)
        if outdir is None or str(outdir).lower() == "auto":
            outdir = srt_path.parent / f"rag_index_{now_stamp()}"
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"• Parsing SRT: {srt_path}")
        entries = parse_srt(srt_path)

        print(f"• Windowing into {window_sec}s chunks (anchor={anchor})…")
        chunks = window_chunks(entries, window_sec=window_sec, anchor=anchor)
        if not chunks:
            raise RuntimeError("No chunks created from SRT.")

        df = pd.DataFrame([{
            "chunk_id": c.chunk_id,
            "start": float(c.start),
            "end": float(c.end),
            "start_srt": seconds_to_srt_time(c.start),
            "end_srt": seconds_to_srt_time(c.end),
            "text": c.text,
        } for c in chunks]).sort_values("start")
        df.to_json(outdir / "chunks.jsonl", orient="records", lines=True, force_ascii=False)

        print("• Building BM25 index (FR-aware)…")
        corpus_tokens = [simple_tokenize_fr(t) for t in df["text"].tolist()]
        bm25 = build_bm25(corpus_tokens)
        (outdir / "tokens.pkl").write_bytes(pickle.dumps(corpus_tokens))
        (outdir / "bm25.pkl").write_bytes(pickle.dumps(bm25))

        print(f"• Encoding embeddings with: {model_name}")
        be = EmbeddingBackend(model_name=model_name, device=device)
        embs = be.encode(df["text"].tolist())
        np.save(outdir / "embeddings.npy", embs)

        ids = df["chunk_id"].tolist()
        (outdir / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = Meta(model_name=model_name, window_sec=window_sec, built_at=now_stamp(),
                    num_chunks=len(df), srt_source=str(srt_path), anchor=anchor)
        (outdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

        obj = cls(outdir)
        obj.meta, obj.df = meta, df.reset_index(drop=True)
        obj.corpus_tokens, obj.bm25, obj.embeddings = corpus_tokens, bm25, embs
        return obj

    # ---------- Load ----------
    @classmethod
    def load(cls, root: Path) -> "TranscriptRAGIndex":
        root = Path(root).resolve()
        obj = cls(root)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        obj.meta = Meta(**meta)
        obj.df = pd.read_json(root / "chunks.jsonl", lines=True, dtype=False).reset_index(drop=True)
        obj.corpus_tokens = pickle.loads((root / "tokens.pkl").read_bytes())
        obj.bm25 = pickle.loads((root / "bm25.pkl").read_bytes())
        obj.embeddings = np.load(root / "embeddings.npy")
        return obj

    # ---------- Base retrieval ----------
    def query_bm25(self, q: str, k: int = 10) -> List[Dict]:
        q_tok = simple_tokenize_fr(q)
        scores = self.bm25.get_scores(q_tok)
        idxs = np.argsort(scores)[::-1][:k]
        return [self._mk_hit(int(i), float(scores[i]), method="bm25", rank=r)
                for r, i in enumerate(idxs, start=1)]

    def query_embed(self, q: str, k: int = 10, model_name: Optional[str] = None, device: Optional[str] = None) -> List[Dict]:
        model_name = model_name or (self.meta.model_name if self.meta else "sentence-transformers/all-MiniLM-L6-v2")
        be = EmbeddingBackend(model_name=model_name, device=device)
        qv = be.encode_one(q)
        sims = (self.embeddings @ qv)
        idxs = np.argsort(sims)[::-1][:k]
        return [self._mk_hit(int(i), float(sims[i]), method="embed", rank=r)
                for r, i in enumerate(idxs, start=1)]

    def query_hybrid(self, q: str, k: int = 10, method: str = "rrf", alpha: float = 0.5,
                     overfetch: int = 50, model_name: Optional[str] = None, device: Optional[str] = None) -> List[Dict]:
        bm25_hits = self.query_bm25(q, k=max(k, overfetch))
        bm25_ranks = {h["index"]: r for r, h in enumerate(bm25_hits, start=1)}
        bm25_scores = {h["index"]: h["score"] for h in bm25_hits}

        embed_hits = self.query_embed(q, k=max(k, overfetch), model_name=model_name, device=device)
        embed_ranks = {h["index"]: r for r, h in enumerate(embed_hits, start=1)}
        embed_scores = {h["index"]: h["score"] for h in embed_hits}

        cand = set(bm25_ranks) | set(embed_ranks)
        fused: List[Tuple[int, float]] = []

        if method.lower() == "rrf":
            K = 60.0
            for i in cand:
                r1 = bm25_ranks.get(i, 10_000); r2 = embed_ranks.get(i, 10_000)
                score = 1.0 / (K + r1) + 1.0 / (K + r2)
                fused.append((i, score))
        else:
            def zscores(d: Dict[int, float]) -> Dict[int, float]:
                arr = np.array([d.get(i, 0.0) for i in cand], dtype=np.float64)
                mu, sd = float(arr.mean()), float(arr.std() + 1e-9)
                return {i: (d.get(i, mu) - mu) / sd for i in cand}
            bzn = zscores(bm25_scores); ezn = zscores(embed_scores)
            for i in cand:
                fused.append((i, alpha * ezn[i] + (1 - alpha) * bzn[i]))

        fused.sort(key=lambda x: x[1], reverse=True)
        idxs = [i for i, _ in fused[:max(k, overfetch)]]
        hits = [self._mk_hit(i, score=0.0, method=f"hybrid-{method}", rank=rank+1) for rank, i in enumerate(idxs)]
        for h, (_, s) in zip(hits, fused[:len(hits)]):
            h["hretr_score"] = float(s)
        return hits

    # ---------- Reranking ----------
    def rerank_cross(self, q: str, hits: List[Dict], model_name: str, device: Optional[str], top_k: int) -> List[Dict]:
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder not available. Install sentence-transformers.")
        reranker = CrossEncoderReranker(model_name=model_name, device=device)
        texts = [h["text"] for h in hits]
        scores = reranker.score(q, texts)
        order = np.argsort(scores)[::-1][:top_k]
        out = []
        for rank, j in enumerate(order, start=1):
            h = dict(hits[int(j)])
            h["rerank_score"] = float(scores[int(j)])
            h["method"] += "+cross"
            h["rank"] = rank
            out.append(h)
        return out

    def rerank_mmr(self, q: str, hits: List[Dict], top_k: int, lambda_mult: float = 0.7,
                   model_name: Optional[str] = None, device: Optional[str] = None) -> List[Dict]:
        model_name = model_name or (self.meta.model_name if self.meta else "sentence-transformers/all-MiniLM-L6-v2")
        be = EmbeddingBackend(model_name=model_name, device=device)
        qv = be.encode_one(q)
        idxs = [h["index"] for h in hits]
        selected = mmr_rerank(qv, self.embeddings, idxs, k=top_k, lambda_mult=lambda_mult)
        order = {idx: r for r, idx in enumerate(selected, start=1)}
        picked = [h for h in hits if h["index"] in order]
        picked.sort(key=lambda h: order[h["index"]])
        for h in picked:
            h["method"] += "+mmr"
            h["rerank_score"] = None
            h["rank"] = order[h["index"]]
        return picked

    # ---------- Context windows ----------
    def context_window_for_index(self, center_idx: int, before: int, after: int) -> List[Dict]:
        lo = max(0, center_idx - before)
        hi = min(len(self.df) - 1, center_idx + after)
        items = []
        for j in range(lo, hi + 1):
            r = self.df.iloc[j]
            items.append({
                "offset": j - center_idx,          # negative: previous, 0: center, positive: following
                "index": int(j),
                "chunk_id": r["chunk_id"],
                "start": float(r["start"]),
                "end": float(r["end"]),
                "start_srt": r["start_srt"],
                "end_srt": r["end_srt"],
                "text": r["text"],
            })
        return items

    def attach_context(self, hits: List[Dict], before: int, after: int) -> List[Dict]:
        enriched = []
        for h in hits:
            ctx = self.context_window_for_index(h["index"], before, after)
            h2 = dict(h)
            h2["context"] = ctx  # list of chunk dicts with offsets
            enriched.append(h2)
        return enriched

    # ---------- Helpers ----------
    def _mk_hit(self, row_idx: int, score: float, method: str, rank: int) -> Dict:
        r = self.df.iloc[row_idx]
        return {
            "rank": rank,
            "method": method,
            "index": row_idx,
            "chunk_id": r["chunk_id"],
            "start": float(r["start"]),
            "end": float(r["end"]),
            "start_srt": r["start_srt"],
            "end_srt": r["end_srt"],
            "score": float(score),
            "text": r["text"],
        }


# ------------------------------
# CLI
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="Build and query a hybrid RAG index over SRT transcript (with reranking & context windows).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build index from transcript.srt")
    pb.add_argument("srt", help="Path to transcript.srt")
    pb.add_argument("--window", type=int, default=5, help="Chunk window size in seconds (default: 5)")
    pb.add_argument("--outdir", default="auto", help="Output folder (default: auto under transcript dir)")
    pb.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model")
    pb.add_argument("--device", default=None, help="Force device ('cpu' or 'cuda')")
    pb.add_argument("--anchor", choices=["first","zero"], default="first", help="Anchor first window at 'first' subtitle start or 0.0")

    pq = sub.add_parser("query", help="Query an existing index (with optional reranking and context windows)")
    pq.add_argument("index", help="Path to built index folder (rag_index_*)")
    pq.add_argument("--q", required=True, help="Query text")
    pq.add_argument("--k", type=int, default=8, help="Top-K final results to return")
    pq.add_argument("--method", choices=["rrf", "weighted", "bm25", "embed"], default="rrf", help="Base retrieval")
    pq.add_argument("--alpha", type=float, default=0.5, help="Weight for 'weighted' hybrid (embed weight)")
    pq.add_argument("--model", default=None, help="Override embedding model at query time")
    pq.add_argument("--device", default=None, help="Override device at query time")

    # Rerank options
    pq.add_argument("--rerank", choices=["none","cross","mmr"], default="none", help="Reranking strategy")
    pq.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model")
    pq.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR lambda (relevance vs diversity)")
    pq.add_argument("--overfetch", type=int, default=50, help="Candidates to fetch before reranking")

    # Context options
    pq.add_argument("--ctx-before", type=int, default=3, help="Number of previous chunks to include per hit")
    pq.add_argument("--ctx-after", type=int, default=3, help="Number of following chunks to include per hit")
    pq.add_argument("--show-context", action="store_true", help="Pretty print context windows in CLI")
    pq.add_argument("--json", default=None, help="Write full results (hits + contexts) as JSON to this path")

    args = p.parse_args()

    if args.cmd == "build":
        out = TranscriptRAGIndex.build_from_srt(
            srt_path=Path(args.srt),
            outdir=(None if args.outdir == "auto" else Path(args.outdir)),
            window_sec=int(args.window),
            model_name=args.model,
            device=args.device,
            anchor=args.anchor,
        )
        print("\n✅ Index built.")
        print(f"   Path: {out.root}")
        print(f"   Chunks: {out.meta.num_chunks}")
        print(f"   Anchor: {out.meta.anchor}")
        return

    # Query path
    idx = TranscriptRAGIndex.load(Path(args.index))

    # Base retrieval (overfetch pool)
    if args.method in ("bm25", "embed"):
        base_hits = (idx.query_bm25(args.q, k=max(args.k, args.overfetch))
                     if args.method == "bm25"
                     else idx.query_embed(args.q, k=max(args.k, args.overfetch), model_name=args.model, device=args.device))
    else:
        base_hits = idx.query_hybrid(args.q, k=args.k, method=args.method, alpha=args.alpha,
                                     overfetch=args.overfetch, model_name=args.model, device=args.device)

    # Optional rerank
    if args.rerank == "cross":
        final_hits = idx.rerank_cross(args.q, base_hits, model_name=args.rerank_model,
                                      device=args.device, top_k=args.k)
    elif args.rerank == "mmr":
        final_hits = idx.rerank_mmr(args.q, base_hits, top_k=args.k, lambda_mult=args.mmr_lambda,
                                    model_name=args.model, device=args.device)
    else:
        final_hits = base_hits[:args.k]

    # Attach context windows
    enriched = idx.attach_context(final_hits, before=args.ctx_before, after=args.ctx_after)

    # Pretty print
    print()
    for h in enriched:
        label_parts = [h["method"]]
        if "hretr_score" in h:
            label_parts.append(f"hyb={h['hretr_score']:.4f}")
        if "rerank_score" in h and h["rerank_score"] is not None:
            label_parts.append(f"rerank={h['rerank_score']:.4f}")
        label = ", ".join(label_parts)

        print(f"[{h['rank']:02}] {h['start_srt']} → {h['end_srt']}  ({label})")
        preview = h["text"]
        if len(preview) > 220:
            preview = preview[:217] + "…"
        print("     " + preview)

        if args.show_context:
            for c in h["context"]:
                flag = "●" if c["offset"] == 0 else "·"
                clean_text = c['text'][:120].replace("\n", " ")
                print(f"{flag} [{c['start_srt']}–{c['end_srt']}] (offset {c['offset']:>+2}) {clean_text}" + ('…' if len(c['text'])>120 else ''))

        print()

    # Optional JSON dump
    if args.json:
        Path(args.json).write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📝 Saved JSON → {args.json}")

if __name__ == "__main__":
    main()
