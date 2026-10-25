"""Offline before/after comparison of retrieval strategies.

Compares:
  - Vector-only (baseline): cosine similarity over Gemini embeddings
  - Hybrid (after): vector + keyword (lexical) fused with RRF

Metrics: hit@K, MRR, recall@K, precision@K

Usage:
    python evals/retrieval_compare.py
    python evals/retrieval_compare.py --k 8 --topks 1,3,5,8,10 --rrf-k 30,60,100
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "server" / ".env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from eval_utils import atomic_json_write, content_hash, validate_output_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "when", "where", "who", "which", "why", "how",
    "do", "does", "did", "this", "these", "those", "i", "you", "your", "we",
    "they", "them", "their", "can", "could", "would", "should", "may", "might",
}


def tokenize(text):
    """Simple word tokenizer: lowercase, split on non-alphanumeric."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else 0.0


def embed_texts_with_retry(embedder, texts, task_type="retrieval_document", max_retries=5):
    """Embed with exponential backoff on 429 rate-limit errors."""
    import time
    for attempt in range(max_retries):
        try:
            return np.asarray(embedder.embed_documents(texts), dtype=np.float32)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                wait = min(2 ** attempt, 60)
                print(f"  [rate limited, retrying in {wait}s...]")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed to embed after {max_retries} retries")


def embed_query_with_retry(embedder, text, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return np.asarray(embedder.embed_query(text), dtype=np.float32)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                wait = min(2 ** attempt, 60)
                print(f"  [rate limited, retrying in {wait}s...]")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed to embed query after {max_retries} retries")


def cosine_topk(query_vec, doc_matrix, k):
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    dn = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)
    sims = dn @ qn
    order = np.argsort(-sims)[:k]
    return order.tolist(), sims[order].tolist()


def keyword_topk(question, chunk_texts, k):
    """Simulate PostgreSQL plainto_tsquery + ts_rank.

    Approximation of full-text search:
      - Tokenize question into query terms
      - For each chunk, score by TF-IDF-like weight of query terms
      - Only chunks containing >= 1 query term are eligible (mimics @@ operator)
      - Rank by score descending
    """
    q_terms = tokenize(question)
    if not q_terms:
        return list(range(len(chunk_texts)))[:k], [0.0] * min(k, len(chunk_texts))

    q_set = set(q_terms)
    # Document frequency for IDF: in how many chunks does each term appear?
    doc_freq = Counter()
    all_tokens = [tokenize(t) for t in chunk_texts]
    for tokens in all_tokens:
        for t in set(tokens):
            if t in q_set:
                doc_freq[t] += 1

    n_chunks = len(chunk_texts)
    scores = []
    for tokens in all_tokens:
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        total = len(tokens)
        # BM25-like: sum of tf * idf for query terms present
        score = 0.0
        has_match = False
        for t in q_set:
            if tf.get(t, 0) > 0:
                has_match = True
                idf = math.log((n_chunks - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5) + 1.0)
                tf_norm = tf[t] / total
                score += tf_norm * idf
        # Mimic @@ : chunks with zero matching terms get score 0 (not ranked)
        if not has_match:
            score = 0.0
        scores.append(score)

    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return order, [scores[i] for i in order]


def fuse_rrf(vector_order, keyword_order, doc_ids, k=60, limit=None):
    """Reciprocal Rank Fusion.

    vector_order / keyword_order are lists of indices into doc_ids.
    Returns list of indices fused and re-ranked.
    """
    scores = {}
    for rank, idx in enumerate(vector_order, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(keyword_order, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.keys(), key=lambda i: -scores[i])
    if limit:
        fused = fused[:limit]
    return fused


def evaluate(retrieved_ids, gold_ids, k):
    topk = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hit = int(bool(topk & gold_set)) if gold_set else 0
    recall = (len(topk & gold_set) / len(gold_set)) if gold_set else 0.0
    precision = (len(topk & gold_set) / k) if gold_set else 0.0
    # MRR: rank of first gold chunk in full retrieved list
    first_rank = None
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_set:
            first_rank = rank
            break
    mrr = (1.0 / first_rank) if first_rank else 0.0
    return hit, recall, precision, mrr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Compare vector-only vs hybrid retrieval")
    parser.add_argument("--data", default=str(Path(__file__).parent / "rag_eval.json"))
    parser.add_argument("--k", type=int, default=8, help="primary top-K")
    parser.add_argument("--topks", default="1,3,5,8,10")
    parser.add_argument(
        "--rrf-k", default="30,60,100", help="comma-separated RRF k values to test"
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
    )
    parser.add_argument("--cache", default=str(Path(__file__).parent / ".compare_cache.json"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).parent / "retrieval_compare_results.json"))
    args = parser.parse_args()
    validate_output_path(args.output)

    topks = sorted({int(x) for x in args.topks.split(",") if x.strip()})
    if args.k not in topks:
        topks = sorted(set(topks) | {args.k})
    rrf_ks = [int(x) for x in args.rrf_k.split(",") if x.strip()]

    data_path = Path(args.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    questions = data["questions"]

    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]

    print(f"Chunks: {len(chunks)}  Questions: {len(questions)}")
    print(f"Embedding model: {args.embedding_model}")

    embedder = GoogleGenerativeAIEmbeddings(
        model=args.embedding_model, google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # ---- cache / embed chunks ----
    cache_path = Path(args.cache)
    cache = {}
    if not args.no_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    cache.setdefault("model", args.embedding_model)
    cache.setdefault("chunks", {})
    cache.setdefault("questions", {})

    if cache.get("model") != args.embedding_model:
        cache = {"model": args.embedding_model, "chunks": {}, "questions": {}}

    chunk_hashes = [content_hash(text) for text in chunk_texts]
    text_by_hash = dict(zip(chunk_hashes, chunk_texts))
    missing = list(dict.fromkeys(key for key in chunk_hashes if key not in cache["chunks"]))
    if missing:
        missing_texts = [text_by_hash[key] for key in missing]
        print(f"Embedding {len(missing_texts)} chunks (batched)...")
        vecs = embed_texts_with_retry(embedder, missing_texts)
        for key, v in zip(missing, vecs):
            cache["chunks"][key] = v.tolist()
        if not args.no_cache:
            atomic_json_write(cache_path, cache)
    else:
        print("Chunk embeddings cached.")

    chunk_matrix = np.asarray([cache["chunks"][key] for key in chunk_hashes], dtype=np.float32)

    # ---- embed questions ----
    q_texts = [q["question"] for q in questions]
    q_hashes = [content_hash(text) for text in q_texts]
    question_by_hash = dict(zip(q_hashes, q_texts))
    missing_q = list(dict.fromkeys(key for key in q_hashes if key not in cache["questions"]))
    print(f"Embedding {len(missing_q)} uncached questions (batched)...")
    if missing_q:
        q_vecs_missing = embed_texts_with_retry(embedder, [question_by_hash[key] for key in missing_q], task_type="retrieval_query")
        for key, vector in zip(missing_q, q_vecs_missing):
            cache["questions"][key] = vector.tolist()
        if not args.no_cache:
            atomic_json_write(cache_path, cache)
    q_vecs = np.asarray([cache["questions"][key] for key in q_hashes], dtype=np.float32)

    # ---- run strategies ----
    results = {}

    # Strategy 1: Vector-only (baseline)
    print("\n=== Strategy: Vector-only (baseline) ===")
    vec_results = []
    for qi, q in enumerate(questions):
        gold = set(q["all_expected_chunk_ids"])
        order, sims = cosine_topk(q_vecs[qi], chunk_matrix, max(topks))
        retrieved = [chunk_ids[i] for i in order]
        row = {"id": q["id"], "type": q["type"], "difficulty": q["difficulty"],
               "retrieved": retrieved, "gold": sorted(gold)}
        for k in topks:
            h, r, p, m = evaluate(retrieved, gold, k)
            row[f"hit@{k}"] = h
            row[f"recall@{k}"] = r
            row[f"precision@{k}"] = p
            row[f"mrr@{k}"] = m
        vec_results.append(row)
    results["vector_only"] = vec_results

    # Strategy 2: Hybrid for each RRF k
    for rrf_k in rrf_ks:
        print(f"\n=== Strategy: Hybrid (RRF k={rrf_k}) ===")
        hyb_results = []
        for qi, q in enumerate(questions):
            gold = set(q["all_expected_chunk_ids"])
            q_text = q["question"]

            # Vector: top-3*K (wider pool to preserve vector-only results)
            vec_order, _ = cosine_topk(q_vecs[qi], chunk_matrix, max(topks) * 3)
            # Keyword: top-3*K
            kw_order, _ = keyword_topk(q_text, chunk_texts, max(topks) * 3)

            # RRF fuse
            fused = fuse_rrf(vec_order, kw_order, chunk_ids, k=rrf_k, limit=max(topks))
            retrieved = [chunk_ids[i] for i in fused]

            row = {"id": q["id"], "type": q["type"], "difficulty": q["difficulty"],
                   "retrieved": retrieved, "gold": sorted(gold), "rrf_k": rrf_k}
            for k in topks:
                h, r, p, m = evaluate(retrieved, gold, k)
                row[f"hit@{k}"] = h
                row[f"recall@{k}"] = r
                row[f"precision@{k}"] = p
                row[f"mrr@{k}"] = m
            hyb_results.append(row)
        results[f"hybrid_rrf_{rrf_k}"] = hyb_results

    # ---- aggregate & print ----
    def agg(rows, key):
        return mean([r[key] for r in rows])

    print("\n" + "=" * 60)
    print(f"{'Metric':<20} " + " ".join(f"{s:<12}" for s in ["vector", "hybrid60", "hybrid30", "hybrid100"]))
    print("-" * 60)

    strategies = ["vector_only"] + [f"hybrid_rrf_{rk}" for rk in rrf_ks]
    labels = {"vector_only": "vector", **{f"hybrid_rrf_{rk}": f"hybrid{rk}" for rk in rrf_ks}}

    # Build header dynamically
    header_labels = [labels[s] for s in strategies]
    print(f"{'Metric':<20} " + " ".join(f"{l:<12}" for l in header_labels))
    print("-" * 60)

    for k in topks:
        line = f"{'hit@'+str(k):<20}"
        for s in strategies:
            line += f" {agg(results[s], f'hit@{k}'):<12.4f}"
        print(line)
        line = f"{'recall@'+str(k):<20}"
        for s in strategies:
            line += f" {agg(results[s], f'recall@{k}'):<12.4f}"
        print(line)
        line = f"{'precision@'+str(k):<20}"
        for s in strategies:
            line += f" {agg(results[s], f'precision@{k}'):<12.4f}"
        print(line)
    line = f"{'MRR@'+str(args.k):<20}"
    for s in strategies:
        line += f" {agg(results[s], f'mrr@{args.k}'):<12.4f}"
    print(line)
    print("=" * 60)

    # ---- improvement summary ----
    print("\n=== Improvement (best hybrid vs vector) ===")
    best_hybrid = None
    best_mrr = -1
    for rk in rrf_ks:
        s = f"hybrid_rrf_{rk}"
        m = agg(results[s], f"mrr@{args.k}")
        if m > best_mrr:
            best_mrr = m
            best_hybrid = s

    if best_hybrid:
        vec_mrr = agg(results["vector_only"], f"mrr@{args.k}")
        hyb_mrr = agg(results[best_hybrid], f"mrr@{args.k}")
        vec_hit = agg(results["vector_only"], f"hit@{args.k}")
        hyb_hit = agg(results[best_hybrid], f"hit@{args.k}")
        best_k = best_hybrid.split("_")[-1]
        print(f"Best RRF k: {best_k}")
        print(f"  MRR@{args.k}:  {vec_mrr:.4f} -> {hyb_mrr:.4f}  ({(hyb_mrr-vec_mrr)/max(vec_mrr,1e-9)*100:+.1f}%)")
        print(f"  hit@{args.k}:  {vec_hit:.4f} -> {hyb_hit:.4f}  ({(hyb_hit-vec_hit)/max(vec_hit,1e-9)*100:+.1f}%)")

    # ---- save ----
    out = {
        "args": vars(args),
        "topks": topks,
        "rrf_k_tested": rrf_ks,
        "num_questions": len(questions),
        "num_chunks": len(chunks),
        "results": results,
    }
    atomic_json_write(args.output, out)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
