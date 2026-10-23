"""Offline Ragas evaluation for the Quill RAG pipeline.

Loads `evals/rag_eval.json` (questions + gold chunks + reference answers),
retrieves the top-K chunks with Gemini embeddings, generates an answer with
Groq (primary) / Gemini (fallback) exactly like the production app, then scores
with Ragas:

  - Context Precision@K
  - Context Recall@K
  - Faithfulness
  - Answer Relevancy
  - Answer Correctness

Plus retrieval-only metrics computed directly (hit@K, MRR, recall@K) so you
get resume-ready numbers even for the unanswerable questions.

Usage:
    python evals/ragas_eval.py
    python evals/ragas_eval.py --k 8 --topks 1,3,5,8,10 --limit 10
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "server" / ".env")

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def safe_str(s):
    """Make a string safe to print on any console (drop unencodable chars)."""
    if s is None:
        return ""
    try:
        return str(s).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8"
        )
    except Exception:
        return str(s).encode("ascii", errors="replace").decode("ascii")


def parse_args():
    p = argparse.ArgumentParser(description="Ragas offline eval for Quill")
    p.add_argument("--data", default=str(Path(__file__).parent / "rag_eval.json"))
    p.add_argument("--k", type=int, default=8, help="primary top-K (retrieval + ragas)")
    p.add_argument(
        "--topks",
        default="1,3,5,8,10",
        help="comma-separated K values for retrieval metrics",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
    )
    p.add_argument("--groq-model", default=os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"))
    p.add_argument("--gemini-model", default=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"))
    p.add_argument("--cache", default=str(Path(__file__).parent / ".embeddings_cache.json"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output", default=str(Path(__file__).parent / "ragas_results.json"))
    p.add_argument("--limit", type=int, default=None, help="run only first N questions (dry run)")
    p.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between API batches")
    p.add_argument("--skip-ragas", action="store_true", help="skip Ragas scoring (retrieval + answer generation only)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Embedding helpers (same model family as production)
# --------------------------------------------------------------------------- #
class Embedder:
    def __init__(self, model, api_key):
        self.model = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key
        )

    def embed_texts(self, texts, task_type="retrieval_document"):
        # langchain wrapper accepts a list and returns list of vectors
        return np.asarray(self.model.embed_documents(texts), dtype=np.float32)

    def embed_query(self, text):
        return np.asarray(self.model.embed_query(text), dtype=np.float32)


def cosine_topk(query_vec, doc_matrix, k):
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    dn = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)
    sims = dn @ qn
    order = np.argsort(-sims)[:k]
    return order.tolist(), sims[order].tolist()


# --------------------------------------------------------------------------- #
# Answer generation (Groq primary -> Gemini fallback), mirrors production
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a PDF document. "
    "Answer using ONLY the provided context excerpts. If the context does not "
    "contain the answer, say you don't know - never invent facts. When you use "
    "an excerpt, cite only its numbered label, for example [1] or [2]. Do not "
    "invent page numbers, line numbers, URLs, or any other citation format. "
    "Use plain Markdown and be concise and specific."
)


class AnswerGenerator:
    def __init__(self, groq_model, gemini_model, groq_api_key, gemini_api_key):
        self.groq = ChatGroq(
            model=groq_model,
            groq_api_key=groq_api_key,
            temperature=0.2,
            max_tokens=2048,
        )
        self.gemini = GoogleGenerativeAIEmbeddings  # placeholder, unused directly
        self.gemini_model_name = gemini_model
        self.gemini_api_key = gemini_api_key

    def _build_user_content(self, question, contexts):
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return (
            f'Document: "evaluation document"\n\n'
            f"Context excerpts:\n{ctx}\n\n"
            f"Question: {question}\n\n"
            "Citation rule: cite only the available chunk labels [1], [2], etc. "
            "Do not invent page numbers, line numbers, URLs, or any other citation format."
        )

    def generate(self, question, contexts):
        """Groq primary, Gemini fallback. Returns answer text."""
        user_content = self._build_user_content(question, contexts)

        # --- Groq attempt ---
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
            resp = self.groq.invoke(messages)
            ans = (resp.content or "").strip()
            if ans:
                return ans
        except Exception as e:
            print(f"  [groq failed: {e}]", flush=True)

        # --- Gemini fallback ---
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel(self.gemini_model_name)
            full = f"{SYSTEM_PROMPT}\n\n{user_content}"
            resp = model.generate_content(
                full,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
            )
            ans = (resp.text or "").strip()
            if ans:
                return ans
        except Exception as e:
            print(f"  [gemini fallback failed: {e}]", flush=True)

        return "I don't have enough information in the provided context to answer that."


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_existing_results(output_path):
    """Load previously saved results so we can skip already-processed questions."""
    p = Path(output_path)
    if not p.exists():
        return [], [], set()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("per_question", []), d.get("retrieval_rows", ), set(
            r["id"] for r in d.get("per_question", [])
        )
    except Exception:
        return [], [], set()


def save_incremental(output_path, per_question, retrieval_rows, summary_meta):
    """Write current progress to disk after each question."""
    out = {
        "summary": summary_meta,
        "per_question": per_question,
        "retrieval_rows": retrieval_rows,
    }
    Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    topks = sorted({int(x) for x in args.topks.split(",") if x.strip()})
    if args.k not in topks:
        topks = sorted(set(topks) | {args.k})

    data_path = Path(args.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    all_questions = data["questions"]
    if args.limit:
        all_questions = all_questions[: args.limit]

    chunk_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_ids = list(chunk_by_id.keys())
    print(f"Chunks: {len(chunks)}  Questions: {len(all_questions)}  Top-K: {args.k}")
    print(f"Embedding model: {args.embedding_model}")

    emb = Embedder(args.embedding_model, os.getenv("GEMINI_API_KEY"))

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

    if cache.get("model") != args.embedding_model:
        cache = {"model": args.embedding_model, "chunks": {}}

    missing_chunks = [cid for cid in chunk_ids if cid not in cache["chunks"]]
    if missing_chunks:
        print(f"Embedding {len(missing_chunks)} chunks...")
        vecs = emb.embed_texts([chunk_by_id[c] for c in missing_chunks])
        for cid, v in zip(missing_chunks, vecs):
            cache["chunks"][cid] = v.tolist()
        if not args.no_cache:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
    else:
        print("Chunk embeddings cached.")

    chunk_matrix = np.asarray([cache["chunks"][cid] for cid in chunk_ids], dtype=np.float32)

    # ---- answer generator ----
    gen = AnswerGenerator(
        args.groq_model,
        args.gemini_model,
        os.getenv("GROQ_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
    )

    # ---- load existing progress ----
    per_question, retrieval_rows, done_ids = load_existing_results(args.output)
    remaining = [q for q in all_questions if q["id"] not in done_ids]
    print(f"Already done: {len(done_ids)}  Remaining: {len(remaining)}")

    maxk = max(topks)

    # ---- process remaining questions one at a time ----
    for idx, q in enumerate(remaining, 1):
        qid = q["id"]
        gold = set(q.get("all_expected_chunk_ids", []))
        answerable = q.get("answerable", True)
        question = q["question"]
        reference = q.get("reference_answer", "")

        print(
            f"\n[{idx}/{len(remaining)}] {qid} ({q.get('difficulty')}, {q.get('type')}): "
            f"{safe_str(question[:70])}"
        )

        # --- retrieve top-K ---
        qvec = emb.embed_query(question)
        idxs, sims = cosine_topk(qvec, chunk_matrix, maxk)
        retrieved_ids = [chunk_ids[i] for i in idxs]
        retrieved_texts = [chunk_by_id[c] for c in retrieved_ids]

        # retrieval metrics (computed directly)
        ret_row = {
            "id": qid,
            "type": q.get("type"),
            "difficulty": q.get("difficulty"),
            "answerable": answerable,
            "gold": sorted(gold),
            "retrieved": retrieved_ids[: args.k],
            "first_relevant_rank": None,
        }
        for rank, cid in enumerate(retrieved_ids, 1):
            if cid in gold:
                ret_row["first_relevant_rank"] = rank
                break
        mrr = (1.0 / ret_row["first_relevant_rank"]) if ret_row["first_relevant_rank"] else 0.0
        ret_row["mrr"] = mrr
        for k in topks:
            topk_ids = set(retrieved_ids[:k])
            ret_row[f"hit@{k}"] = int(bool(gold & topk_ids)) if gold else 0
            ret_row[f"recall@{k}"] = (len(gold & topk_ids) / len(gold)) if gold else 0.0
            ret_row[f"precision@{k}"] = (len(gold & topk_ids) / k) if gold else 0.0
        retrieval_rows.append(ret_row)

        # --- generate answer (Groq primary, Gemini fallback) ---
        contexts_for_answer = retrieved_texts[: args.k]
        answer = gen.generate(question, contexts_for_answer)
        print(f"  answer: {safe_str(answer[:120])}")

        per_question.append(
            {
                "id": qid,
                "type": q.get("type"),
                "difficulty": q.get("difficulty"),
                "answerable": answerable,
                "question": question,
                "response": answer,
                "retrieved_contexts": contexts_for_answer,
                "reference": reference,
                "reference_contexts": [chunk_by_id[c] for c in gold if c in chunk_by_id],
                "retrieved_ids": retrieved_ids[: args.k],
                "gold_ids": sorted(gold),
                "mrr": mrr,
            }
        )

        # save after every question so we can resume on timeout
        summary_meta = {
            "embedding_model": args.embedding_model,
            "groq_model": args.groq_model,
            "gemini_fallback_model": args.gemini_model,
            "top_k": args.k,
            "num_questions": len(all_questions),
            "num_chunks": len(chunks),
            "aggregate": {},
        }
        save_incremental(args.output, per_question, retrieval_rows, summary_meta)

        if args.sleep:
            time.sleep(args.sleep)

    # ---- Ragas evaluation (run once on all completed rows) ----
    if args.skip_ragas:
        print("\n=== Skipping Ragas (--skip-ragas) ===")
        ragas_scores = {}
    else:
        print("\n=== Running Ragas ===")
        # rebuild full rows for Ragas (per_question has all needed fields)
        eval_ds = Dataset.from_list(per_question)

        ragas_llm = LangchainLLMWrapper(
            ChatGroq(
                model=args.groq_model,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2,
            )
        )
        ragas_emb = LangchainEmbeddingsWrapper(emb.model)

        metrics = [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]
        result = evaluate(
            dataset=eval_ds,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_emb,
            show_progress=True,
            column_map={
                "user_input": "question",
                "response": "response",
                "retrieved_contexts": "retrieved_contexts",
                "reference": "reference",
                "reference_contexts": "reference_contexts",
            },
        )

        # ragas 0.4.3 returns a pandas DataFrame-like EvaluationResult; convert
        try:
            df = result.to_pandas()
        except Exception:
            df = result

        ragas_scores = {}
        if hasattr(df, "to_dict"):
            ragas_scores = df.to_dict(orient="records")
        elif isinstance(df, dict):
            ragas_scores = [df]
        else:
            ragas_scores = list(df)

        # attach ragas scores back to per_question by question text
        score_by_question = {}
        for rec in ragas_scores:
            qtext = rec.get("user_input") or rec.get("question")
            if qtext:
                score_by_question[qtext] = rec

        for row in per_question:
            rec = score_by_question.get(row["question"], {})
            for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
                row[m] = rec.get(m)

    # ---- aggregate ----
    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else 0.0

    agg = {}
    for k in topks:
        vals = [r[f"hit@{k}"] for r in retrieval_rows]
        agg[f"retrieval_hit@{k}"] = mean(vals)
        vals = [r[f"recall@{k}"] for r in retrieval_rows]
        agg[f"retrieval_recall@{k}"] = mean(vals)
        vals = [r[f"precision@{k}"] for r in retrieval_rows]
        agg[f"retrieval_precision@{k}"] = mean(vals)
    agg["retrieval_mrr"] = mean([r["mrr"] for r in retrieval_rows])

    if not args.skip_ragas:
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
            vals = []
            for row in per_question:
                v = row.get(metric)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
            agg[f"ragas_{metric}"] = mean(vals)

    # final per_question combined (clean output)
    combined = []
    for r in retrieval_rows:
        row = {
            "id": r["id"],
            "type": r["type"],
            "difficulty": r["difficulty"],
            "answerable": r["answerable"],
            **{k: v for k, v in r.items() if k.startswith(("hit@", "recall@", "precision@", "mrr", "gold", "retrieved", "first_relevant_rank"))},
        }
        qrow = next((x for x in per_question if x["id"] == r["id"]), None)
        if qrow:
            for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
                row[m] = qrow.get(m)
        combined.append(row)

    summary = {
        "embedding_model": args.embedding_model,
        "groq_model": args.groq_model,
        "gemini_fallback_model": args.gemini_model,
        "top_k": args.k,
        "num_questions": len(all_questions),
        "num_chunks": len(chunks),
        "aggregate": agg,
    }

    out = {
        "summary": summary,
        "per_question": per_question,
        "retrieval_rows": retrieval_rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- print summary ----
    print("\n========== RETRIEVAL (direct) ==========")
    print(f"N questions: {len(all_questions)}")
    for k in topks:
        print(f"  hit@{k:2d}:   {agg[f'retrieval_hit@{k}']:.3f}")
        print(f"  recall@{k:2d}: {agg[f'retrieval_recall@{k}']:.3f}")
        print(f"  precision@{k:2d}: {agg[f'retrieval_precision@{k}']:.3f}")
    print(f"  MRR:            {agg['retrieval_mrr']:.3f}")

    if not args.skip_ragas:
        print("\n========== RAGAS (answer quality) ==========")
        for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
            print(f"  {m:24s}: {agg[f'ragas_{m}']:.3f}")

    print(f"\nWrote {args.output}")
    print("\nResume-ready headline:")
    print(f"  Retrieval hit@{args.k}: {agg[f'retrieval_hit@{args.k}']:.3f} | "
          f"MRR: {agg['retrieval_mrr']:.3f}", end="")
    if not args.skip_ragas:
        print(f" | Ragas answer_correctness: {agg['ragas_answer_correctness']:.3f} | Ragas faithfulness: {agg['ragas_faithfulness']:.3f}")
    else:
        print()


if __name__ == "__main__":
    main()
