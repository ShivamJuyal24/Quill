"""Offline Ragas evaluation for the Quill RAG pipeline.

Loads `evals/rag_eval.json` (questions + gold chunks + reference answers),
retrieves the top-K chunks with Gemini embeddings, generates an answer with
Groq (rotating through the keys in GROQ_API_KEYS), then optionally scores
with Ragas:

  - Context Precision@K
  - Context Recall@K
  - Faithfulness
  - Answer Relevancy
  - Answer Correctness

Plus retrieval-only metrics computed directly (hit@K, MRR, recall@K) so you
get resume-ready numbers even for the unanswerable questions.

Quota behaviour: when every Groq key has hit its daily limit the script saves
what it has and stops WITHOUT recording placeholder answers. Run it again after
the quotas reset and it continues from the first unanswered question.

Usage:
    python evals/ragas_eval.py --skip-ragas
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(ROOT / "server" / ".env")

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

from eval_utils import RAGAS_METRICS, atomic_json_write, content_hash, dataset_fingerprint, exception_contains, load_json_strict, merge_ragas_scores, numeric_mean, ragas_dataset_rows, result_rows, scores_complete, validate_answer_rows, validate_output_path
from key_pool import GroqKeyPool, GroqPoolChatModel, AllKeysExhausted

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
    p.add_argument("--k", type=int, default=4, help="primary top-K (retrieval + ragas; matches production 4-chunk context)")
    p.add_argument(
        "--topks",
        default="1,3,5,8,10",
        help="comma-separated K values for retrieval metrics",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    p.add_argument("--groq-model", default=os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"))
    p.add_argument("--gemini-model", default=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"))
    p.add_argument(
        "--gemini-fallback",
        action="store_true",
        help="when all Groq keys are spent, answer with Gemini instead of stopping "
        "(off by default so every answer comes from the same model)",
    )
    p.add_argument("--cache", default=str(Path(__file__).parent / ".embeddings_cache.json"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--resume-from", default=str(Path(__file__).parent / "ragas_results.json"),
                   help="existing answer records to reuse and update in place")
    p.add_argument("--output", default=str(Path(__file__).parent / "ragas_results.json"),
                   help="output artifact; defaults to the same ragas_results.json for safe in-place updates")
    p.add_argument("--max-wait", type=float, default=90.0)
    p.add_argument("--limit", type=int, default=None, help="run only first N questions (dry run)")
    p.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between questions")
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
# Answer generation (Groq key pool; optional Gemini fallback)
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
    def __init__(self, pool, groq_model, gemini_model, gemini_api_key, use_gemini_fallback=False):
        self.pool = pool
        self.groq_model = groq_model
        self.gemini_model_name = gemini_model
        self.gemini_api_key = gemini_api_key
        self.use_gemini_fallback = use_gemini_fallback

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
        """Returns (answer, answered_by).

        Raises AllKeysExhausted when no provider can answer because of quota.
        Returns (None, None) for other failures so the caller skips the
        question (it is retried on the next run) instead of saving a placeholder.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        user_content = self._build_user_content(question, contexts)
        groq_exhausted = False

        # --- Groq (rotating keys) ---
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
            resp = self.pool.invoke(
                messages, model=self.groq_model, temperature=0.2, max_tokens=2048
            )
            ans = (resp.content or "").strip()
            if ans:
                return ans, f"groq:{self.groq_model}"
            print("  [groq returned an empty answer]", flush=True)
        except AllKeysExhausted:
            groq_exhausted = True
            if not self.use_gemini_fallback:
                raise
            print("  [all Groq keys are out of quota; trying Gemini fallback]", flush=True)
        except Exception as e:
            print(f"  [groq failed: {e}]", flush=True)

        # --- Gemini fallback (only with --gemini-fallback) ---
        if self.use_gemini_fallback:
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
                    return ans, f"gemini:{self.gemini_model_name}"
            except Exception as e:
                print(f"  [gemini fallback failed: {e}]", flush=True)
            if groq_exhausted:
                raise AllKeysExhausted("Groq keys and Gemini fallback are all out of quota")

        return None, None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def is_placeholder_response(value):
    if not isinstance(value, str) or not value.strip():
        return True
    return value.strip().lower() == "i don't have enough information in the provided context to answer that."


def load_existing_results(output_paths, questions, current_fingerprint=None):
    """Load previously saved results so we can skip already-processed questions.

    Any non-empty saved response is treated as completed, including a refusal
    or "I don't know" answer. This prevents silently regenerating existing rows.
    """
    by_question = {str(row["id"]): row for row in questions}
    answer_by_id = {}
    retrieval_by_id = {}
    for path in output_paths:
        p = Path(path)
        if not p.exists():
            continue
        d = load_json_strict(p)
        saved_fingerprint = d.get("summary", {}).get("dataset_fingerprint")
        if saved_fingerprint and current_fingerprint and saved_fingerprint != current_fingerprint:
            raise ValueError(f"Saved results in {p.name} were generated from a different evaluation dataset")
        saved = validate_answer_rows(d.get("per_question", []), p.name)
        for row in saved:
            key = str(row["id"])
            if key not in by_question:
                raise ValueError(f"Saved result contains unknown question ID {row['id']!r} in {p.name}")
            if row["question"] != by_question[key]["question"]:
                raise ValueError(f"Saved question differs from current dataset for ID {row['id']!r}")
            if row.get("response") and not is_placeholder_response(row.get("response")):
                answer_by_id[key] = row
        for row in d.get("retrieval_rows", []):
            retrieval_by_id[str(row.get("id"))] = row
    per_question = list(answer_by_id.values())
    done = set(answer_by_id)
    retrieval_rows = [row for key, row in retrieval_by_id.items() if key in done]
    return per_question, retrieval_rows, done


def save_incremental(output_path, per_question, retrieval_rows, summary_meta):
    """Write current progress to disk after each question."""
    out = {
        "summary": summary_meta,
        "per_question": per_question,
        "retrieval_rows": retrieval_rows,
    }
    atomic_json_write(output_path, out, allow_protected=True)


def main():
    args = parse_args()
    validate_output_path(args.output, allow_protected=True)
    topks = sorted({int(x) for x in args.topks.split(",") if x.strip()})
    if args.k not in topks:
        topks = sorted(set(topks) | {args.k})

    data_path = Path(args.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    all_questions = data["questions"]
    original_rows_by_id = {str(row["id"]): row for row in data.get("per_question", [])}
    original_retrieval_by_id = {str(row.get("id")): row for row in data.get("retrieval_rows", [])}
    current_fingerprint = dataset_fingerprint(all_questions, chunks)
    questions_to_process = all_questions[: args.limit] if args.limit else all_questions

    chunk_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_ids = list(chunk_by_id.keys())
    print(f"Chunks: {len(chunks)}  Questions: {len(questions_to_process)}  Top-K: {args.k}")
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
    cache.setdefault("questions", {})

    if cache.get("model") != args.embedding_model:
        cache = {"model": args.embedding_model, "chunks": {}, "questions": {}}

    chunk_hash_by_id = {cid: content_hash(chunk_by_id[cid]) for cid in chunk_ids}
    missing_hashes = list(dict.fromkeys(chunk_hash_by_id[cid] for cid in chunk_ids if chunk_hash_by_id[cid] not in cache["chunks"]))
    text_by_hash = {content_hash(text): text for text in chunk_by_id.values()}
    missing_chunks = [key for key in missing_hashes if key in text_by_hash]
    if missing_chunks:
        print(f"Embedding {len(missing_chunks)} chunks...")
        vecs = emb.embed_texts([text_by_hash[key] for key in missing_chunks])
        for key, v in zip(missing_chunks, vecs):
            cache["chunks"][key] = v.tolist()
        if not args.no_cache:
            atomic_json_write(cache_path, cache)
    else:
        print("Chunk embeddings cached.")

    chunk_matrix = np.asarray([cache["chunks"][chunk_hash_by_id[cid]] for cid in chunk_ids], dtype=np.float32)

    # ---- answer generator (Groq key pool) ----
    pool = GroqKeyPool(max_wait=args.max_wait)
    gen = AnswerGenerator(
        pool,
        args.groq_model,
        args.gemini_model,
        os.getenv("GEMINI_API_KEY"),
        use_gemini_fallback=args.gemini_fallback,
    )

    # ---- load existing progress ----
    resume_path = Path(args.resume_from).resolve()
    output_path = Path(args.output).resolve()
    sources = [resume_path]
    if output_path != resume_path:
        sources.append(output_path)
    per_question, saved_retrieval_rows, done_ids = load_existing_results(sources, all_questions, current_fingerprint)
    by_id = {str(row["id"]): row for row in per_question}
    per_question = [dict(by_id.get(str(row["id"]), row)) for row in original_rows_by_id.values()] if original_rows_by_id else per_question
    if not original_rows_by_id:
        per_question = [dict(row) for row in per_question]
    retrieval_rows = saved_retrieval_rows if saved_retrieval_rows else []
    if not retrieval_rows and per_question:
        # Reconstruct retrieval_rows from per_question data (retrieved_ids, gold_ids, mrr)
        # when saved retrieval_rows is empty (e.g., rag_eval.json has no retrieval_rows key)
        for row in per_question:
            gold = set(row.get("gold_ids", []))
            retrieved = row.get("retrieved_ids", [])
            ret_row = {
                "id": row["id"],
                "type": row.get("type"),
                "difficulty": row.get("difficulty"),
                "answerable": row.get("answerable"),
                "gold": sorted(gold),
                "retrieved": retrieved,
                "first_relevant_rank": None,
                "mrr": row.get("mrr", 0.0),
            }
            for rank, cid in enumerate(retrieved, 1):
                if cid in gold:
                    ret_row["first_relevant_rank"] = rank
                    break
            for k in topks:
                topk_ids = set(retrieved[:k])
                ret_row[f"hit@{k}"] = int(bool(gold & topk_ids)) if gold else 0.0
                ret_row[f"recall@{k}"] = (len(gold & topk_ids) / len(gold)) if gold else 0.0
                ret_row[f"precision@{k}"] = (len(gold & topk_ids) / k) if gold else 0.0
            retrieval_rows.append(ret_row)
    elif not retrieval_rows:
        retrieval_rows = [original_retrieval_by_id[str(row["id"])] for row in all_questions if str(row["id"]) in original_retrieval_by_id]
    remaining = [q for q in questions_to_process if str(q["id"]) not in done_ids]
    print(f"Already done: {len(done_ids)}  Remaining: {len(remaining)}")

    maxk = max(topks)
    stopped_early = False
    skipped = 0

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

        # --- generate answer (Groq key pool) ---
        original_row = original_rows_by_id.get(str(qid), {})
        if original_row.get("retrieved_contexts") and not args.no_cache:
            contexts_for_answer = list(original_row["retrieved_contexts"])
            ret_row = None
        else:
            # --- retrieve top-K ---
            question_hash = content_hash(question)
            if question_hash not in cache["questions"]:
                cache["questions"][question_hash] = emb.embed_query(question).tolist()
                if not args.no_cache:
                    atomic_json_write(cache_path, cache)
            qvec = np.asarray(cache["questions"][question_hash], dtype=np.float32)
            fetch_pool = max(maxk, 20)
            idxs, sims = cosine_topk(qvec, chunk_matrix, fetch_pool)
            retrieved_ids = [chunk_ids[i] for i in idxs]
            retrieved_texts = [chunk_by_id[c] for c in retrieved_ids]
            cohere_key = os.getenv("COHERE_API_KEY")
            print(f"  [rerank] key present: {bool(cohere_key)}")
            if cohere_key:
                import time as _time
                import cohere as _cohere
                _client = _cohere.Client(cohere_key)
                # Respect Cohere free tier: 10 requests/minute → ≥6s between calls.
                _time.sleep(6.5)
                for _attempt in range(3):
                    try:
                        _resp = _client.rerank(
                            model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
                            query=question,
                            documents=retrieved_texts,
                            top_n=4,
                        )
                        _ranked = [retrieved_ids[r.index] for r in (_resp.results or [])]
                        if _ranked:
                            retrieved_ids = _ranked
                            retrieved_texts = [chunk_by_id[c] for c in retrieved_ids]
                        break
                    except Exception as _e:
                        msg = str(_e)
                        if _attempt == 2:
                            print(f"  [rerank failed: {_e}]")
                        elif "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                            _delay = min(2 ** (_attempt + 2), 30)
                            print(f"  [rerank rate-limited, retry {_attempt+1}/3 in {_delay}s]")
                            _time.sleep(_delay)
                        else:
                            _delay = [4, 6][_attempt]
                            print(f"  [rerank retry {_attempt+1}/3 in {_delay}s]")
                            _time.sleep(_delay)

            # retrieval metrics (computed directly; only saved once the answer exists)
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
                ret_row[f"hit@{k}"] = int(bool(gold & topk_ids)) if gold else 0.0
                ret_row[f"recall@{k}"] = (len(gold & topk_ids) / len(gold)) if gold else 0.0
                ret_row[f"precision@{k}"] = (len(gold & topk_ids) / k) if gold else 0.0
            contexts_for_answer = retrieved_texts[: args.k]

        try:
            answer, answered_by = gen.generate(question, contexts_for_answer)
        except AllKeysExhausted as e:
            stopped_early = True
            print(f"\n  [quota exhausted: {e}]")
            break

        if not answer:
            skipped += 1
            print("  [no answer produced - skipped, will be retried on the next run]")
            continue

        print(f"  answer ({answered_by}): {safe_str(answer[:120])}")

        if original_row:
            updated_row = dict(original_row)
            updated_row["response"] = answer
            updated_row["answered_by"] = answered_by
            for i, existing in enumerate(per_question):
                if str(existing.get("id")) == str(qid):
                    per_question[i] = updated_row
                    break
            else:
                per_question.append(updated_row)
        else:
            per_question.append(
                {
                    "id": qid,
                    "type": q.get("type"),
                    "difficulty": q.get("difficulty"),
                    "answerable": answerable,
                    "question": question,
                    "response": answer,
                    "answered_by": answered_by,
                    "retrieved_contexts": contexts_for_answer,
                    "reference": reference,
                    "reference_contexts": [chunk_by_id[c] for c in gold if c in chunk_by_id],
                    "retrieved_ids": retrieved_ids[: args.k] if "retrieved_ids" in locals() else [],
                    "gold_ids": sorted(gold),
                    "mrr": mrr if "mrr" in locals() else 0.0,
                }
            )

        if ret_row is not None and str(qid) not in {str(r.get("id")) for r in retrieval_rows}:
            retrieval_rows.append(ret_row)

        # save after every question so we can resume on quota / timeout
        summary_meta = {
            "embedding_model": args.embedding_model,
            "groq_model": args.groq_model,
            "gemini_fallback_model": args.gemini_model,
            "top_k": args.k,
            "num_questions": len(all_questions),
            "num_chunks": len(chunks),
            "dataset_fingerprint": current_fingerprint,
            "aggregate": {},
        }
        save_incremental(output_path, per_question, retrieval_rows, summary_meta)

        if args.sleep:
            time.sleep(args.sleep)

    # ---- stop here if the run could not finish ----
    summary_meta = {
        "embedding_model": args.embedding_model,
        "groq_model": args.groq_model,
        "gemini_fallback_model": args.gemini_model,
        "top_k": args.k,
        "num_questions": len(all_questions),
        "num_chunks": len(chunks),
        "dataset_fingerprint": current_fingerprint,
        "aggregate": {},
    }
    save_incremental(output_path, per_question, retrieval_rows, summary_meta)
    if stopped_early or skipped:
        print("\n" + "=" * 60)
        print(f"Answered {len(per_question)}/{len(all_questions)} questions so far "
              f"({len(all_questions) - len(per_question)} still to do).")
        if stopped_early:
            print("All Groq keys are out of quota. Progress is saved in:")
        else:
            print(f"{skipped} question(s) produced no answer. Progress is saved in:")
        print(f"  {output_path}")
        print("Run the same command again after the quotas reset (or add more keys) to continue.")
        print("=" * 60)
        return

    # ---- Resumable per-question Ragas evaluation ----
    if args.skip_ragas:
        print("\n=== Skipping Ragas (--skip-ragas) ===")
        ragas_scores = {}
    else:
        print("\n=== Running Ragas ===")
        ragas_llm = LangchainLLMWrapper(
            GroqPoolChatModel(pool=pool, model=args.groq_model, temperature=0.2)
        )
        ragas_emb = LangchainEmbeddingsWrapper(emb.model)

        metrics = [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]
        for position, row in enumerate(per_question, 1):
            if scores_complete(row):
                continue
            print(f"[{position}/{len(per_question)}] Ragas question {row['id']}", flush=True)
            try:
                result = evaluate(
                    dataset=Dataset.from_list(ragas_dataset_rows([row])),
                    metrics=metrics,
                    llm=ragas_llm,
                    embeddings=ragas_emb,
                    show_progress=False,
                    raise_exceptions=True,
                    column_map={
                        "user_input": "question",
                        "response": "response",
                        "retrieved_contexts": "retrieved_contexts",
                        "reference": "reference",
                        "reference_contexts": "reference_contexts",
                    },
                )
                per_question[:] = merge_ragas_scores(per_question, result_rows(result, [row]))
                for metric in RAGAS_METRICS:
                    values = [
                        item[metric] for item in per_question
                        if item.get(metric) is not None
                        and not (isinstance(item[metric], float) and math.isnan(item[metric]))
                    ]
                    summary_meta["aggregate"][f"ragas_{metric}"] = numeric_mean(values)
                save_incremental(output_path, per_question, retrieval_rows, summary_meta)
            except Exception as error:
                save_incremental(output_path, per_question, retrieval_rows, summary_meta)
                if exception_contains(error, AllKeysExhausted):
                    print(f"All daily Groq keys exhausted during Ragas; progress is saved to {output_path}.")
                    return
                raise

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

    summary = {
        "embedding_model": args.embedding_model,
        "groq_model": args.groq_model,
        "gemini_fallback_model": args.gemini_model,
        "top_k": args.k,
        "num_questions": len(all_questions),
        "num_chunks": len(chunks),
        "dataset_fingerprint": current_fingerprint,
        "aggregate": agg,
    }

    out = {
        "summary": summary,
        "per_question": per_question,
        "retrieval_rows": retrieval_rows,
    }
    out["source_input"] = str(resume_path)
    atomic_json_write(output_path, out, allow_protected=True)

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

    print(f"\nWrote {output_path}")
    print("\nResume-ready headline:")
    print(f"  Retrieval hit@{args.k}: {agg[f'retrieval_hit@{args.k}']:.3f} | "
          f"MRR: {agg['retrieval_mrr']:.3f}", end="")
    if not args.skip_ragas:
        print(f" | Ragas answer_correctness: {agg['ragas_answer_correctness']:.3f} | Ragas faithfulness: {agg['ragas_faithfulness']:.3f}")
    else:
        print()


if __name__ == "__main__":
    main()
