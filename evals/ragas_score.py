"""Run Ragas scoring on already-generated eval results.

Reads evals/ragas_results.json (which must have per_question rows with
question, response, retrieved_contexts, reference, reference_contexts),
runs Ragas metrics, and writes back the merged results.

Usage:
    python evals/ragas_score.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "server" / ".env")

from langchain_groq import ChatGroq
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else 0.0


def main():
    results_path = Path(__file__).parent / "ragas_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    per_question = data["per_question"]
    retrieval_rows = data.get("retrieval_rows", [])

    print(f"Loaded {len(per_question)} questions for Ragas scoring")

    emb_model = GoogleGenerativeAIEmbeddings(
        model=data["summary"]["embedding_model"],
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    eval_ds = Dataset.from_list(per_question)

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model=data["summary"]["groq_model"],
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
        )
    )
    ragas_emb = LangchainEmbeddingsWrapper(emb_model)

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

    try:
        df = result.to_pandas()
    except Exception:
        df = result

    if hasattr(df, "to_dict"):
        ragas_scores = df.to_dict(orient="records")
    elif isinstance(df, dict):
        ragas_scores = [df]
    else:
        ragas_scores = list(df)

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
    topks = [1, 3, 5, 8, 10]
    agg = {}
    for k in topks:
        hit_key = f"hit@{k}"
        rec_key = f"recall@{k}"
        prec_key = f"precision@{k}"
        if retrieval_rows and hit_key in retrieval_rows[0]:
            agg[f"retrieval_hit@{k}"] = mean([r[hit_key] for r in retrieval_rows])
            agg[f"retrieval_recall@{k}"] = mean([r[rec_key] for r in retrieval_rows])
            agg[f"retrieval_precision@{k}"] = mean([r[prec_key] for r in retrieval_rows])
    agg["retrieval_mrr"] = mean([r["mrr"] for r in retrieval_rows])

    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
        vals = []
        for row in per_question:
            v = row.get(metric)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        agg[f"ragas_{metric}"] = mean(vals)

    # per-question combined (clean output)
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
        "embedding_model": data["summary"]["embedding_model"],
        "groq_model": data["summary"]["groq_model"],
        "gemini_fallback_model": data["summary"]["gemini_fallback_model"],
        "top_k": data["summary"]["top_k"],
        "num_questions": data["summary"]["num_questions"],
        "num_chunks": data["summary"]["num_chunks"],
        "aggregate": agg,
    }

    out = {
        "summary": summary,
        "per_question": combined,
        "retrieval_rows": retrieval_rows,
    }
    results_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- print summary ----
    print("\n========== RETRIEVAL (direct) ==========")
    print(f"N questions: {data['summary']['num_questions']}")
    for k in topks:
        print(f"  hit@{k:2d}:   {agg[f'retrieval_hit@{k}']:.3f}")
        print(f"  recall@{k:2d}: {agg[f'retrieval_recall@{k}']:.3f}")
        print(f"  precision@{k:2d}: {agg[f'retrieval_precision@{k}']:.3f}")
    print(f"  MRR:            {agg['retrieval_mrr']:.3f}")

    print("\n========== RAGAS (answer quality) ==========")
    for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
        print(f"  {m:24s}: {agg[f'ragas_{m}']:.3f}")

    print(f"\nWrote {results_path}")
    print("\nResume-ready headline:")
    top_k = data["summary"]["top_k"]
    print(f"  Retrieval hit@{top_k}: {agg[f'retrieval_hit@{top_k}']:.3f} | "
          f"MRR: {agg['retrieval_mrr']:.3f} | "
          f"Ragas answer_correctness: {agg['ragas_answer_correctness']:.3f} | "
          f"Ragas faithfulness: {agg['ragas_faithfulness']:.3f}")


if __name__ == "__main__":
    main()
