"""LLM-as-judge evaluation for Quill RAG answers.

Reads evals/ragas_results.json (50 questions with generated answers),
scores each answer with Groq for:
  - correctness: does the answer match the reference answer? (0-1)
  - faithfulness: is the answer grounded in the retrieved context? (0-1)
  - relevancy: does the answer actually address the question? (0-1)

No Ragas dependency — uses Groq directly as the judge. Fast and reliable.

Usage:
    python evals/llm_judge_eval.py
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
from langchain_core.messages import HumanMessage, SystemMessage


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else 0.0


CORRECTNESS_PROMPT = """You are an expert evaluator. Score how well the candidate answer matches the reference answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Score 0.0 to 1.0:
- 1.0: Candidate fully matches reference (all key facts present, no contradictions)
- 0.7: Candidate captures most key facts, minor omissions or wording differences
- 0.4: Candidate partially correct, missing important facts
- 0.0: Candidate completely wrong or contradicts reference

Respond with ONLY a number between 0 and 1 (e.g., 0.85)."""


FAITHFULNESS_PROMPT = """You are an expert evaluator. Score how faithful the candidate answer is to the provided context.

Question: {question}
Context: {context}
Candidate answer: {candidate}

Score 0.0 to 1.0:
- 1.0: Every claim in the candidate is supported by the context
- 0.7: Most claims supported, minor unsupported details
- 0.4: Some claims unsupported or contradicted by context
- 0.0: Candidate is mostly hallucinated or contradicts context

If the candidate says "I don't know" or refuses to answer, score 0.0.

Respond with ONLY a number between 0 and 1 (e.g., 0.85)."""


RELEVANCY_PROMPT = """You are an expert evaluator. Score how relevant the candidate answer is to the question.

Question: {question}
Candidate answer: {candidate}

Score 0.0 to 1.0:
- 1.0: Directly and fully answers the question
- 0.7: Answers the question with minor tangents
- 0.4: Partially addresses the question
- 0.0: Does not answer the question at all

If the candidate says "I don't know" or refuses to answer, score 0.0.

Respond with ONLY a number between 0 and 1 (e.g., 0.85)."""


def judge_score(llm, prompt_template, **kwargs):
    """Ask Groq to produce a 0-1 score. Returns float or None."""
    prompt = prompt_template.format(**kwargs)
    try:
        messages = [
            SystemMessage(content="You are a precise scoring evaluator. Respond with only a decimal number between 0 and 1."),
            HumanMessage(content=prompt),
        ]
        resp = llm.invoke(messages)
        text = (resp.content or "").strip()
        # Extract the first number found
        import re
        m = re.search(r"0?\.\d+|1\.0|1|0", text)
        if m:
            val = float(m.group())
            return max(0.0, min(1.0, val))
    except Exception as e:
        print(f"  [judge error: {e}]", flush=True)
    return None


def main():
    results_path = Path(__file__).parent / "ragas_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    per_question = data["per_question"]
    retrieval_rows = data.get("retrieval_rows", [])

    print(f"Loaded {len(per_question)} questions")

    llm = ChatGroq(
        model=data["summary"]["groq_model"],
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )

    correctness_scores = []
    faithfulness_scores = []
    relevancy_scores = []

    for i, row in enumerate(per_question, 1):
        qid = row["id"]
        question = row["question"]
        candidate = row["response"]
        reference = row.get("reference", "")
        context = "\n".join(row.get("retrieved_contexts", []))

        print(f"[{i}/{len(per_question)}] {qid}", end=" ", flush=True)

        c = judge_score(llm, CORRECTNESS_PROMPT,
                        question=question, reference=reference, candidate=candidate)
        f = judge_score(llm, FAITHFULNESS_PROMPT,
                        question=question, context=context, candidate=candidate)
        r = judge_score(llm, RELEVANCY_PROMPT,
                        question=question, candidate=candidate)

        if c is not None:
            correctness_scores.append(c)
            row["judge_correctness"] = c
        if f is not None:
            faithfulness_scores.append(f)
            row["judge_faithfulness"] = f
        if r is not None:
            relevancy_scores.append(r)
            row["judge_relevancy"] = r

        print(f"correctness={c} faithfulness={f} relevancy={r}", flush=True)

    # ---- aggregate retrieval ----
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

    # ---- aggregate judge ----
    agg["judge_correctness"] = mean(correctness_scores)
    agg["judge_faithfulness"] = mean(faithfulness_scores)
    agg["judge_relevancy"] = mean(relevancy_scores)
    agg["judge_n"] = len(correctness_scores)

    # per-question combined
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
            for m in ["judge_correctness", "judge_faithfulness", "judge_relevancy"]:
                row[m] = qrow.get(m)
        combined.append(row)

    summary = {
        "embedding_model": data["summary"]["embedding_model"],
        "groq_model": data["summary"]["groq_model"],
        "top_k": data["summary"]["top_k"],
        "num_questions": data["summary"]["num_questions"],
        "num_chunks": data["summary"]["num_chunks"],
        "judge_n": agg["judge_n"],
        "aggregate": agg,
    }

    out = {
        "summary": summary,
        "per_question": combined,
        "retrieval_rows": retrieval_rows,
    }
    results_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- print summary ----
    print("\n========== RETRIEVAL (all 50 questions) ==========")
    print(f"N questions: {data['summary']['num_questions']}")
    for k in topks:
        print(f"  hit@{k:2d}:   {agg[f'retrieval_hit@{k}']:.3f}")
        print(f"  recall@{k:2d}: {agg[f'retrieval_recall@{k}']:.3f}")
        print(f"  precision@{k:2d}: {agg[f'retrieval_precision@{k}']:.3f}")
    print(f"  MRR:            {agg['retrieval_mrr']:.3f}")

    print("\n========== LLM-AS-JUDGE (answer quality) ==========")
    print(f"  correctness:     {agg['judge_correctness']:.3f} (n={agg['judge_n']})")
    print(f"  faithfulness:    {agg['judge_faithfulness']:.3f} (n={agg['judge_n']})")
    print(f"  relevancy:       {agg['judge_relevancy']:.3f} (n={agg['judge_n']})")

    print(f"\nWrote {results_path}")
    print("\nResume-ready headline:")
    top_k = data["summary"]["top_k"]
    print(f"  Retrieval hit@{top_k}: {agg[f'retrieval_hit@{top_k}']:.3f} | "
          f"MRR: {agg['retrieval_mrr']:.3f} | "
          f"Judge correctness: {agg['judge_correctness']:.3f} | "
          f"Judge faithfulness: {agg['judge_faithfulness']:.3f}")


if __name__ == "__main__":
    main()
