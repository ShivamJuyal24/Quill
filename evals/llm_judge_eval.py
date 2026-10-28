"""Resumable LLM-as-judge evaluation for Quill RAG answers (Groq key pool).

Reads evals/ragas_results.json (per_question rows with `response`,
`retrieved_contexts`, `reference`; produced by ragas_eval.py) and scores each
answer with Groq for correctness / faithfulness / relevancy (0-1).

  * Never overwrites the input file; writes evals/llm_judge_results.json
  * Rotates through the keys in GROQ_API_KEYS (see key_pool.py)
  * When every key is out of daily quota it saves and stops; run it again
    later and already-scored questions/metrics are skipped
  * Failed calls stay unscored (None) and are retried next run; n is reported

Usage:
    python evals/llm_judge_eval.py
    python evals/llm_judge_eval.py --limit 10
    python evals/llm_judge_eval.py --model llama-3.1-8b-instant
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(ROOT / "server" / ".env")

from langchain_core.messages import HumanMessage, SystemMessage

from eval_utils import atomic_json_write, load_json_strict, validate_answer_rows, validate_output_path
from key_pool import GroqKeyPool, AllKeysExhausted


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


METRICS = [
    ("judge_correctness", CORRECTNESS_PROMPT),
    ("judge_faithfulness", FAITHFULNESS_PROMPT),
    ("judge_relevancy", RELEVANCY_PROMPT),
]


def judge_score(pool, model, template, **kwargs):
    """Ask the judge for a 0-1 score. Returns float, or None on a non-quota failure.
    AllKeysExhausted propagates so the caller can save and stop."""
    prompt = template.format(**kwargs)
    messages = [
        SystemMessage(content="You are a precise scoring evaluator. Respond with only a decimal number between 0 and 1."),
        HumanMessage(content=prompt),
    ]
    try:
        resp = pool.invoke(messages, model=model, temperature=0.0)
        text = (resp.content or "").strip()
        m = re.search(r"\d*\.?\d+", text)
        if m:
            return max(0.0, min(1.0, float(m.group())))
    except AllKeysExhausted:
        raise
    except Exception as e:
        print(f"  [judge error: {str(e)[:200]}]", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge for Quill RAG answers")
    ap.add_argument("--input", default=str(Path(__file__).parent / "ragas_results.json"))
    ap.add_argument("--output", default=str(Path(__file__).parent / "llm_judge_results.json"))
    ap.add_argument("--model", default=None, help="judge model (default: model recorded in the results file)")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N questions")
    ap.add_argument("--max-wait", type=float, default=90.0,
                    help="retained for compatibility; temporary cooldowns never count as daily exhaustion")
    args = ap.parse_args()

    src_path = Path(args.input)
    out_path = Path(args.output)
    validate_output_path(out_path)
    src = load_json_strict(src_path)
    per_question = validate_answer_rows(src["per_question"], src_path.name)
    retrieval_rows = src.get("retrieval_rows", [])
    summary_in = src.get("summary", {})

    if not per_question:
        sys.exit(
            f"{src_path.name} has no generated answers (per_question rows lack 'response').\n"
            "Generate them first:  python evals/ragas_eval.py --skip-ragas"
        )
    if args.limit:
        per_question = per_question[: args.limit]

    model = args.model or summary_in.get("groq_model") or os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    signature_data = [
        {
            "id": str(row["id"]),
            "question": row["question"],
            "reference": row.get("reference", ""),
            "response": row["response"],
            "retrieved_contexts": row.get("retrieved_contexts", []),
        }
        for row in per_question
    ]
    input_fingerprint = hashlib.sha256(
        json.dumps({"model": model, "rows": signature_data}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    pool = GroqKeyPool(max_wait=args.max_wait)
    print(f"Judge model: {model}   Questions: {len(per_question)}")

    # ---- resume ----
    scored = {}
    if out_path.exists():
        try:
            prev = load_json_strict(out_path)
            if prev.get("input_fingerprint") != input_fingerprint:
                raise SystemExit(
                    f"Existing judge output {out_path.name} belongs to different answers/model; "
                    "choose a new --output path to avoid reusing stale scores."
                )
            scored = {r["id"]: r for r in prev.get("per_question", [])}
        except SystemExit:
            raise
        except Exception:
            raise SystemExit(f"Existing judge output {out_path.name} is invalid; choose a new --output path")

    def aggregate(rows):
        agg = {}
        for key, _ in METRICS:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            agg[key] = mean(vals)
            agg[key + "_n"] = len(vals)
        return agg

    def save():
        rows = [scored[q["id"]] for q in per_question if q["id"] in scored]
        agg = aggregate(rows)
        out = {
            "summary": {
                "judge_model": model,
                "embedding_model": summary_in.get("embedding_model"),
                "top_k": summary_in.get("top_k"),
                "num_questions": len(per_question),
                "judged_rows": len(rows),
                "aggregate": agg,
                "input_fingerprint": input_fingerprint,
            },
            "input_fingerprint": input_fingerprint,
            "per_question": rows,
        }
        atomic_json_write(out_path, out)
        return agg

    # ---- judge ----
    stopped = False
    try:
        for i, row in enumerate(per_question, 1):
            qid = row["id"]
            have = scored.get(qid, {"id": qid})
            todo = [(k, t) for k, t in METRICS if have.get(k) is None]
            if not todo:
                continue

            print(f"[{i}/{len(per_question)}] {qid}", end=" ", flush=True)
            kwargs = dict(
                question=row["question"],
                reference=row.get("reference", ""),
                candidate=row["response"],
                context="\n".join(row.get("retrieved_contexts", [])),
            )
            try:
                for key, template in todo:
                    have[key] = judge_score(pool, model, template, **kwargs)
            finally:
                scored[qid] = have   # keep whatever was scored before a stop
                save()
            print(" ".join(f"{k.replace('judge_', '')}={have.get(k)}" for k, _ in METRICS), flush=True)
    except AllKeysExhausted as e:
        stopped = True
        print(f"\nAll Groq keys are out of quota ({e}).")
        print(f"Progress saved to {out_path.name}. Run again after the quotas reset, or add more keys.")

    agg = save()

    # ---- retrieval aggregate (computed directly, no LLM) ----
    topks = set()
    for r in retrieval_rows:
        for key in r:
            if key.startswith("hit@"):
                try:
                    topks.add(int(key.split("@")[1]))
                except (ValueError, IndexError):
                    pass
    topks = sorted(topks) or [1, 3, 4, 5, 8, 10]
    print("\n========== RETRIEVAL ==========")
    for k in topks:
        hit_key = f"hit@{k}"
        recall_key = f"recall@{k}"
        hit_vals = [r.get(hit_key, 0.0) for r in retrieval_rows]
        recall_vals = [r.get(recall_key, 0.0) for r in retrieval_rows]
        print(f"  hit@{k:<2d}: {mean(hit_vals):.3f}   "
              f"recall@{k:<2d}: {mean(recall_vals):.3f}")
    if retrieval_rows:
        print(f"  MRR: {mean([r.get('mrr', 0.0) for r in retrieval_rows]):.3f}")

    print("\n========== LLM-AS-JUDGE ==========")
    for key, _ in METRICS:
        print(f"  {key.replace('judge_', ''):<13}: {agg[key]:.3f}  (n={agg[key + '_n']}/{len(per_question)})")
    if stopped or any(agg[k + "_n"] < len(per_question) for k, _ in METRICS):
        print("\n  INCOMPLETE: some questions are not scored yet. Re-run to finish before quoting these numbers.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
