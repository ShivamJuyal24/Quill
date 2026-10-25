"""Resumable representative-subset Ragas scoring with a separate output file."""

import argparse
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVALS))
load_dotenv(ROOT / "server" / ".env")

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

from eval_utils import (
    RAGAS_METRICS,
    answer_fingerprint,
    atomic_json_write,
    exception_contains,
    load_json_strict,
    merge_ragas_scores,
    numeric_mean,
    ragas_dataset_rows,
    result_rows,
    scores_complete,
    valid_score_count,
    validate_output_path,
    validate_answer_rows,
)
from key_pool import AllKeysExhausted, GroqKeyPool, GroqPoolChatModel


def parse_args():
    parser = argparse.ArgumentParser(description="Score a representative Ragas subset")
    parser.add_argument("--n", type=int, default=10, help="number of questions in representative sample")
    parser.add_argument("--input", default=str(EVALS / "ragas_results.json"))
    parser.add_argument("--output", default=str(EVALS / "ragas_subset_metrics_results.json"))
    parser.add_argument("--max-wait", type=float, default=90.0)
    return parser.parse_args()


def select_subset(rows, n):
    if n < 1:
        raise ValueError("--n must be at least 1")
    random.seed(42)
    groups = {}
    for row in rows:
        groups.setdefault((row.get("type"), row.get("difficulty")), []).append(row)
    selected = []
    keys = sorted(groups, key=lambda key: (str(key[0]), str(key[1])))
    base, remainder = divmod(n, max(1, len(keys)))
    for index, key in enumerate(keys):
        count = base + (1 if index < remainder else 0)
        selected.extend(random.sample(groups[key], min(count, len(groups[key]))))
    # Fill any shortfall caused by undersized groups, without duplicates.
    selected_ids = {str(row["id"]) for row in selected}
    remaining = [row for row in rows if str(row["id"]) not in selected_ids]
    if len(selected) < min(n, len(rows)):
        selected.extend(random.sample(remaining, min(n, len(rows)) - len(selected)))
    return selected[:n]


def main():
    args = parse_args()
    input_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    validate_output_path(output_path)
    if input_path == output_path:
        raise SystemExit("Refusing to overwrite the Ragas input file; choose a separate --output path")
    data = load_json_strict(input_path)
    if not isinstance(data, dict) or "per_question" not in data or "summary" not in data:
        raise SystemExit(f"{input_path.name} must contain summary and per_question")
    per_question = validate_answer_rows(data["per_question"], input_path.name)
    if not per_question:
        raise SystemExit("No answer-bearing question records to score")
    summary_in = data["summary"]
    if not summary_in.get("embedding_model") or not summary_in.get("groq_model"):
        raise SystemExit("Input summary must identify embedding_model and groq_model")
    source_fingerprint = answer_fingerprint(per_question, summary_in)

    selected = select_subset(per_question, args.n)
    if output_path.exists():
        previous = load_json_strict(output_path)
        if previous.get("source_input") != str(input_path) or previous.get("source_fingerprint") != source_fingerprint:
            raise SystemExit(f"Existing output {output_path.name} belongs to different input data; choose another output")
        previous_rows = validate_answer_rows(previous.get("per_question", []), output_path.name)
        prior = {str(row["id"]): row for row in previous_rows}
        per_question = [
            {**original, **{m: prior[str(original["id"])][m] for m in RAGAS_METRICS
                            if str(original["id"]) in prior and prior[str(original["id"])].get(m) is not None}}
            for original in per_question
        ]
        by_id = {str(row["id"]): row for row in per_question}
        selected = [by_id[str(row["id"])] for row in selected]

    pool = GroqKeyPool(max_wait=args.max_wait)
    ragas_llm = LangchainLLMWrapper(
        GroqPoolChatModel(pool=pool, model=summary_in["groq_model"], temperature=0.2)
    )
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=summary_in["embedding_model"], google_api_key=os.getenv("GEMINI_API_KEY")
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]
    output = {
        **data,
        "source_input": str(input_path),
        "source_fingerprint": source_fingerprint,
        "per_question": per_question,
    }

    def save():
        selected_ids = {str(row["id"]) for row in selected}
        scored_rows = [row for row in selected if scores_complete(row)]
        aggregate = dict(summary_in.get("aggregate", {}))
        for metric in RAGAS_METRICS:
            vals = [row.get(metric) for row in selected if str(row["id"]) in selected_ids]
            aggregate[f"ragas_{metric}"] = numeric_mean(vals)
            aggregate[f"ragas_{metric}_n"] = valid_score_count(vals)
        output["per_question"] = per_question
        output["source_fingerprint"] = source_fingerprint
        output["summary"] = {
            **summary_in,
            "ragas_scored_n": len(scored_rows),
            "ragas_subset_n": len(selected),
            "aggregate": aggregate,
        }
        atomic_json_write(output_path, output)

    save()
    try:
        for position, row in enumerate(selected, 1):
            if scores_complete(row):
                continue
            print(f"[{position}/{len(selected)}] Ragas question {row['id']}", flush=True)
            dataset = Dataset.from_list(ragas_dataset_rows([row]))
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
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
            by_id = {str(item["id"]): item for item in per_question}
            selected = [by_id[str(item["id"])] for item in selected]
            save()
    except Exception as error:
        save()
        if exception_contains(error, AllKeysExhausted):
            print(f"All daily Groq keys are exhausted. Progress saved to {output_path}.", flush=True)
            return
        raise

    print(f"Subset Ragas scores saved to {output_path}; selected {len(selected)} questions.")


if __name__ == "__main__":
    main()
