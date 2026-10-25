"""Resumable Ragas scoring; reads answer results and writes a separate artifact."""

import argparse
import os
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
    parser = argparse.ArgumentParser(description="Resume Ragas scoring from generated answer records")
    parser.add_argument("--input", default=str(EVALS / "ragas_results.json"))
    parser.add_argument("--output", default=str(EVALS / "ragas_metrics_results.json"))
    parser.add_argument("--max-wait", type=float, default=90.0,
                        help="legacy pool setting; long temporary waits remain cooldowns")
    return parser.parse_args()


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
    required_summary = ("embedding_model", "groq_model")
    missing = [field for field in required_summary if not summary_in.get(field)]
    if missing:
        raise SystemExit(f"Input summary is missing required fields: {', '.join(missing)}")
    source_fingerprint = answer_fingerprint(per_question, summary_in)

    # Resume from the output artifact only when it belongs to the same input.
    if output_path.exists():
        previous = load_json_strict(output_path)
        if previous.get("source_input") != str(input_path) or previous.get("source_fingerprint") != source_fingerprint:
            raise SystemExit(f"Existing output {output_path.name} belongs to different input data; choose another output")
        previous_rows = validate_answer_rows(previous.get("per_question", []), output_path.name)
        old_by_id = {str(row["id"]): row for row in previous_rows}
        # Keep original input fields authoritative while carrying metric fields forward.
        merged = []
        for original in per_question:
            row = dict(original)
            old = old_by_id.get(str(original["id"]), {})
            for metric in RAGAS_METRICS:
                if old.get(metric) is not None:
                    row[metric] = old[metric]
            merged.append(row)
        per_question = merged

    pool = GroqKeyPool(max_wait=args.max_wait)
    chat_model = GroqPoolChatModel(pool=pool, model=summary_in["groq_model"], temperature=0.2)
    ragas_llm = LangchainLLMWrapper(chat_model)
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]

    output = {
        **data,
        "source_input": str(input_path),
        "source_fingerprint": source_fingerprint,
        "per_question": per_question,
        "summary": {**summary_in, "ragas_scored_n": 0, "aggregate": dict(summary_in.get("aggregate", {}))},
    }

    def save():
        scored = [row for row in per_question if scores_complete(row)]
        aggregate = dict(summary_in.get("aggregate", {}))
        for metric in RAGAS_METRICS:
            aggregate[f"ragas_{metric}"] = numeric_mean([row.get(metric) for row in per_question])
            aggregate[f"ragas_{metric}_n"] = valid_score_count([row.get(metric) for row in per_question])
        output["per_question"] = per_question
        output["source_fingerprint"] = source_fingerprint
        output["summary"] = {
            **summary_in,
            "ragas_scored_n": len(scored),
            "aggregate": aggregate,
        }
        atomic_json_write(output_path, output)

    save()
    exhausted = False
    for position, row in enumerate(per_question, 1):
        if scores_complete(row):
            continue
        print(f"[{position}/{len(per_question)}] Ragas question {row['id']}", flush=True)
        try:
            ragas_embeddings = LangchainEmbeddingsWrapper(
                GoogleGenerativeAIEmbeddings(
                    model=summary_in["embedding_model"], google_api_key=os.getenv("GEMINI_API_KEY")
                )
            )
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
            save()
        except AllKeysExhausted:
            save()
            print(f"All daily Groq keys are exhausted. Progress saved to {output_path}.", flush=True)
            exhausted = True
            break
        except Exception as error:
            save()
            print(f"Ragas question {row['id']} skipped ({error}); will retry on next run.", flush=True)
            continue

    if exhausted:
        return
    if not all(scores_complete(row) for row in per_question):
        print(f"Progress saved to {output_path}. Run again to score remaining questions.", flush=True)
        return
    print(f"Ragas scores saved to {output_path}")


if __name__ == "__main__":
    main()
