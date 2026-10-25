"""Shared safe JSON and evaluation-result helpers."""

import hashlib
import json
import math
import numbers
import os
import tempfile
from pathlib import Path


RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)
ANSWER_FIELDS = ("id", "question", "response", "retrieved_contexts", "reference", "reference_contexts")


def atomic_json_write(path, data, allow_protected=False):
    """Write JSON to a sibling temp file, then atomically replace the target."""
    path = Path(path)
    validate_output_path(path, allow_protected=allow_protected)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def validate_output_path(path, allow_protected=False):
    path = Path(path)
    lowered = path.name.lower()
    if ("backup" in lowered or "before-restore" in lowered) or (lowered == "ragas_results.json" and not allow_protected):
        raise ValueError(f"Refusing to write protected evaluation input/backup: {path}")
    return path
    return path


def load_json_strict(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot safely read JSON file {path}: {exc}") from exc


def dataset_fingerprint(questions, chunks=None):
    canonical = json.dumps(
        {
            "questions": [{"id": str(row["id"]), "question": row["question"]} for row in questions],
            "chunks": [
                {"chunk_id": str(row["chunk_id"]), "text": row["text"]}
                for row in (chunks or [])
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def answer_fingerprint(rows, summary=None):
    payload = {
        "model": (summary or {}).get("groq_model"),
        "embedding_model": (summary or {}).get("embedding_model"),
        "rows": [
            {
                "id": str(row["id"]),
                "question": row["question"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["reference"],
                "reference_contexts": row["reference_contexts"],
            }
            for row in rows
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_answer_rows(rows, source_name="evaluation input"):
    if not isinstance(rows, list):
        raise ValueError(f"{source_name}: per_question must be a list")
    seen = set()
    for index, row in enumerate(rows):
        missing = [field for field in ANSWER_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{source_name}: row {index} missing required fields: {', '.join(missing)}")
        key = str(row["id"])
        if key in seen:
            raise ValueError(f"{source_name}: duplicate question ID {row['id']!r}")
        seen.add(key)
        if not isinstance(row["retrieved_contexts"], list) or not isinstance(row["reference_contexts"], list):
            raise ValueError(f"{source_name}: contexts for ID {row['id']!r} must be lists")
    return rows


def merge_ragas_scores(rows, score_records, metrics=RAGAS_METRICS):
    """Merge score records by stable ID, preserving source fields and old scores."""
    copied = [dict(row) for row in rows]
    by_id = {str(row["id"]): row for row in copied}
    for score in score_records:
        raw_id = score.get("eval_id", score.get("id"))
        if raw_id is None:
            raise ValueError("Ragas result row is missing eval_id/id; refusing text-based matching")
        key = str(raw_id)
        if key not in by_id:
            raise ValueError(f"Ragas returned unknown question ID {raw_id!r}")
        for metric in metrics:
            value = score.get(metric)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                by_id[key][metric] = value
    return copied


def ragas_dataset_rows(rows):
    """Add a stable evaluation ID column without dropping any source fields."""
    return [{**row, "eval_id": str(row["id"])} for row in rows]


def result_rows(result, source_rows=None):
    frame = result.to_pandas() if hasattr(result, "to_pandas") else result
    if hasattr(frame, "to_dict"):
        records = frame.to_dict(orient="records")
    elif isinstance(frame, dict):
        records = [frame]
    else:
        records = list(frame)
    if source_rows is not None:
        if len(records) != len(source_rows):
            raise ValueError(f"Ragas returned {len(records)} rows for {len(source_rows)} input rows")
        for source, record in zip(source_rows, records):
            returned_question = record.get("user_input", record.get("question"))
            if returned_question is not None and returned_question != source["question"]:
                raise ValueError(f"Ragas result order/question mismatch for ID {source['id']!r}")
            record["eval_id"] = str(source["id"])
    return records


def numeric_mean(values):
    vals = [float(v) for v in values if isinstance(v, numbers.Real) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def valid_score_count(values):
    return sum(isinstance(value, numbers.Real) and math.isfinite(float(value)) for value in values)


def scores_complete(row, metrics=RAGAS_METRICS):
    return all(
        isinstance(row.get(metric), numbers.Real) and math.isfinite(float(row[metric]))
        for metric in metrics
    )


def exception_contains(error, exception_type):
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        if isinstance(current, exception_type):
            return True
        seen.add(id(current))
        nested = getattr(current, "exceptions", None)
        if nested and any(exception_contains(item, exception_type) for item in nested):
            return True
        current = current.__cause__ or current.__context__
    return False
