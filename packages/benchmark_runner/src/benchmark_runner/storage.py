from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from benchmark_runner.models import (
    AggregateMetricRow,
    BenchmarkSpec,
    BenchmarkSuiteResult,
    ScoreRecord,
    ScoreSummary,
)


def ensure_suite_directory(output_root: Path, suite_id: str) -> Path:
    suite_dir = output_root / suite_id
    (suite_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (suite_dir / "scores").mkdir(parents=True, exist_ok=True)
    (suite_dir / "summaries").mkdir(parents=True, exist_ok=True)
    return suite_dir


def pair_run_name(dataset_name: str, model: str) -> str:
    return f"{_slugify(dataset_name)}__{_slugify(model)}"


def score_path(suite_dir: Path, dataset_name: str, model: str) -> Path:
    return suite_dir / "scores" / f"{pair_run_name(dataset_name, model)}.jsonl"


def summary_path(suite_dir: Path, dataset_name: str, model: str) -> Path:
    return suite_dir / "summaries" / f"{pair_run_name(dataset_name, model)}.json"


def aggregate_summary_path(suite_dir: Path) -> Path:
    return suite_dir / "suite_metrics.json"


def append_score_record(path: Path, record: ScoreRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(record), sort_keys=True))
        handle.write("\n")


def write_summary(path: Path, summary: ScoreSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_aggregate_summary(path: Path, rows: list[AggregateMetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(rows), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_manifest(
    suite_dir: Path,
    *,
    suite_id: str,
    spec: BenchmarkSpec,
    datasets: list[dict[str, Any]],
    result: BenchmarkSuiteResult | None = None,
) -> Path:
    path = suite_dir / "manifest.json"
    payload = {
        "suite_id": suite_id,
        "spec": _to_jsonable(spec),
        "datasets": _to_jsonable(datasets),
    }
    if result is not None:
        payload["summary"] = _to_jsonable(result)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return path


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("_") or "value"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
