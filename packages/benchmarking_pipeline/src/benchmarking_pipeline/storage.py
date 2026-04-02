from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from benchmarking_pipeline.models import (
    BenchmarkDataset,
    BenchmarkPrediction,
    BenchmarkRunConfig,
    BenchmarkRunResult,
)


def ensure_run_directory(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    return run_dir


def prediction_path(run_dir: Path, dataset_name: str, model: str) -> Path:
    filename = f"{_slugify(dataset_name)}__{_slugify(model)}.jsonl"
    return run_dir / "predictions" / filename


def append_prediction(path: Path, prediction: BenchmarkPrediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(prediction), sort_keys=True))
        handle.write("\n")


def load_recorded_example_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    example_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            example_id = payload.get("example_id")
            if isinstance(example_id, str):
                example_ids.add(example_id)
    return example_ids


def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    config: BenchmarkRunConfig,
    datasets: tuple[BenchmarkDataset, ...],
    result: BenchmarkRunResult | None = None,
) -> Path:
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "config": _to_jsonable(config),
        "datasets": [
            {
                "name": dataset.name,
                "example_count": len(dataset.examples),
                "metadata": _to_jsonable(dataset.metadata),
            }
            for dataset in datasets
        ],
    }
    if result is not None:
        manifest["summary"] = _to_jsonable(result)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest_path


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
