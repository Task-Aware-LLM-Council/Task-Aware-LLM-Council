from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_gateway import ProviderConfig
from task_eval import EvaluationCase

BenchmarkCase = EvaluationCase


@dataclass(slots=True, frozen=True)
class DatasetRunConfig:
    name: str
    split: str = "validation"
    profile_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AggregateMetricRow:
    suite_id: str
    dataset_name: str
    model: str
    primary_metric: str
    primary_metric_value: float | None
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    scored_examples: int = 0
    failed_examples: int = 0
    total_examples: int = 0
    summary_path: Path | None = None


@dataclass(slots=True, frozen=True)
class BenchmarkSpec:
    provider_config: ProviderConfig
    models: tuple[str, ...]
    output_root: Path
    suite_name: str | None = None
    prompt_version: str = "v1"
    batch_size: int = 32
    delay_between_requests: int = 5
    max_concurrency: int = 5
    max_examples_per_dataset: int | None = None
    continue_on_error: bool = True
    skip_existing_predictions: bool = True
    save_raw_response: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ScoreRecord:
    suite_id: str
    dataset_name: str
    model: str
    example_id: str
    status: str
    prediction_status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    example_metadata: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)
    metric_name: str | None = None
    metric_metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class ScoreSummary:
    suite_id: str
    dataset_name: str
    model: str
    metric_name: str
    total_examples: int
    scored_examples: int
    failed_examples: int
    skipped_examples: int
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BenchmarkSuiteResult:
    suite_id: str
    output_dir: Path
    manifest_path: Path
    score_files: tuple[Path, ...]
    summary_files: tuple[Path, ...]
    aggregate_summary_path: Path
    total_pairs: int
    total_examples: int
    scored_examples: int
    failed_examples: int


def default_suite_id() -> str:
    return datetime.now(timezone.utc).strftime("suite_%Y%m%dT%H%M%SZ")
