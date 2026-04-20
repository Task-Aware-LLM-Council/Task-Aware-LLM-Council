from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from task_eval.models import EvaluationCase, MetricResult, PredictionRecord


class MetricCalculator(Protocol):
    name: str

    def score(
        self,
        *,
        case: EvaluationCase,
        prediction: PredictionRecord,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> MetricResult:
        ...


class DatasetProfile(MetricCalculator, Protocol):
    name: str
    metric_names: tuple[str, ...]
    primary_metric: str
    metadata: dict[str, Any]

    def iter_cases(self) -> Iterator[EvaluationCase]:
        ...
