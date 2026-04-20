from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarking_pipeline import BenchmarkExample


PredictionRecord = dict[str, Any]


@dataclass(slots=True, frozen=True)
class EvaluationCase:
    example: BenchmarkExample
    reference: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MetricResult:
    values: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
