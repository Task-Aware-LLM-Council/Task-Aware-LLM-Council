from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class UnifiedInput:
    """All data a policy needs from the benchmark pipeline."""

    example_id: str
    dataset_name: str
    question: str | None = None
    context: str | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PolicyEvaluationResult:
    """Standard return type for every council policy."""

    example_id: str
    dataset_name: str
    output: str          # final answer scored against ground truth
    status: str          # "success" | "error"
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
