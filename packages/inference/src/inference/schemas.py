from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from task_eval.schemas import TaskTag


@dataclass
class ModelResponse:
    model_id: str
    raw_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    extracted_answer: str = ""
    latency_ms: float | None = None


@dataclass
class Query:
    id: str
    text: str
    dataset: str
    gold_answers: list[str]
    context: str | None = None
    task_tag: TaskTag | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    policy: str
    query_id: str
    dataset: str
    task_tag: TaskTag
    predicted_tag: TaskTag | None
    final_answer: str
    gold_answers: list[str]
    primary_metric: float
    models_called: list[str]
    total_input_tokens: int | None
    total_output_tokens: int | None
    wall_latency_ms: float
    routed_correctly: bool | None
    model_responses: list[ModelResponse] = field(default_factory=list)


__all__ = ["TaskTag", "Query", "PolicyResult", "ModelResponse"]
