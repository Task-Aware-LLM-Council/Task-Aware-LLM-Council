from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_gateway import PromptRequest
from llm_gateway.models import Usage


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


@dataclass(slots=True, frozen=True)
class P2RunConfig:
    output_root: Path
    dataset_name: str = "task-aware-llm-council/router_dataset"
    dataset_alias: str = "router_dataset"
    split: str = "validation"
    max_examples: int | None = None
    batch_size: int = 64
    max_concurrency: int = 5
    run_name: str | None = None
    provider: str = "openai-compatible"
    api_base: str | None = None
    api_key_env: str | None = None
    local_launch_port: int = 8000
    gpu_utilization: float = 0.30
    synth_max_concurrency: int | None = None
    synth_model: str | None = None


@dataclass(slots=True, frozen=True)
class P2SystemMetrics:
    answer_usage: Usage = field(default_factory=Usage)
    vote_usage: Usage = field(default_factory=Usage)
    specialist_usage: Usage = field(default_factory=Usage)
    answer_latency_ms_total: float | None = None
    vote_latency_ms_total: float | None = None
    specialist_latency_ms_total: float | None = None
    wall_clock_latency_ms: float | None = None


@dataclass(slots=True, frozen=True)
class P2RoleResult:
    role: str
    model: str
    provider: str | None = None
    provider_mode: str | None = None
    text: str = ""
    latency_ms: float | None = None
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class P2CouncilDecision:
    example_id: str
    dataset_name: str
    request: PromptRequest
    winning_label: str
    winning_role: str
    winning_model: str
    winning_answer: str
    votes: dict[str, str] = field(default_factory=dict)
    role_results: dict[str, P2RoleResult] = field(default_factory=dict)
    aggregated_usage: Usage = field(default_factory=Usage)
    system_metrics: P2SystemMetrics = field(default_factory=P2SystemMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class P2SynthesizedRecord:
    example_id: str
    dataset_name: str
    source_dataset: str
    synthesized_text: str
    metric_name: str
    request: PromptRequest
    winning_answer: str
    winning_role: str
    winning_model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None


@dataclass(slots=True, frozen=True)
class P2ScoreRecord:
    example_id: str
    dataset_name: str
    source_dataset: str
    metric_name: str
    primary_metric: str
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class P2RunResult:
    run_id: str
    output_dir: Path
    manifest_path: Path
    prediction_file: Path
    score_file: Path
    summary_files: tuple[Path, ...]
    aggregate_summary_path: Path
    total_examples: int
    completed_examples: int
    failed_examples: int
    combined_metric: float | None = None
    dataset_scores: dict[str, float] = field(default_factory=dict)
