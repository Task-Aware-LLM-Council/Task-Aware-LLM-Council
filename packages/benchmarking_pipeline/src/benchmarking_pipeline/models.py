from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_gateway import Message, ProviderConfig


@dataclass(slots=True, frozen=True)
class BenchmarkExample:
    example_id: str
    dataset_name: str
    question: str | None = None
    context: str | None = None
    system_prompt: str | None = None
    messages: tuple[Message, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BenchmarkDataset:
    name: str
    examples: tuple[BenchmarkExample, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(slots=True, frozen=True)
class BenchmarkRunConfig:
    provider_config: ProviderConfig
    models: tuple[str, ...]
    output_root: Path
    run_name: str | None = None
    prompt_version: str = "v1"
    max_concurrency: int = 5
    continue_on_error: bool = True
    skip_existing: bool = True
    save_raw_response: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BenchmarkPrediction:
    run_id: str
    dataset_name: str
    model: str
    example_id: str
    status: str
    prompt_version: str
    response_text: str | None = None
    latency_ms: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    provider: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    example_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class BenchmarkRunResult:
    run_id: str
    output_dir: Path
    manifest_path: Path
    created_at: str
    total_examples: int
    attempted_examples: int
    completed_examples: int
    failed_examples: int
    skipped_existing: int
    prediction_files: tuple[Path, ...]


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
