from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_gateway import PromptRequest, PromptResponse, ProviderConfig


ModelRole = str


@dataclass(slots=True, frozen=True)
class JSONLRecordingConfig:
    output_path: Path


@dataclass(slots=True, frozen=True)
class LocalVLLMPresetConfig:
    base_port: int = 8000
    image: str = "/scratch1/sureshag/Task-Aware-LLM-Council/vllm-openai_latest.sif"
    bind: str | None = None
    startup_timeout_seconds: float = 1800.0
    max_model_len: str = "8192"
    gpu_memory_utilization: float = 0.30       
    quantization: str = "compressed-tensors"
    use_gpu: bool = True
    client_host: str | None = None
    server_host: str | None = None
    container_cache_dir: str | None = None
    executable: str | None = None
    poll_interval_seconds: float | None = None
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    dtype: str | None = None
    load_format: str | None = None
    cpu_offload_gb: float | None = None
    provider_defaults: dict[str, Any] = field(default_factory=dict)
    role_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ModelSpec:
    role: ModelRole
    model: str
    provider_config: ProviderConfig
    aliases: tuple[str, ...] = ()
    description: str | None = None
    request_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrchestratorConfig:
    models: tuple[ModelSpec, ...]
    default_role: ModelRole
    recording: JSONLRecordingConfig | None = None
    mode_label: str | None = None
    sequential_local_gpu: bool = False


@dataclass(slots=True, frozen=True)
class OrchestratorRequest:
    prompt: PromptRequest
    target: ModelRole | None = None


@dataclass(slots=True, frozen=True)
class OrchestratorCallRecord:
    event_id: str
    started_at: str
    completed_at: str | None
    target: ModelRole
    resolved_alias: str
    model: str
    provider: str
    provider_mode: str
    request: dict[str, Any]
    request_metadata: dict[str, Any]
    response_text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class OrchestratorResponse:
    target: ModelRole
    resolved_alias: str
    model: str
    provider: str
    provider_mode: str
    request: PromptRequest
    prompt_response: PromptResponse
    started_at: str
    completed_at: str
    call_record: OrchestratorCallRecord
    recorder_output_path: Path | None = None

    @property
    def text(self) -> str:
        return self.prompt_response.text

    @property
    def usage(self):  # pragma: no cover - simple delegation
        return self.prompt_response.usage

    @property
    def latency_ms(self) -> float | None:
        return self.prompt_response.latency_ms
