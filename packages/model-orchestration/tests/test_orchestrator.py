from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from llm_gateway import (
    LOCAL_LAUNCH_BIND,
    LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION,
    LOCAL_LAUNCH_IMAGE,
    LOCAL_LAUNCH_MAX_MODEL_LEN,
    LOCAL_LAUNCH_PORT,
    LOCAL_LAUNCH_QUANTIZATION,
    LOCAL_LAUNCH_STARTUP_TIMEOUT,
    PromptRequest,
    PromptResponse,
    Provider,
    ProviderConfig,
    Usage,
)
from model_orchestration import (
    JSONLRecordingConfig,
    InMemoryRecorder,
    LocalVLLMPresetConfig,
    ModelOrchestrator,
    ModelSpec,
    OrchestratorConfig,
    build_default_orchestrator_config,
    build_default_local_vllm_orchestrator_config,
)

from model_orchestration.runtime import ResolvedProviderHandle

class FakeClient:
    def __init__(self, config: ProviderConfig, *, fail: bool = False) -> None:
        self.config = config
        self.fail = fail
        self.requests: list[PromptRequest] = []
        self.closed = False

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"boom:{self.config.default_model}")

        return PromptResponse(
            model=request.model or self.config.default_model or "unknown-model",
            text=f"response:{request.model or self.config.default_model}",
            latency_ms=12.5,
            request_id=f"req-{self.config.default_model}",
            finish_reason="stop",
            provider=getattr(self.config.provider, "value", str(self.config.provider)),
            usage=Usage(input_tokens=7, output_tokens=4, total_tokens=11),
            metadata={"attempt_count": 1},
            raw_response={"id": f"raw-{self.config.default_model}"},
        )

    async def close(self) -> None:
        self.closed = True


def _config(*, recording: JSONLRecordingConfig | None = None, provider=Provider.OPENAI_COMPATIBLE):
    return OrchestratorConfig(
        models=(
            ModelSpec(
                role="qa",
                model="qa-model",
                aliases=("qa",),
                provider_config=ProviderConfig(
                    provider=provider,
                    api_base="http://qa.test/v1/chat/completions" if provider != Provider.LOCAL else None,
                    default_model="qa-model",
                ),
                request_defaults={"temperature": 0.2, "provider_params": {"top_p": 0.9}},
            ),
            ModelSpec(
                role="reasoning",
                model="reasoning-model",
                aliases=("reasoning", "math", "code"),
                provider_config=ProviderConfig(
                    provider=provider,
                    api_base=(
                        "http://reasoning.test/v1/chat/completions"
                        if provider != Provider.LOCAL
                        else None
                    ),
                    default_model="reasoning-model",
                ),
            ),
            ModelSpec(
                role="general",
                model="general-model",
                aliases=("general", "fever"),
                provider_config=ProviderConfig(
                    provider=provider,
                    api_base="http://general.test/v1/chat/completions" if provider != Provider.LOCAL else None,
                    default_model="general-model",
                ),
            ),
        ),
        default_role="general",
        recording=recording,
    )


@pytest.mark.unit
def test_default_config_exposes_expected_roles_and_aliases() -> None:
    config = build_default_orchestrator_config(provider=Provider.OPENAI_COMPATIBLE)

    assert config.default_role == "general"
    assert tuple(spec.role for spec in config.models) == (
        "qa",
        "reasoning",
        "general",
        "synthesizer",
    )
    assert config.models[0].model == "google/gemma-2-9b-it"
    assert config.models[1].aliases == ("reasoning", "math", "code")
    assert config.models[2].aliases == ("general", "fever")
    assert config.models[3].aliases == ("synthesizer",)
    assert config.mode_label == "http"


@pytest.mark.unit
def test_default_local_vllm_config_uses_benchmark_aligned_params_and_distinct_ports() -> None:
    config = build_default_local_vllm_orchestrator_config()

    assert config.default_role == "general"
    assert config.mode_label == "local"
    assert tuple(spec.provider_config.provider for spec in config.models) == (
        Provider.LOCAL,
        Provider.LOCAL,
        Provider.LOCAL,
        Provider.LOCAL,
    )

    ports = [spec.provider_config.default_params[LOCAL_LAUNCH_PORT] for spec in config.models]
    assert ports == [8000, 8001, 8002, 8003]

    qa_params = config.models[0].provider_config.default_params
    assert qa_params[LOCAL_LAUNCH_IMAGE] == "vllm-openai_latest.sif"
    assert qa_params[LOCAL_LAUNCH_STARTUP_TIMEOUT] == 600.0
    assert qa_params[LOCAL_LAUNCH_MAX_MODEL_LEN] == "8192"
    assert qa_params[LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION] == 0.33
    assert qa_params[LOCAL_LAUNCH_QUANTIZATION] == "compressed-tensors"
    assert LOCAL_LAUNCH_BIND in qa_params


@pytest.mark.unit
def test_default_orchestrator_config_uses_local_vllm_preset_for_local_provider() -> None:
    config = build_default_orchestrator_config()

    assert config.mode_label == "local"
    assert tuple(spec.provider_config.provider for spec in config.models) == (
        Provider.LOCAL,
        Provider.LOCAL,
        Provider.LOCAL,
        Provider.LOCAL,
    )
    assert [spec.provider_config.default_params[LOCAL_LAUNCH_PORT] for spec in config.models] == [
        8000,
        8001,
        8002,
        8003,
    ]


@pytest.mark.asyncio
async def test_orchestrator_routes_alias_and_merges_request_defaults() -> None:
    clients: list[FakeClient] = []

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        client = FakeClient(provider_config)
        clients.append(client)
        return client

    orchestrator = ModelOrchestrator(_config(), recorder=InMemoryRecorder(), client_builder=client_builder)

    response = await orchestrator.math_client.get_response(
        PromptRequest(user_prompt="solve this", metadata={"sample_id": "ex-1"})
    )

    assert response.target == "reasoning"
    assert response.resolved_alias == "math"
    assert response.model == "reasoning-model"
    assert response.provider_mode == "http"
    assert clients[0].requests[0].model == "reasoning-model"
    assert clients[0].requests[0].metadata["sample_id"] == "ex-1"

    response = await orchestrator.qa_client.get_response(
        user_prompt="answer this",
        provider_params={"seed": 7},
    )

    qa_request = clients[1].requests[0]
    assert qa_request.model == "qa-model"
    assert qa_request.temperature == 0.2
    assert qa_request.provider_params == {"top_p": 0.9, "seed": 7}
    await orchestrator.close()


@pytest.mark.asyncio
async def test_jsonl_recorder_writes_success_records(tmp_path: Path) -> None:
    output_path = tmp_path / "records" / "calls.jsonl"

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        return FakeClient(provider_config)

    orchestrator = ModelOrchestrator(
        _config(recording=JSONLRecordingConfig(output_path=output_path)),
        client_builder=client_builder,
    )

    response = await orchestrator.general_client.get_response(user_prompt="check facts")
    await orchestrator.close()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["target"] == "general"
    assert record["response_text"] == "response:general-model"
    assert record["request_id"] == "req-general-model"
    assert response.recorder_output_path == output_path


@pytest.mark.asyncio
async def test_jsonl_recorder_writes_error_records(tmp_path: Path) -> None:
    output_path = tmp_path / "records" / "errors.jsonl"

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        return FakeClient(provider_config, fail=True)

    orchestrator = ModelOrchestrator(
        _config(recording=JSONLRecordingConfig(output_path=output_path)),
        client_builder=client_builder,
    )

    with pytest.raises(RuntimeError, match="boom:qa-model"):
        await orchestrator.qa_client.get_response(user_prompt="fail please")

    await orchestrator.close()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["target"] == "qa"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "boom:qa-model"


@pytest.mark.asyncio
async def test_local_provider_uses_runtime_resolver_and_closes_context() -> None:
    resolver_events: list[tuple[str, str, str | None]] = []
    built_configs: list[ProviderConfig] = []
    created_clients: list[FakeClient] = []

    @asynccontextmanager
    async def fake_resolver(provider_config: ProviderConfig, *, model: str):
        resolver_events.append(("enter", model, provider_config.api_base))
        resolved = replace(
            provider_config,
            api_base=f"http://localhost/{model}/v1/chat/completions",
        )
        yield ResolvedProviderHandle(provider_config=resolved, close=_noop_close)
        resolver_events.append(("exit", model, resolved.api_base))

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        built_configs.append(provider_config)
        client = FakeClient(provider_config)
        created_clients.append(client)
        return client

    async with ModelOrchestrator(
        _config(provider=Provider.LOCAL),
        recorder=InMemoryRecorder(),
        client_builder=client_builder,
        provider_config_resolver=fake_resolver,
    ) as orchestrator:
        response = await orchestrator.qa_client.get_response(user_prompt="local run")

    assert response.provider_mode == "local"
    assert resolver_events[0] == ("enter", "qa-model", None)
    assert resolver_events[1] == ("exit", "qa-model", "http://localhost/qa-model/v1/chat/completions")
    assert built_configs[0].api_base == "http://localhost/qa-model/v1/chat/completions"
    assert created_clients[0].closed is True


@pytest.mark.asyncio
async def test_default_local_vllm_roles_resolve_to_distinct_ports_and_aliases_reuse_clients() -> None:
    resolver_events: list[tuple[str, str, int]] = []
    built_configs: list[ProviderConfig] = []
    created_clients: list[FakeClient] = []

    @asynccontextmanager
    async def fake_resolver(provider_config: ProviderConfig, *, model: str):
        port = provider_config.default_params[LOCAL_LAUNCH_PORT]
        resolver_events.append(("enter", model, port))
        resolved = replace(
            provider_config,
            api_base=f"http://127.0.0.1:{port}/v1/chat/completions",
        )
        yield ResolvedProviderHandle(provider_config=resolved, close=_noop_close)
        resolver_events.append(("exit", model, port))

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        built_configs.append(provider_config)
        client = FakeClient(provider_config)
        created_clients.append(client)
        return client

    config = build_default_local_vllm_orchestrator_config(
        preset=LocalVLLMPresetConfig(base_port=8100)
    )

    async with ModelOrchestrator(
        config,
        recorder=InMemoryRecorder(),
        client_builder=client_builder,
        provider_config_resolver=fake_resolver,
    ) as orchestrator:
        await orchestrator.qa_client.get_response(user_prompt="qa")
        await orchestrator.math_client.get_response(user_prompt="math")
        await orchestrator.reasoning_client.get_response(user_prompt="reasoning")
        await orchestrator.fever_client.get_response(user_prompt="fever")

    assert resolver_events == [
        ("enter", "google/gemma-2-9b-it", 8100),
        ("enter", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 8101),
        ("enter", "Qwen/Qwen2.5-14B-Instruct", 8102),
        ("exit", "google/gemma-2-9b-it", 8100),
        ("exit", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 8101),
        ("exit", "Qwen/Qwen2.5-14B-Instruct", 8102),
    ]
    assert len(created_clients) == 3
    assert [config.api_base for config in built_configs] == [
        "http://127.0.0.1:8100/v1/chat/completions",
        "http://127.0.0.1:8101/v1/chat/completions",
        "http://127.0.0.1:8102/v1/chat/completions",
    ]
    assert all(client.closed for client in created_clients)


@pytest.mark.unit
def test_local_vllm_preset_allows_global_and_role_overrides() -> None:
    config = build_default_local_vllm_orchestrator_config(
        preset=LocalVLLMPresetConfig(
            base_port=9000,
            gpu_memory_utilization=0.5,
            provider_defaults={"temperature": 0.1},
            role_overrides={"reasoning": {LOCAL_LAUNCH_PORT: 9015}},
        )
    )

    qa_params = config.models[0].provider_config.default_params
    reasoning_params = config.models[1].provider_config.default_params
    general_params = config.models[2].provider_config.default_params

    assert qa_params["temperature"] == 0.1
    assert qa_params[LOCAL_LAUNCH_PORT] == 9000
    assert qa_params[LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION] == 0.5
    assert reasoning_params[LOCAL_LAUNCH_PORT] == 9015
    assert general_params[LOCAL_LAUNCH_PORT] == 9002


def test_sync_wrapper_returns_response() -> None:
    created_clients: list[FakeClient] = []

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        client = FakeClient(provider_config)
        created_clients.append(client)
        return client

    orchestrator = ModelOrchestrator(
        _config(),
        recorder=InMemoryRecorder(),
        client_builder=client_builder,
    )

    response = orchestrator.qa_client.get_response_sync(user_prompt="hello")

    assert response.target == "qa"
    assert response.text == "response:qa-model"
    assert created_clients[0].requests[0].user_prompt == "hello"


async def _noop_close() -> None:
    return None
