from __future__ import annotations

import signal

import pytest

from llm_gateway import Provider, ProviderConfig
from llm_gateway.vllm_runtime import (
    LOCAL_LAUNCH_BIND,
    LOCAL_LAUNCH_CLIENT_HOST,
    LOCAL_LAUNCH_CONTAINER_CACHE_DIR,
    LOCAL_LAUNCH_ENV,
    LOCAL_LAUNCH_EXTRA_ARGS,
    LOCAL_LAUNCH_IMAGE,
    LOCAL_LAUNCH_PORT,
    LOCAL_LAUNCH_USE_GPU,
    VLLMRuntime,
    VLLMRuntimeConfig,
    build_vllm_runtime_config,
    normalize_local_provider_config,
)


def test_build_vllm_runtime_config_from_provider_defaults() -> None:
    config = ProviderConfig(
        provider=Provider.LOCAL,
        api_base="http://unused",
        default_model="model-a",
        default_params={
            LOCAL_LAUNCH_IMAGE: "vllm-openai.sif",
            LOCAL_LAUNCH_USE_GPU: False,
            LOCAL_LAUNCH_BIND: "/scratch/cache",
            LOCAL_LAUNCH_CONTAINER_CACHE_DIR: "/root/.cache/hf",
            LOCAL_LAUNCH_CLIENT_HOST: "localhost",
            LOCAL_LAUNCH_PORT: 9000,
            LOCAL_LAUNCH_EXTRA_ARGS: ("--dtype", "auto"),
            LOCAL_LAUNCH_ENV: {"HF_TOKEN": "abc"},
            "temperature": 0.0,
        },
    )

    launch = build_vllm_runtime_config(config)

    assert launch is not None
    assert launch.image == "vllm-openai.sif"
    assert launch.use_gpu is False
    assert launch.bind == "/scratch/cache"
    assert launch.container_cache_dir == "/root/.cache/hf"
    assert launch.client_host == "localhost"
    assert launch.port == 9000
    assert launch.extra_args == ("--dtype", "auto")
    assert launch.env == {"HF_TOKEN": "abc"}


def test_normalize_local_provider_config_strips_launch_keys() -> None:
    config = ProviderConfig(
        provider=Provider.LOCAL,
        api_base="http://127.0.0.1:8000/v1/chat/completions",
        default_model="model-a",
        default_params={
            LOCAL_LAUNCH_IMAGE: "vllm-openai.sif",
            "temperature": 0.0,
        },
    )

    normalized = normalize_local_provider_config(config)

    assert LOCAL_LAUNCH_IMAGE not in normalized.default_params
    assert normalized.default_params["temperature"] == 0.0


def test_vllm_runtime_builds_expected_command() -> None:
    runtime = VLLMRuntime(
        VLLMRuntimeConfig(
            image="vllm-openai.sif",
            bind="/scratch/cache",
            container_cache_dir="/root/.cache/huggingface",
            port=8123,
            extra_args=("--dtype", "auto"),
        )
    )

    command = runtime._build_command("facebook/opt-125m")

    assert command == (
        "apptainer",
        "run",
        "--nv",
        "--cleanenv",
        "--bind",
        "/scratch/cache:/root/.cache/huggingface",
        "--env",
        "HF_HOME=/root/.cache/huggingface",
        "vllm-openai.sif",
        "--model",
        "facebook/opt-125m",
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        "8123",
        "--dtype",
        "auto",
    )


@pytest.mark.asyncio
async def test_vllm_runtime_start_returns_api_base(monkeypatch) -> None:
    events: list[object] = []

    class FakeProcess:
        returncode = None
        pid = 123

        def kill(self) -> None:
            events.append("kill")

        async def wait(self) -> int:
            events.append("wait")
            self.returncode = 0
            return 0

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        events.append(cmd)
        return FakeProcess()

    async def fake_wait_until_ready(self) -> None:
        events.append("ready")

    monkeypatch.setattr(
        "llm_gateway.vllm_runtime.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "llm_gateway.vllm_runtime.VLLMRuntime._wait_until_ready",
        fake_wait_until_ready,
    )
    monkeypatch.setattr("llm_gateway.vllm_runtime.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("llm_gateway.vllm_runtime.os.killpg", lambda pid, sig: events.append(("killpg", pid, sig)))

    runtime = VLLMRuntime(VLLMRuntimeConfig(image="vllm-openai.sif"))

    api_base = await runtime.start("facebook/opt-125m")
    await runtime.stop()

    assert api_base == "http://127.0.0.1:8000/v1/chat/completions"
    assert events[1] == "ready"
    assert any(event == ("killpg", 123, signal.SIGTERM) for event in events)


@pytest.mark.asyncio
async def test_vllm_runtime_resolves_provider_config(monkeypatch) -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(image="vllm-openai.sif"))
    base_config = ProviderConfig(
        provider=Provider.LOCAL,
        default_params={LOCAL_LAUNCH_IMAGE: "vllm-openai.sif", "temperature": 0.1},
    )

    async def fake_start(model: str) -> str:
        assert model == "model-a"
        return "http://127.0.0.1:8001/v1/chat/completions"

    monkeypatch.setattr(runtime, "start", fake_start)

    resolved = await runtime.resolved_provider_config(base_config, model="model-a")

    assert resolved.api_base == "http://127.0.0.1:8001/v1/chat/completions"
    assert resolved.default_params == {"temperature": 0.1}


@pytest.mark.asyncio
async def test_vllm_runtime_rejects_model_switch_without_stop() -> None:
    runtime = VLLMRuntime(VLLMRuntimeConfig(image="vllm-openai.sif"))
    runtime._process = object()  # type: ignore[assignment]
    runtime._loaded_model = "model-a"

    with pytest.raises(RuntimeError, match="stop it before switching"):
        await runtime.start("model-b")
