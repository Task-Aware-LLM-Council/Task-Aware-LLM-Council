from __future__ import annotations

import os

import pytest
import pytest_asyncio
import asyncio

from llm_gateway.factory import create_client
from llm_gateway.models import PromptRequest, ProviderConfig, Provider, RetryPolicy
from llm_gateway.providers.openai import OpenAIClient
from llm_gateway.providers.openrouter import OpenRouterClient


pytestmark = pytest.mark.smoke

@pytest_asyncio.fixture(autouse=True)
async def delay_between_tests():
    await asyncio.sleep(5)


def _require_env(*names: str) -> dict[str, str]:
    values: dict[str, str] = {}
    missing = [name for name in names if not os.getenv(name)]
    print(missing)
    if missing:
        pytest.skip(f"missing environment variables: {', '.join(missing)}")
    for name in names:
        values[name] = os.environ[name]
    return values


@pytest.mark.asyncio
async def test_openai_smoke() -> None:
    env = _require_env("OPENAI_API_KEY", "OPENAI_SMOKE_MODEL")
    client = OpenAIClient(
        ProviderConfig(
            provider=Provider.OPENAI,
            default_model=env["OPENAI_SMOKE_MODEL"],
        )
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Reply with exactly two words about testing.",
            max_tokens=12,
            temperature=0.0,
        )
    )

    assert response.text.strip()
    assert response.provider == "openai"
    assert response.model

    await client.close()


@pytest.mark.asyncio
async def test_openrouter_smoke() -> None:
    env = _require_env("OPENROUTER_API_KEY", "OPENROUTER_SMOKE_MODEL")
    client = OpenRouterClient(
        ProviderConfig(
            provider=Provider.OPENROUTER,
            default_model=env["OPENROUTER_SMOKE_MODEL"],
        )
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Reply with exactly two words about testing.",
            max_tokens=12,
            temperature=0.0,
        )
    )

    assert response.text.strip()
    assert response.provider == Provider.OPENROUTER
    assert response.model

    await client.close()


@pytest.mark.asyncio
async def test_huggingface_smoke() -> None:
    env = _require_env("HUGGINGFACE_API_KEY", "HUGGINGFACE_SMOKE_MODEL")
    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
            default_model=env["HUGGINGFACE_SMOKE_MODEL"],
        ),
        retry_policy=RetryPolicy(max_retries=6)
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Reply with exactly two words about testing.",
            max_tokens=12,
            temperature=0.0,
        )
    )
    print("Response is: ", response)

    assert response.text.strip()
    assert response.provider == "huggingface"
    assert response.model.lower() == env["HUGGINGFACE_SMOKE_MODEL"].lower()

    await client.close()


@pytest.mark.asyncio
async def test_local_vllm_smoke() -> None:
    env = _require_env("LOCAL_VLLM_SMOKE_MODEL")

    default_params: dict[str, object] = {}
    if os.getenv("LOCAL_VLLM_SMOKE_DTYPE"):
        default_params["dtype"] = os.environ["LOCAL_VLLM_SMOKE_DTYPE"]
    if os.getenv("LOCAL_VLLM_SMOKE_TENSOR_PARALLEL_SIZE"):
        default_params["tensor_parallel_size"] = int(
            os.environ["LOCAL_VLLM_SMOKE_TENSOR_PARALLEL_SIZE"]
        )
    if os.getenv("LOCAL_VLLM_SMOKE_GPU_MEMORY_UTILIZATION"):
        default_params["gpu_memory_utilization"] = float(
            os.environ["LOCAL_VLLM_SMOKE_GPU_MEMORY_UTILIZATION"]
        )
    if os.getenv("LOCAL_VLLM_SMOKE_MAX_MODEL_LEN"):
        default_params["max_model_len"] = int(os.environ["LOCAL_VLLM_SMOKE_MAX_MODEL_LEN"])
    if os.getenv("LOCAL_VLLM_SMOKE_TRUST_REMOTE_CODE"):
        default_params["trust_remote_code"] = os.environ[
            "LOCAL_VLLM_SMOKE_TRUST_REMOTE_CODE"
        ].strip().lower() in {"1", "true", "yes"}

    client = create_client(
        ProviderConfig(
            provider=Provider.LOCAL,
            default_model=env["LOCAL_VLLM_SMOKE_MODEL"],
            default_params=default_params,
        ),
        retry_policy=RetryPolicy(max_retries=0),
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Reply with exactly two words about testing.",
            max_tokens=12,
            temperature=0.0,
        )
    )

    assert response.text.strip()
    assert response.provider == "local"
    assert response.model == env["LOCAL_VLLM_SMOKE_MODEL"]
    assert response.metadata["runtime"] == "vllm"
    assert response.metadata["loaded_model"] == env["LOCAL_VLLM_SMOKE_MODEL"]

    await client.close()
