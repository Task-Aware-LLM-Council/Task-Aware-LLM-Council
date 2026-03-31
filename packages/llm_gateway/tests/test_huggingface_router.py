from __future__ import annotations

import pytest

from llm_gateway.factory import create_client
from llm_gateway.base import LLMRateLimitError
from llm_gateway.models import PromptRequest, ProviderConfig, RetryPolicy, Provider
from llm_gateway.providers.openai_compatible import OpenAICompatibleClient


pytestmark = pytest.mark.unit


def _hf_router_response(text: str = "hf router ok") -> dict:
    return {
        "id": "hf_123",
        "model": "Qwen/Qwen3-32B",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "total_tokens": 15,
        },
    }


@pytest.mark.asyncio
async def test_huggingface_factory_returns_openai_compatible_client(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-key")
    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
            default_model="Qwen/Qwen3-32B",
        )
    )

    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "huggingface"
    assert client.default_model == "Qwen/Qwen3-32B"
    assert client.require_api_base() == "https://router.huggingface.co/v1/chat/completions"
    assert client._get_api_key() == "hf-key"
    assert client._build_headers()["Authorization"] == "Bearer hf-key"

    await client.close()


@pytest.mark.asyncio
async def test_huggingface_router_uses_openai_payload(httpx_mock) -> None:
    httpx_mock.add_response(json=_hf_router_response("router payload ok"))
    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
            default_model="Qwen/Qwen3-32B",
            default_params={"top_p": 0.95},
        )
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Who are you?",
            temperature=0.0,
            max_tokens=24,
        )
    )

    [request] = httpx_mock.get_requests()
    payload = request.read().decode("utf-8")

    assert str(request.url) == "https://router.huggingface.co/v1/chat/completions"
    assert '"model":"Qwen/Qwen3-32B"' in payload
    assert '"messages":[{"role":"user","content":"Who are you?"}]' in payload
    assert '"top_p":0.95' in payload
    assert response.text == "router payload ok"
    assert response.provider == "huggingface"
    assert response.metadata["attempt_count"] == 1

    await client.close()


@pytest.mark.asyncio
async def test_huggingface_router_retries_like_openai_compatible(httpx_mock, monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("llm_gateway.base.asyncio.sleep", fake_sleep)
    httpx_mock.add_response(status_code=503, json={"error": "busy"})
    httpx_mock.add_response(json=_hf_router_response("retry ok"))

    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
            default_model="Qwen/Qwen3-32B",
        ),
        retry_policy=RetryPolicy(
            max_retries=1, initial_backoff_seconds=0.25, jitter_ratio=0.0),
    )

    response = await client.generate(PromptRequest(user_prompt="Retry"))

    assert response.text == "retry ok"
    assert sleep_calls == [0.25]
    assert response.metadata["attempt_count"] == 2

    await client.close()


@pytest.mark.asyncio
async def test_huggingface_router_retries_on_402_rate_limit(httpx_mock, monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("llm_gateway.base.asyncio.sleep", fake_sleep)
    httpx_mock.add_response(status_code=402, json={
                            "error": "payment required"})
    httpx_mock.add_response(json=_hf_router_response("402 retry ok"))

    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
            default_model="Qwen/Qwen3-32B",
        ),
        retry_policy=RetryPolicy(
            max_retries=1, initial_backoff_seconds=0.1, jitter_ratio=0.0),
    )

    response = await client.generate(PromptRequest(user_prompt="Retry on 402"))

    assert response.text == "402 retry ok"
    assert sleep_calls == [0.1]
    assert response.metadata["attempt_count"] == 2

    await client.close()


@pytest.mark.asyncio
async def test_huggingface_router_classifies_402_as_rate_limit(httpx_mock) -> None:
    httpx_mock.add_response(status_code=402, json={
                            "error": "payment required"})

    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
            default_model="Qwen/Qwen3-32B",
        ),
        retry_policy=RetryPolicy(max_retries=0, jitter_ratio=0.0),
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        await client.generate(PromptRequest(user_prompt="Classify 402"))

    assert exc_info.value.status_code == 402

    await client.close()
