from __future__ import annotations

import httpx
import pytest

from llm_gateway.base import LLMRequestError
from llm_gateway.models import PromptRequest, ProviderConfig, RetryPolicy
from llm_gateway.providers.openai_compatible import OpenAICompatibleClient


pytestmark = pytest.mark.unit


def _openai_client(*, retry_policy: RetryPolicy | None = None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        ProviderConfig(
            provider="openai-compatible",
            api_base="https://example.com/v1/chat/completions",
            default_model="test-model",
        ),
        retry_policy=retry_policy,
    )


def _openai_response(text: str = "hello") -> dict:
    return {
        "id": "resp_123",
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
        },
    }


@pytest.mark.asyncio
async def test_openai_client_builds_payload_and_parses_response(httpx_mock) -> None:
    httpx_mock.add_response(json=_openai_response("payload ok"))
    client = _openai_client()

    response = await client.generate(
        PromptRequest(
            user_prompt="Solve this",
            system_prompt="Be concise",
            temperature=0.2,
            max_tokens=32,
            stop_sequences=("END",),
        )
    )

    [request] = httpx_mock.get_requests()
    payload = request.read().decode("utf-8")
    assert str(request.url) == "https://example.com/v1/chat/completions"
    assert '"model":"test-model"' in payload
    assert '"temperature":0.2' in payload
    assert '"max_tokens":32' in payload
    assert '"stop":["END"]' in payload
    assert '"role":"system"' in payload
    assert '"role":"user"' in payload

    assert response.text == "payload ok"
    assert response.provider == "openai-compatible"
    assert response.usage.total_tokens == 17
    assert response.metadata["attempt_count"] == 1
    assert response.metadata["status_code"] == 200

    await client.close()


@pytest.mark.asyncio
async def test_openai_client_retries_on_rate_limit(httpx_mock, monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("llm_gateway.base.asyncio.sleep", fake_sleep)

    httpx_mock.add_response(status_code=429, headers={"Retry-After": "0.2"}, json={"error": "busy"})
    httpx_mock.add_response(json=_openai_response("recovered"))

    client = _openai_client(
        retry_policy=RetryPolicy(max_retries=1, jitter_ratio=0.0),
    )

    response = await client.generate(PromptRequest(user_prompt="retry me"))

    assert response.text == "recovered"
    assert sleep_calls == [0.2]
    assert response.metadata["attempt_count"] == 2
    assert response.metadata["retry_after_used"] == 0.2
    assert len(httpx_mock.get_requests()) == 2

    await client.close()


@pytest.mark.asyncio
async def test_openai_client_retries_on_transport_error(httpx_mock, monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("llm_gateway.base.asyncio.sleep", fake_sleep)

    httpx_mock.add_exception(httpx.ConnectError("boom"))
    httpx_mock.add_response(json=_openai_response("transport recovered"))

    client = _openai_client(
        retry_policy=RetryPolicy(
            max_retries=1,
            initial_backoff_seconds=0.5,
            backoff_multiplier=2.0,
            jitter_ratio=0.0,
        ),
    )

    response = await client.generate(PromptRequest(user_prompt="network"))

    assert response.text == "transport recovered"
    assert sleep_calls == [0.5]
    assert response.metadata["attempt_count"] == 2

    await client.close()


@pytest.mark.asyncio
async def test_openai_client_does_not_retry_non_retryable_4xx(httpx_mock, monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("llm_gateway.base.asyncio.sleep", fake_sleep)
    httpx_mock.add_response(status_code=400, json={"error": "bad request"})

    client = _openai_client(retry_policy=RetryPolicy(max_retries=3, jitter_ratio=0.0))

    with pytest.raises(LLMRequestError) as exc_info:
        await client.generate(PromptRequest(user_prompt="bad"))

    assert exc_info.value.status_code == 400
    assert sleep_calls == []
    assert len(httpx_mock.get_requests()) == 1

    await client.close()


@pytest.mark.asyncio
async def test_openai_client_does_not_retry_402_by_default(httpx_mock, monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("llm_gateway.base.asyncio.sleep", fake_sleep)
    httpx_mock.add_response(status_code=402, json={"error": "payment required"})

    client = _openai_client(retry_policy=RetryPolicy(max_retries=2, jitter_ratio=0.0))

    with pytest.raises(LLMRequestError) as exc_info:
        await client.generate(PromptRequest(user_prompt="bill me"))

    assert exc_info.value.status_code == 402
    assert sleep_calls == []
    assert len(httpx_mock.get_requests()) == 1

    await client.close()
