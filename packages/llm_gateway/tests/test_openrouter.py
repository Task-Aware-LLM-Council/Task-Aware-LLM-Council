from __future__ import annotations

import pytest

from llm_gateway.models import PromptRequest, ProviderConfig, Provider
from llm_gateway.providers.openrouter import OPENROUTER_API_BASE, OpenRouterClient


pytestmark = pytest.mark.unit


def _openrouter_response(text: str = "router ok") -> dict:
    return {
        "id": "or_123",
        "model": "openrouter/test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
        },
    }


@pytest.mark.asyncio
async def test_openrouter_client_applies_default_config(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = OpenRouterClient(
        ProviderConfig(
            provider=Provider.OPENROUTER,
            default_model="openrouter/test-model",
        )
    )

    assert client.provider == Provider.OPENROUTER
    assert client.default_model == "openrouter/test-model"
    assert client.require_api_base() == OPENROUTER_API_BASE
    assert client._get_api_key() == "test-key"
    assert client._build_headers()["Authorization"] == "Bearer test-key"

    await client.close()


@pytest.mark.asyncio
async def test_openrouter_client_respects_explicit_overrides(httpx_mock, monkeypatch) -> None:
    monkeypatch.setenv("CUSTOM_OPENROUTER_KEY", "custom-key")
    httpx_mock.add_response(json=_openrouter_response("override ok"))

    client = OpenRouterClient(
        ProviderConfig(
            provider=Provider.OPENROUTER,
            api_base="https://custom.openrouter.test/chat/completions",
            api_key_env="CUSTOM_OPENROUTER_KEY",
            default_model="openrouter/test-model",
            headers={"HTTP-Referer": "https://example.com"},
            default_params={"top_p": 0.9},
        )
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Say hi",
            provider_params={"presence_penalty": 0.1},
        )
    )

    [request] = httpx_mock.get_requests()
    payload = request.read().decode("utf-8")

    assert str(request.url) == "https://custom.openrouter.test/chat/completions"
    assert request.headers["Authorization"] == "Bearer custom-key"
    assert request.headers["HTTP-Referer"] == "https://example.com"
    assert '"top_p":0.9' in payload
    assert '"presence_penalty":0.1' in payload
    assert response.text == "override ok"
    assert response.provider == Provider.OPENROUTER

    await client.close()
