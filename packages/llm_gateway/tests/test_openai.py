from __future__ import annotations

import pytest

from llm_gateway.models import PromptRequest, ProviderConfig, Provider
from llm_gateway.providers.openai import OPENAI_API_BASE, OpenAIClient


pytestmark = pytest.mark.unit


def _openai_response(text: str = "openai ok") -> dict:
    return {
        "id": "oa_123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
        },
    }


@pytest.mark.asyncio
async def test_openai_client_applies_default_config(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient(
        ProviderConfig(
            provider=Provider.OPENAI,
            default_model="gpt-4o-mini",
        )
    )

    assert client.provider == "openai"
    assert client.default_model == "gpt-4o-mini"
    assert client.require_api_base() == OPENAI_API_BASE
    assert client._get_api_key() == "test-key"
    assert client._build_headers()["Authorization"] == "Bearer test-key"

    await client.close()


@pytest.mark.asyncio
async def test_openai_client_respects_explicit_overrides(httpx_mock, monkeypatch) -> None:
    monkeypatch.setenv("CUSTOM_OPENAI_KEY", "custom-key")
    httpx_mock.add_response(json=_openai_response("override ok"))

    client = OpenAIClient(
        ProviderConfig(
            provider=Provider.OPENAI,
            api_base="https://custom.openai.test/v1/chat/completions",
            api_key_env="CUSTOM_OPENAI_KEY",
            default_model="gpt-4o-mini",
            headers={"OpenAI-Organization": "org_123"},
            default_params={"top_p": 0.8},
        )
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Say hi",
            provider_params={"presence_penalty": 0.2},
        )
    )

    [request] = httpx_mock.get_requests()
    payload = request.read().decode("utf-8")

    assert str(request.url) == "https://custom.openai.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer custom-key"
    assert request.headers["OpenAI-Organization"] == "org_123"
    assert '"top_p":0.8' in payload
    assert '"presence_penalty":0.2' in payload
    assert response.text == "override ok"
    assert response.provider == "openai"

    await client.close()
