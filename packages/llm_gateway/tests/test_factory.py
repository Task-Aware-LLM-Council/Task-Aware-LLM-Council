from __future__ import annotations

import pytest

from llm_gateway.base import LLMClientError, LLMRequestError
from llm_gateway.factory import create_client
from llm_gateway.models import Provider, ProviderConfig
from llm_gateway.providers.openai import OpenAIClient
from llm_gateway.providers.openai_compatible import OpenAICompatibleClient
from llm_gateway.providers.openrouter import OpenRouterClient


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_factory_returns_openrouter_client() -> None:
    client = create_client(ProviderConfig(provider=Provider.OPENROUTER))
    assert isinstance(client, OpenRouterClient)
    await client.close()


@pytest.mark.asyncio
async def test_factory_accepts_provider_enum_for_openrouter() -> None:
    client = create_client(ProviderConfig(provider=Provider.OPENROUTER))
    assert isinstance(client, OpenRouterClient)
    assert client.provider == Provider.OPENROUTER
    await client.close()


@pytest.mark.asyncio
async def test_factory_returns_openai_client() -> None:
    client = create_client(ProviderConfig(provider=Provider.OPENAI))
    assert isinstance(client, OpenAIClient)
    await client.close()


@pytest.mark.asyncio
async def test_factory_returns_openai_compatible_client() -> None:
    client = create_client(
        ProviderConfig(
            provider="openai-compatible",
            api_base="https://example.com/v1/chat/completions",
        )
    )
    assert isinstance(client, OpenAICompatibleClient)
    await client.close()


@pytest.mark.asyncio
async def test_factory_returns_local_openai_compatible_client() -> None:
    client = create_client(
        ProviderConfig(
            provider=Provider.LOCAL,
            api_base="http://127.0.0.1:8000/v1/chat/completions",
            default_model="Qwen/Qwen2.5-7B-Instruct",
        )
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "local"
    assert client.require_api_base() == "http://127.0.0.1:8000/v1/chat/completions"
    await client.close()


@pytest.mark.asyncio
async def test_factory_accepts_vllm_alias_for_local_provider() -> None:
    client = create_client(
        ProviderConfig(
            provider="vllm",
            api_base="http://127.0.0.1:8001/v1/chat/completions",
            default_model="Qwen/Qwen2.5-7B-Instruct",
        )
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "local"
    await client.close()


def test_factory_rejects_local_provider_without_api_base() -> None:
    with pytest.raises(LLMRequestError):
        create_client(
            ProviderConfig(
                provider=Provider.LOCAL,
                default_model="Qwen/Qwen2.5-7B-Instruct",
            )
        )


@pytest.mark.asyncio
async def test_factory_returns_huggingface_openai_compatible_client() -> None:
    client = create_client(
        ProviderConfig(
            provider=Provider.HUGGINGFACE,
        )
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "huggingface"
    assert client.require_api_base() == "https://router.huggingface.co/v1/chat/completions"
    await client.close()


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(LLMClientError):
        create_client(ProviderConfig(provider="unknown"))
