from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_gateway.base import LLMRequestError
from llm_gateway.models import PromptRequest, Provider, ProviderConfig
from llm_gateway.providers.local_vllm import LocalVLLMClient


pytestmark = pytest.mark.unit


class _FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeLLM:
    init_calls: list[dict[str, object]] = []
    chat_calls: list[dict[str, object]] = []

    def __init__(self, *, model: str, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs
        _FakeLLM.init_calls.append({"model": model, **kwargs})

    def chat(self, *, messages, sampling_params, use_tqdm):
        _FakeLLM.chat_calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        return [
            SimpleNamespace(
                request_id="req-local-1",
                prompt_token_ids=[1, 2, 3],
                num_cached_tokens=0,
                finished=True,
                outputs=[
                    SimpleNamespace(
                        text="local ok",
                        finish_reason="stop",
                        token_ids=[10, 11],
                    )
                ],
            )
        ]


def _patch_vllm_import(monkeypatch) -> None:
    _FakeLLM.init_calls.clear()
    _FakeLLM.chat_calls.clear()
    monkeypatch.setattr(
        "llm_gateway.providers.local_vllm._import_vllm",
        lambda: (_FakeLLM, _FakeSamplingParams),
    )


@pytest.mark.asyncio
async def test_local_vllm_client_loads_model_lazily_and_parses_response(monkeypatch) -> None:
    _patch_vllm_import(monkeypatch)
    client = LocalVLLMClient(
        ProviderConfig(
            provider=Provider.LOCAL,
            default_model="Qwen/Qwen2.5-1.5B-Instruct",
            default_params={"dtype": "float16", "tensor_parallel_size": 1},
        )
    )

    response = await client.generate(
        PromptRequest(
            user_prompt="Say hello",
            temperature=0.0,
            max_tokens=16,
            provider_params={"top_p": 0.9, "ignored": 123},
        )
    )

    assert _FakeLLM.init_calls == [
        {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "dtype": "float16",
            "tensor_parallel_size": 1,
        }
    ]
    assert _FakeLLM.chat_calls[0]["messages"] == [{"role": "user", "content": "Say hello"}]
    assert _FakeLLM.chat_calls[0]["use_tqdm"] is False
    assert _FakeLLM.chat_calls[0]["sampling_params"].kwargs == {
        "temperature": 0.0,
        "max_tokens": 16,
        "top_p": 0.9,
    }
    assert response.text == "local ok"
    assert response.provider == "local"
    assert response.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2
    assert response.metadata["runtime"] == "vllm"
    assert response.metadata["loaded_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert response.metadata["attempt_count"] == 1

    await client.close()


@pytest.mark.asyncio
async def test_local_vllm_client_reuses_loaded_model(monkeypatch) -> None:
    _patch_vllm_import(monkeypatch)
    client = LocalVLLMClient(
        ProviderConfig(
            provider=Provider.LOCAL,
            default_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
    )

    await client.generate(PromptRequest(user_prompt="first"))
    await client.generate(PromptRequest(user_prompt="second"))

    assert len(_FakeLLM.init_calls) == 1
    assert len(_FakeLLM.chat_calls) == 2

    await client.close()


@pytest.mark.asyncio
async def test_local_vllm_client_rejects_switching_models_on_one_client(monkeypatch) -> None:
    _patch_vllm_import(monkeypatch)
    client = LocalVLLMClient(
        ProviderConfig(
            provider=Provider.LOCAL,
            default_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
    )

    await client.generate(PromptRequest(user_prompt="first"))

    with pytest.raises(LLMRequestError) as exc_info:
        await client.generate(
            PromptRequest(
                model="Qwen/Qwen2.5-7B-Instruct",
                user_prompt="switch",
            )
        )

    assert "create a new client" in str(exc_info.value)
    assert len(_FakeLLM.init_calls) == 1

    await client.close()
