from __future__ import annotations

import llm_gateway


def test_public_api_exports_documented_symbols() -> None:
    exported_names = {
        "BaseLLMClient",
        "ClientInfo",
        "HTTPErrorPolicy",
        "create_client",
        "OpenAIClient",
        "OpenAICompatibleClient",
        "OpenRouterClient",
        "LLMClientError",
        "LLMRequestError",
        "LLMRateLimitError",
        "LLMResponseError",
        "LLMTransportError",
        "Message",
        "PromptRequest",
        "PromptResponse",
        "Provider",
        "ProviderConfig",
        "RetryPolicy",
        "ResponseChoice",
        "Usage",
        "VLLMRuntime",
        "VLLMRuntimeConfig",
        "build_vllm_runtime_config",
        "managed_local_provider_config",
        "normalize_local_provider_config",
    }

    for name in exported_names:
        assert hasattr(llm_gateway, name), f"missing public export: {name}"


def test_provider_enum_exports_local() -> None:
    assert llm_gateway.Provider.LOCAL.value == "local"
