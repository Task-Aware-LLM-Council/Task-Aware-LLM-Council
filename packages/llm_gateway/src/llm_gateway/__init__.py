"""Public API for llm_gateway.

Preferred imports come from this package root:

    from llm_gateway import OpenRouterClient, PromptRequest, ProviderConfig

Use provider-specific wrappers for common cases and `create_client(...)` when
the caller wants config-driven client construction.
"""

from dotenv import load_dotenv
load_dotenv()

from llm_gateway.base import (
    BaseLLMClient,
    ClientInfo,
    HTTPErrorPolicy,
    LLMClientError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMTransportError,
)
from llm_gateway.factory import create_client
from llm_gateway.models import (
    Message,
    PromptRequest,
    PromptResponse,
    Provider,
    ProviderConfig,
    ResponseChoice,
    RetryPolicy,
    Usage,
)
from llm_gateway.providers import (
    OpenAIClient,
    OpenAICompatibleClient,
    OpenRouterClient,
)
from llm_gateway.vllm_runtime import (
    LOCAL_LAUNCH_BIND,
    LOCAL_LAUNCH_CLIENT_HOST,
    LOCAL_LAUNCH_CONTAINER_CACHE_DIR,
    LOCAL_LAUNCH_CPU_OFFLOAD_GB,
    LOCAL_LAUNCH_DTYPE,
    LOCAL_LAUNCH_ENV,
    LOCAL_LAUNCH_EXECUTABLE,
    LOCAL_LAUNCH_EXTRA_ARGS,
    LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION,
    LOCAL_LAUNCH_IMAGE,
    LOCAL_LAUNCH_LOAD_FORMAT,
    LOCAL_LAUNCH_MAX_MODEL_LEN,
    LOCAL_LAUNCH_POLL_INTERVAL,
    LOCAL_LAUNCH_PORT,
    LOCAL_LAUNCH_PREFIX,
    LOCAL_LAUNCH_QUANTIZATION,
    LOCAL_LAUNCH_SERVER_HOST,
    LOCAL_LAUNCH_STARTUP_TIMEOUT,
    LOCAL_LAUNCH_USE_GPU,
    VLLMRuntime,
    VLLMRuntimeConfig,
    build_vllm_runtime_config,
    managed_local_provider_config,
    normalize_local_provider_config,
)

__all__ = [
    "BaseLLMClient",
    "ClientInfo",
    "HTTPErrorPolicy",
    "create_client",
    "OpenAIClient",
    "LLMClientError",
    "LLMRateLimitError",
    "LLMRequestError",
    "LLMResponseError",
    "LLMTransportError",
    "Message",
    "PromptRequest",
    "PromptResponse",
    "Provider",
    "ProviderConfig",
    "ResponseChoice",
    "RetryPolicy",
    "Usage",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "LOCAL_LAUNCH_BIND",
    "LOCAL_LAUNCH_CLIENT_HOST",
    "LOCAL_LAUNCH_CONTAINER_CACHE_DIR",
    "LOCAL_LAUNCH_CPU_OFFLOAD_GB",
    "LOCAL_LAUNCH_DTYPE",
    "LOCAL_LAUNCH_ENV",
    "LOCAL_LAUNCH_EXECUTABLE",
    "LOCAL_LAUNCH_EXTRA_ARGS",
    "LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION",
    "LOCAL_LAUNCH_IMAGE",
    "LOCAL_LAUNCH_LOAD_FORMAT",
    "LOCAL_LAUNCH_MAX_MODEL_LEN",
    "LOCAL_LAUNCH_POLL_INTERVAL",
    "LOCAL_LAUNCH_PORT",
    "LOCAL_LAUNCH_PREFIX",
    "LOCAL_LAUNCH_QUANTIZATION",
    "LOCAL_LAUNCH_SERVER_HOST",
    "LOCAL_LAUNCH_STARTUP_TIMEOUT",
    "LOCAL_LAUNCH_USE_GPU",
    "VLLMRuntime",
    "VLLMRuntimeConfig",
    "build_vllm_runtime_config",
    "managed_local_provider_config",
    "normalize_local_provider_config",
]
