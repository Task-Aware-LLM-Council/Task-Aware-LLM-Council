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
    LocalVLLMClient,
    OpenAIClient,
    OpenAICompatibleClient,
    OpenRouterClient,
)

__all__ = [
    "BaseLLMClient",
    "ClientInfo",
    "HTTPErrorPolicy",
    "create_client",
    "LocalVLLMClient",
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
]
