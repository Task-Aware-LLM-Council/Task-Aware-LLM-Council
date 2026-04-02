from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from llm_gateway.base import (
    BaseLLMClient,
    ClientInfo,
    LLMClientError,
    LLMRequestError,
    LLMResponseError,
    LLMTransportError,
)
from llm_gateway.models import (
    Message,
    PromptRequest,
    PromptResponse,
    ProviderConfig,
    ResponseChoice,
    RetryPolicy,
    Usage,
)

LOCAL_RUNTIME_PARAM_KEYS = {
    "tokenizer",
    "tokenizer_mode",
    "skip_tokenizer_init",
    "trust_remote_code",
    "tensor_parallel_size",
    "dtype",
    "quantization",
    "revision",
    "tokenizer_revision",
    "seed",
    "gpu_memory_utilization",
    "swap_space",
    "cpu_offload_gb",
    "enforce_eager",
    "max_model_len",
    "hf_token",
    "device"
}
LOCAL_SAMPLING_PARAM_KEYS = {
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "min_p",
    "n",
}


def _import_vllm() -> tuple[type[Any], type[Any]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise LLMClientError(
            "local vLLM support requires the `vllm` package to be installed"
        ) from exc
    return LLM, SamplingParams


class LocalVLLMClient(BaseLLMClient):
    """In-process local inference client backed by the vLLM Python API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        print("----INIT----")
        info = ClientInfo(
            provider=config.provider,
            default_model=config.default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        super().__init__(info, config=config, retry_policy=retry_policy)
        self._runtime_lock = asyncio.Lock()
        self._loaded_model: str | None = None
        self._engine: Any | None = None

    async def close(self) -> None:
        self._engine = None
        self._loaded_model = None
        await super().close()

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.ensure_open()
        self.validate_request(request)
        started = perf_counter()
        model = self.resolve_model(request)

        async with self._runtime_lock:
            engine = await self._ensure_engine(model)
            messages = [self._message_to_payload(message) for message in request.resolved_messages()]
            sampling_params = self._build_sampling_params(request)
            try:
                outputs = await asyncio.to_thread(
                    engine.chat,
                    messages=messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except Exception as exc:
                raise LLMTransportError(f"{self.provider} generation failed: {exc}") from exc

        parsed = self._parse_output(
            output=self._unwrap_single_output(outputs),
            model=model,
            latency_ms=(perf_counter() - started) * 1000,
        )
        return self._with_attempt_metadata(parsed, attempt_count=1)

    async def _ensure_engine(self, model: str) -> Any:
        if self._engine is not None:
            if self._loaded_model != model:
                raise LLMRequestError(
                    f"{self.provider} client already loaded model "
                    f"{self._loaded_model!r}; create a new client to switch to {model!r}"
                )
            return self._engine

        try:
            self._engine = await asyncio.to_thread(self._create_engine, model)
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMTransportError(f"{self.provider} failed to initialize runtime: {exc}") from exc
        self._loaded_model = model
        return self._engine

    def _create_engine(self, model: str) -> Any:
        LLM, _ = _import_vllm()
        runtime_kwargs = self._build_runtime_kwargs()
        return LLM(model=model, **runtime_kwargs)

    def _build_runtime_kwargs(self) -> dict[str, Any]:
        if not self.config:
            return {}
        return {
            key: value
            for key, value in self.config.default_params.items()
            if key in LOCAL_RUNTIME_PARAM_KEYS
        }

    def _build_sampling_params(self, request: PromptRequest) -> Any:
        _, SamplingParams = _import_vllm()
        params: dict[str, Any] = {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            params["stop"] = list(request.stop_sequences)
        params.update(
            {
                key: value
                for key, value in request.provider_params.items()
                if key in LOCAL_SAMPLING_PARAM_KEYS
            }
        )
        return SamplingParams(**params)

    @staticmethod
    def _message_to_payload(message: Message) -> dict[str, Any]:
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _unwrap_single_output(outputs: Any) -> Any:
        if not outputs:
            raise LLMResponseError("local response did not include any outputs")
        if isinstance(outputs, list):
            return outputs[0]
        return outputs

    def _parse_output(
        self,
        *,
        output: Any,
        model: str,
        latency_ms: float,
    ) -> PromptResponse:
        completions = getattr(output, "outputs", None)
        if not completions:
            raise LLMResponseError("local response did not include completion choices")

        choices: list[ResponseChoice] = []
        for index, completion in enumerate(completions):
            text = getattr(completion, "text", None)
            if text is None:
                raise LLMResponseError("local completion choice did not include text")
            choices.append(
                ResponseChoice(
                    index=index,
                    message=Message(role="assistant", content=text),
                    finish_reason=getattr(completion, "finish_reason", None),
                )
            )

        prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
        completion_token_ids = getattr(completions[0], "token_ids", None) or []
        usage = Usage(
            input_tokens=len(prompt_token_ids),
            output_tokens=len(completion_token_ids),
            total_tokens=len(prompt_token_ids) + len(completion_token_ids),
        )

        metadata = {
            "runtime": "vllm",
            "loaded_model": self._loaded_model,
        }
        num_cached_tokens = getattr(output, "num_cached_tokens", None)
        if num_cached_tokens is not None:
            metadata["num_cached_tokens"] = num_cached_tokens
        if getattr(output, "finished", None) is not None:
            metadata["finished"] = output.finished

        first_choice = choices[0]
        return PromptResponse(
            model=model,
            text=first_choice.message.content,
            choices=tuple(choices),
            usage=usage,
            latency_ms=latency_ms,
            request_id=getattr(output, "request_id", None),
            finish_reason=first_choice.finish_reason,
            provider=self.provider,
            metadata=metadata,
        )
