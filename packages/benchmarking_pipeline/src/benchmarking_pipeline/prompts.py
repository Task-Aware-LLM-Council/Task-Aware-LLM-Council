from __future__ import annotations

from llm_gateway import PromptRequest

from benchmarking_pipeline.models import BenchmarkExample, BenchmarkRunConfig


def build_prompt_request(
    example: BenchmarkExample,
    *,
    model: str,
    config: BenchmarkRunConfig,
) -> PromptRequest:
    metadata = dict(example.metadata)
    metadata.update(
        {
            "dataset_name": example.dataset_name,
            "example_id": example.example_id,
            "prompt_version": config.prompt_version,
        }
    )

    if example.messages:
        return PromptRequest(
            model=model,
            messages=example.messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_sequences=config.stop_sequences,
            metadata=metadata,
            provider_params=dict(config.provider_params),
        )

    if not example.question and not example.context:
        raise ValueError(
            f"benchmark example {example.example_id} must include messages or prompt content"
        )

    return PromptRequest(
        model=model,
        system_prompt=example.system_prompt,
        user_prompt=example.question,
        context=example.context,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stop_sequences=config.stop_sequences,
        metadata=metadata,
        provider_params=dict(config.provider_params),
    )
