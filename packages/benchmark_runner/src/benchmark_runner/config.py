from __future__ import annotations
from pathlib import Path

from llm_gateway import Provider, ProviderConfig

from benchmark_runner.models import BenchmarkSpec, DatasetRunConfig

MODEL_POOL: tuple[str, ...] = (
    # Generalists (MuSiQue, FEVER)
    "internlm/internlm2_5-7b-chat-1m", # Done
    "google/gemma-2-9b-it", # Done
    "Qwen/Qwen2.5-7B-Instruct", # Done
    "NousResearch/Hermes-3-Llama-3.1-8B", # Done
    # Long-Context Specialists (QuALITY)
    "mistralai/Mistral-Nemo-Instruct-2407", # Done
    "Qwen/Qwen2.5-14B-Instruct", # Done
    # Math Specialists (HARDMath)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # Done
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # Done
    "01-ai/Yi-1.5-9B-Chat-16K", # Done
    # Code Specialists (HumanEval+)
    "Qwen/Qwen2.5-Coder-7B-Instruct", # Done
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", # Done
    "THUDM/glm-4-9b-chat" # Done
)

DATASET_CONFIGS: tuple[DatasetRunConfig, ...] = (
    DatasetRunConfig(name="musique", split="validation"),
    DatasetRunConfig(name="quality", split="validation"),
    DatasetRunConfig(name="fever", split="validation"),
    DatasetRunConfig(name="hardmath", split="test"),
    DatasetRunConfig(name="humaneval_plus", split="test"),
)


def default_provider_config(
    *,
    provider: Provider | str = Provider.LOCAL,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> ProviderConfig:
    normalized_provider = provider.value if isinstance(provider, Provider) else provider

    if normalized_provider == Provider.LOCAL.value:
        return ProviderConfig(
            provider=Provider.LOCAL,
            api_base=api_base,
            api_key_env=api_key_env,
            default_model=MODEL_POOL[0],
            default_params={},
        )

    return ProviderConfig(
        provider=provider,
        api_base=api_base,
        api_key_env=api_key_env,
        default_model=MODEL_POOL[0],
    )


def build_benchmark_spec(
    *,
    output_root: Path,
    models: tuple[str, ...] = MODEL_POOL,
    max_examples_per_dataset: int | None = 50,
    prompt_version: str = "v1",
    max_concurrency: int = 5,
    provider_config: ProviderConfig | None = None,
    batch_size=32,
    delay_between_requests=5
) -> BenchmarkSpec:
    return BenchmarkSpec(
        provider_config=provider_config or default_provider_config(),
        models=models,
        output_root=output_root,
        prompt_version=prompt_version,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        max_examples_per_dataset=max_examples_per_dataset,
        delay_between_requests=delay_between_requests
    )


def get_preset_spec(preset: str, *, output_root: Path) -> BenchmarkSpec:
    normalized = preset.strip().lower()
    if normalized == "pilot":
        return build_benchmark_spec(
            output_root=output_root,
            max_examples_per_dataset=500,
            # The pipeline slows down (for variable minutes during the end of the batch)
            # Keeping 1 batch (max_examples_per_dataset / batch_size = 1) will lead to slowness only once.
            batch_size=500, 
            # 2. Let the asyncio.Semaphore limit it to exactly 50 active requests.
            # As soon as 1 request finishes, the semaphore instantly pulls next request
            max_concurrency=100,
            delay_between_requests=0)
    if normalized == "full":
        return build_benchmark_spec(output_root=output_root, max_examples_per_dataset=160)
    raise ValueError("Unsupported preset. Expected one of: pilot, full")


def get_dataset_configs(names: tuple[str, ...] | None = None) -> tuple[DatasetRunConfig, ...]:
    if not names:
        return DATASET_CONFIGS
    selected = {name.strip().lower() for name in names}
    return tuple(config for config in DATASET_CONFIGS if config.name in selected)
