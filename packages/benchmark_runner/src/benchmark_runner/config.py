from __future__ import annotations

from pathlib import Path

from llm_gateway import Provider, ProviderConfig, create_client

from benchmark_runner.models import BenchmarkSpec, DatasetRunConfig


MODEL_POOL: tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",
    # "openai/gpt-oss-120b",
    # "qwen/qwen-2.5-72b-instruct",
    # "qwen/qwen-2.5-coder-32b-instruct",
    # "deepseek/deepseek-r1",
    # "z-ai/glm-z1-32b",
    # "qwen/qwq-32b",
    # "anthropic/claude-opus-4",
    # "minimax/minimax-m1",
    # "qwen/qwen3-30b-a3b",
    # "deepseek/deepseek-v3.2",
    # "z-ai/glm-5",
    # "moonshotai/kimi-k2.5",
    # "google/gemini-3-pro-preview",
)

DATASET_CONFIGS: tuple[DatasetRunConfig, ...] = (
    DatasetRunConfig(name="musique", split="validation"),
    DatasetRunConfig(name="quality", split="validation"),
    DatasetRunConfig(name="fever", split="validation"),
    DatasetRunConfig(name="hardmath", split="test"),
    DatasetRunConfig(name="humaneval_plus", split="test"),
)


def default_provider_config() -> ProviderConfig:
    return ProviderConfig(
        provider=Provider.HUGGINGFACE,
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
    print(f"get_preset_spec called {preset}, {output_root}")
    normalized = preset.strip().lower()
    if normalized == "pilot":
        return build_benchmark_spec(
            output_root=output_root,
            max_examples_per_dataset=200,
            max_concurrency=1, batch_size=1,
            delay_between_requests=8)
    if normalized == "full":
        return build_benchmark_spec(output_root=output_root, max_examples_per_dataset=160)
    raise ValueError("Unsupported preset. Expected one of: pilot, full")


def get_dataset_configs(names: tuple[str, ...] | None = None) -> tuple[DatasetRunConfig, ...]:
    if not names:
        return DATASET_CONFIGS
    selected = {name.strip().lower() for name in names}
    return tuple(config for config in DATASET_CONFIGS if config.name in selected)
