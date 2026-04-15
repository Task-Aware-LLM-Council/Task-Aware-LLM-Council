from __future__ import annotations

from common import get_current_user
from llm_gateway import (
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
    LOCAL_LAUNCH_QUANTIZATION,
    LOCAL_LAUNCH_SERVER_HOST,
    LOCAL_LAUNCH_STARTUP_TIMEOUT,
    LOCAL_LAUNCH_USE_GPU,
    Provider,
    ProviderConfig,
)

from model_orchestration.models import (
    JSONLRecordingConfig,
    LocalVLLMPresetConfig,
    ModelSpec,
    OrchestratorConfig,
)


DEFAULT_QA_MODEL = "google/gemma-2-9b-it"
DEFAULT_REASONING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_GENERAL_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_LOCAL_VLLM_BIND = f"/scratch1/{get_current_user()}/.cache"


def build_default_orchestrator_config(
    *,
    provider: Provider | str = Provider.LOCAL,
    api_base: str | None = None,
    api_key_env: str | None = None,
    provider_defaults: dict[str, object] | None = None,
    qa_model: str = DEFAULT_QA_MODEL,
    reasoning_model: str = DEFAULT_REASONING_MODEL,
    general_model: str = DEFAULT_GENERAL_MODEL,
    recording: JSONLRecordingConfig | None = None,
    mode_label: str | None = None,
) -> OrchestratorConfig:
    normalized_provider = provider.value if isinstance(provider, Provider) else str(provider).strip().lower()
    if normalized_provider == Provider.LOCAL.value and api_base is None:
        return build_default_local_vllm_orchestrator_config(
            qa_model=qa_model,
            reasoning_model=reasoning_model,
            general_model=general_model,
            recording=recording,
            mode_label=mode_label,
            preset=LocalVLLMPresetConfig(provider_defaults=dict(provider_defaults or {})),
        )

    provider_defaults = dict(provider_defaults or {})
    normalized_mode = mode_label or _mode_label_for(provider)
    return OrchestratorConfig(
        models=(
            ModelSpec(
                role="qa",
                model=qa_model,
                aliases=("qa",),
                description="Question-answering specialist.",
                provider_config=_provider_config(
                    provider=provider,
                    api_base=api_base,
                    api_key_env=api_key_env,
                    model=qa_model,
                    provider_defaults=provider_defaults,
                ),
            ),
            ModelSpec(
                role="reasoning",
                model=reasoning_model,
                aliases=("reasoning", "math", "code"),
                description="Math and code specialist.",
                provider_config=_provider_config(
                    provider=provider,
                    api_base=api_base,
                    api_key_env=api_key_env,
                    model=reasoning_model,
                    provider_defaults=provider_defaults,
                ),
            ),
            ModelSpec(
                role="general",
                model=general_model,
                aliases=("general", "fever"),
                description="Strong generalist and FEVER-oriented model.",
                provider_config=_provider_config(
                    provider=provider,
                    api_base=api_base,
                    api_key_env=api_key_env,
                    model=general_model,
                    provider_defaults=provider_defaults,
                ),
            ),
        ),
        default_role="general",
        recording=recording,
        mode_label=normalized_mode,
    )


def build_default_local_vllm_orchestrator_config(
    *,
    qa_model: str = DEFAULT_QA_MODEL,
    reasoning_model: str = DEFAULT_REASONING_MODEL,
    general_model: str = DEFAULT_GENERAL_MODEL,
    recording: JSONLRecordingConfig | None = None,
    mode_label: str | None = None,
    preset: LocalVLLMPresetConfig | None = None,
) -> OrchestratorConfig:
    active_preset = preset or LocalVLLMPresetConfig(bind=DEFAULT_LOCAL_VLLM_BIND)
    if active_preset.bind is None:
        active_preset = LocalVLLMPresetConfig(
            base_port=active_preset.base_port,
            image=active_preset.image,
            bind=DEFAULT_LOCAL_VLLM_BIND,
            startup_timeout_seconds=active_preset.startup_timeout_seconds,
            max_model_len=active_preset.max_model_len,
            gpu_memory_utilization=active_preset.gpu_memory_utilization,
            quantization=active_preset.quantization,
            use_gpu=active_preset.use_gpu,
            client_host=active_preset.client_host,
            server_host=active_preset.server_host,
            container_cache_dir=active_preset.container_cache_dir,
            executable=active_preset.executable,
            poll_interval_seconds=active_preset.poll_interval_seconds,
            extra_args=active_preset.extra_args,
            env=dict(active_preset.env),
            dtype=active_preset.dtype,
            load_format=active_preset.load_format,
            cpu_offload_gb=active_preset.cpu_offload_gb,
            provider_defaults=dict(active_preset.provider_defaults),
            role_overrides={key: dict(value) for key, value in active_preset.role_overrides.items()},
        )

    specs = (
        ("qa", qa_model, ("qa",), "Question-answering specialist.", 0),
        ("reasoning", reasoning_model, ("reasoning", "math", "code"), "Math and code specialist.", 1),
        ("general", general_model, ("general", "fever"), "Strong generalist and FEVER-oriented model.", 2),
    )

    return OrchestratorConfig(
        models=tuple(
            ModelSpec(
                role=role,
                model=model,
                aliases=aliases,
                description=description,
                provider_config=ProviderConfig(
                    provider=Provider.LOCAL,
                    default_model=model,
                    default_params=_local_vllm_params(
                        active_preset,
                        role=role,
                        port=active_preset.base_port + offset,
                    ),
                ),
            )
            for role, model, aliases, description, offset in specs
        ),
        default_role="general",
        recording=recording,
        mode_label=mode_label or "local",
    )


def _provider_config(
    *,
    provider: Provider | str,
    api_base: str | None,
    api_key_env: str | None,
    model: str,
    provider_defaults: dict[str, object],
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        api_base=api_base,
        api_key_env=api_key_env,
        default_model=model,
        default_params=dict(provider_defaults),
    )


def _mode_label_for(provider: Provider | str) -> str:
    normalized = provider.value if isinstance(provider, Provider) else str(provider).strip().lower()
    if normalized == Provider.LOCAL.value:
        return "local"
    return "http"


def _local_vllm_params(
    preset: LocalVLLMPresetConfig,
    *,
    role: str,
    port: int,
) -> dict[str, object]:
    params = dict(preset.provider_defaults)
    params.update(
        {
            LOCAL_LAUNCH_IMAGE: preset.image,
            LOCAL_LAUNCH_BIND: preset.bind or DEFAULT_LOCAL_VLLM_BIND,
            LOCAL_LAUNCH_STARTUP_TIMEOUT: preset.startup_timeout_seconds,
            LOCAL_LAUNCH_MAX_MODEL_LEN: preset.max_model_len,
            LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION: preset.gpu_memory_utilization,
            LOCAL_LAUNCH_PORT: port,
            LOCAL_LAUNCH_QUANTIZATION: preset.quantization,
            LOCAL_LAUNCH_USE_GPU: preset.use_gpu,
        }
    )

    if preset.client_host is not None:
        params[LOCAL_LAUNCH_CLIENT_HOST] = preset.client_host
    if preset.server_host is not None:
        params[LOCAL_LAUNCH_SERVER_HOST] = preset.server_host
    if preset.container_cache_dir is not None:
        params[LOCAL_LAUNCH_CONTAINER_CACHE_DIR] = preset.container_cache_dir
    if preset.executable is not None:
        params[LOCAL_LAUNCH_EXECUTABLE] = preset.executable
    if preset.poll_interval_seconds is not None:
        params[LOCAL_LAUNCH_POLL_INTERVAL] = preset.poll_interval_seconds
    if preset.extra_args:
        params[LOCAL_LAUNCH_EXTRA_ARGS] = tuple(preset.extra_args)
    if preset.env:
        params[LOCAL_LAUNCH_ENV] = dict(preset.env)
    if preset.dtype is not None:
        params[LOCAL_LAUNCH_DTYPE] = preset.dtype
    if preset.load_format is not None:
        params[LOCAL_LAUNCH_LOAD_FORMAT] = preset.load_format
    if preset.cpu_offload_gb is not None:
        params[LOCAL_LAUNCH_CPU_OFFLOAD_GB] = preset.cpu_offload_gb

    params.update(preset.role_overrides.get(role, {}))
    params[LOCAL_LAUNCH_PORT] = params.get(LOCAL_LAUNCH_PORT, port)
    return params
