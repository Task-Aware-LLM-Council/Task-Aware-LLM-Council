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

ApptainerServerConfig = VLLMRuntimeConfig
ApptainerServerHandle = VLLMRuntime


def build_apptainer_server_config(config):
    return build_vllm_runtime_config(config)
