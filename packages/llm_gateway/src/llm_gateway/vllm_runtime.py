from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass, field, replace
from time import monotonic
from urllib import error, request

from common import get_current_user

from llm_gateway.models import Provider, ProviderConfig


LOCAL_LAUNCH_PREFIX = "local_launch_"
LOCAL_LAUNCH_IMAGE = f"{LOCAL_LAUNCH_PREFIX}image"
LOCAL_LAUNCH_USE_GPU = f"{LOCAL_LAUNCH_PREFIX}use_gpu"
LOCAL_LAUNCH_BIND = f"{LOCAL_LAUNCH_PREFIX}bind"
LOCAL_LAUNCH_CONTAINER_CACHE_DIR = f"{LOCAL_LAUNCH_PREFIX}container_cache_dir"
LOCAL_LAUNCH_CLIENT_HOST = f"{LOCAL_LAUNCH_PREFIX}client_host"
LOCAL_LAUNCH_SERVER_HOST = f"{LOCAL_LAUNCH_PREFIX}server_host"
LOCAL_LAUNCH_PORT = f"{LOCAL_LAUNCH_PREFIX}port"
LOCAL_LAUNCH_EXECUTABLE = f"{LOCAL_LAUNCH_PREFIX}executable"
LOCAL_LAUNCH_STARTUP_TIMEOUT = f"{LOCAL_LAUNCH_PREFIX}startup_timeout_seconds"
LOCAL_LAUNCH_POLL_INTERVAL = f"{LOCAL_LAUNCH_PREFIX}poll_interval_seconds"
LOCAL_LAUNCH_EXTRA_ARGS = f"{LOCAL_LAUNCH_PREFIX}extra_args"
LOCAL_LAUNCH_ENV = f"{LOCAL_LAUNCH_PREFIX}env"
LOCAL_LAUNCH_QUANTIZATION = f"{LOCAL_LAUNCH_PREFIX}quantization"
LOCAL_LAUNCH_LOAD_FORMAT = f"{LOCAL_LAUNCH_PREFIX}load_format"
LOCAL_LAUNCH_DTYPE = f"{LOCAL_LAUNCH_PREFIX}dtype"
LOCAL_LAUNCH_MAX_MODEL_LEN = f"{LOCAL_LAUNCH_PREFIX}max_model_len"
LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION = f"{LOCAL_LAUNCH_PREFIX}gpu_memory_utilization"
LOCAL_LAUNCH_CPU_OFFLOAD_GB = f"{LOCAL_LAUNCH_PREFIX}cpu_offload_gb"


@dataclass(slots=True, frozen=True)
class VLLMRuntimeConfig:
    image: str
    use_gpu: bool = True
    bind: str | None = None
    container_cache_dir: str = f"/scratch1/{get_current_user()}/.cache/huggingface"
    client_host: str = "127.0.0.1"
    server_host: str = "0.0.0.0"
    port: int = 8000
    executable: str = "apptainer"
    startup_timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    quantization: str | None = None
    load_format: str | None = None
    dtype: str | None = None
    max_model_len: str | None = None
    gpu_memory_utilization: float | None = None
    cpu_offload_gb: float | None = None


def normalize_local_provider_config(config: ProviderConfig) -> ProviderConfig:
    if _normalize_provider_name(config.provider) != Provider.LOCAL.value:
        return config
    return replace(
        config,
        default_params={
            key: value
            for key, value in config.default_params.items()
            if not key.startswith(LOCAL_LAUNCH_PREFIX)
        },
    )


def build_vllm_runtime_config(config: ProviderConfig) -> VLLMRuntimeConfig | None:
    if _normalize_provider_name(config.provider) != Provider.LOCAL.value:
        return None

    image = config.default_params.get(LOCAL_LAUNCH_IMAGE)
    if not image:
        return None

    extra_args_raw = config.default_params.get(LOCAL_LAUNCH_EXTRA_ARGS, ())
    if isinstance(extra_args_raw, str):
        extra_args = (extra_args_raw,)
    else:
        extra_args = tuple(str(item) for item in extra_args_raw)

    env_raw = config.default_params.get(LOCAL_LAUNCH_ENV, {})
    env = {str(key): str(value) for key, value in dict(env_raw).items()}

    return VLLMRuntimeConfig(
        image=str(image),
        use_gpu=bool(config.default_params.get(LOCAL_LAUNCH_USE_GPU, True)),
        bind=(
            str(config.default_params[LOCAL_LAUNCH_BIND])
            if config.default_params.get(LOCAL_LAUNCH_BIND) is not None
            else None
        ),
        container_cache_dir=str(
            config.default_params.get(
                LOCAL_LAUNCH_CONTAINER_CACHE_DIR,
                f"/scratch1/{get_current_user()}/.cache/huggingface",
            )
        ),
        client_host=str(config.default_params.get(LOCAL_LAUNCH_CLIENT_HOST, "127.0.0.1")),
        server_host=str(config.default_params.get(LOCAL_LAUNCH_SERVER_HOST, "0.0.0.0")),
        port=int(config.default_params.get(LOCAL_LAUNCH_PORT, 8000)),
        executable=str(config.default_params.get(LOCAL_LAUNCH_EXECUTABLE, "apptainer")),
        startup_timeout_seconds=float(
            config.default_params.get(LOCAL_LAUNCH_STARTUP_TIMEOUT, 120.0)
        ),
        poll_interval_seconds=float(
            config.default_params.get(LOCAL_LAUNCH_POLL_INTERVAL, 1.0)
        ),
        extra_args=extra_args,
        env=env,
        quantization=config.default_params.get(LOCAL_LAUNCH_QUANTIZATION),
        load_format=config.default_params.get(LOCAL_LAUNCH_LOAD_FORMAT),
        dtype=config.default_params.get(LOCAL_LAUNCH_DTYPE),
        max_model_len=config.default_params.get(LOCAL_LAUNCH_MAX_MODEL_LEN),
        gpu_memory_utilization=(
            float(config.default_params[LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION])
            if config.default_params.get(LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION) is not None
            else None
        ),
        cpu_offload_gb=(
            float(config.default_params[LOCAL_LAUNCH_CPU_OFFLOAD_GB])
            if config.default_params.get(LOCAL_LAUNCH_CPU_OFFLOAD_GB) is not None
            else None
        ),
    )


class VLLMRuntime:
    def __init__(self, config: VLLMRuntimeConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._loaded_model: str | None = None

    @property
    def api_base(self) -> str:
        return f"http://{self.config.client_host}:{self.config.port}/v1/chat/completions"

    async def start(self, model: str) -> str:
        if self._process is not None:
            if self._loaded_model != model:
                raise RuntimeError(
                    f"vLLM runtime already running for {self._loaded_model!r}; "
                    f"stop it before switching to {model!r}"
                )
            return self.api_base

        self._process = await asyncio.create_subprocess_exec(
            *self._build_command(model),
            env=self._build_env(),
            start_new_session=True,
        )
        self._loaded_model = model

        try:
            await self._wait_until_ready()
        except Exception:
            await self.stop()
            raise
        return self.api_base

    async def stop(self) -> None:
        if self._process is None:
            self._loaded_model = None
            return

        process = self._process
        self._process = None
        self._loaded_model = None

        if process.returncode is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        else:
            await process.wait()

    async def resolved_provider_config(
        self,
        base_config: ProviderConfig,
        *,
        model: str,
    ) -> ProviderConfig:
        api_base = await self.start(model)
        return replace(normalize_local_provider_config(base_config), api_base=api_base)

    async def _wait_until_ready(self) -> None:
        deadline = monotonic() + self.config.startup_timeout_seconds
        models_url = self.api_base.removesuffix("/chat/completions").rstrip("/") + "/models"

        while monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    f"vLLM runtime exited before becoming ready with code {self._process.returncode}"
                )
            if await asyncio.to_thread(self._probe_ready, models_url):
                return
            await asyncio.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(
            f"vLLM runtime did not become ready within "
            f"{self.config.startup_timeout_seconds} seconds"
        )

    @staticmethod
    def _probe_ready(url: str) -> bool:
        try:
            with request.urlopen(url, timeout=5) as response:
                return 200 <= response.status < 300
        except (error.URLError, TimeoutError, ValueError):
            return False

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.env)
        return env

    def _build_command(self, model: str) -> tuple[str, ...]:
        command: list[str] = [self.config.executable, "run"]
        if self.config.use_gpu:
            command.append("--nv")
        command.append("--cleanenv")

        host_bind_path = self.config.bind or self.config.container_cache_dir
        command.extend(["--bind", f"{host_bind_path}:{self.config.container_cache_dir}"])
        command.extend(["--env", f"HF_HOME={self.config.container_cache_dir}"])

        hf_token = self.config.env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            command.extend(["--env", f"HF_TOKEN={hf_token}"])

        command.extend(
            [
                self.config.image,
                "--model",
                model,
                "--trust-remote-code",
                "--host",
                self.config.server_host,
                "--port",
                str(self.config.port),
            ]
        )

        if self.config.quantization:
            command.extend(["--quantization", self.config.quantization])
        if self.config.load_format:
            command.extend(["--load-format", self.config.load_format])
        if self.config.dtype:
            command.extend(["--dtype", self.config.dtype])
        if self.config.max_model_len:
            command.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(self.config.gpu_memory_utilization)])
        if self.config.cpu_offload_gb is not None:
            command.extend(["--cpu-offload-gb", str(self.config.cpu_offload_gb)])

        command.extend(self.config.extra_args)
        return tuple(command)


@contextlib.asynccontextmanager
async def managed_local_provider_config(
    config: ProviderConfig,
    *,
    model: str,
):
    normalized_config = normalize_local_provider_config(config)
    runtime_config = build_vllm_runtime_config(config)
    if runtime_config is None:
        yield normalized_config
        return

    runtime = VLLMRuntime(runtime_config)
    try:
        yield await runtime.resolved_provider_config(config, model=model)
    finally:
        await runtime.stop()


def _normalize_provider_name(provider: Provider | str) -> str:
    if isinstance(provider, Provider):
        return provider.value
    return str(provider).strip().lower()
