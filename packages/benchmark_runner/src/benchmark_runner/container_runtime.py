from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any
from urllib import error, request

from llm_gateway import Provider, ProviderConfig


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


@dataclass(slots=True, frozen=True)
class ApptainerServerConfig:
    image: str
    use_gpu: bool = True
    bind: str | None = None
    container_cache_dir: str = "/scratch1/bmahajan/.cache/huggingface"
    client_host: str = "127.0.0.1"
    server_host: str = "0.0.0.0"
    port: int = 8000
    executable: str = "apptainer"
    startup_timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


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


def build_apptainer_server_config(config: ProviderConfig) -> ApptainerServerConfig | None:
    if _normalize_provider_name(config.provider) != Provider.LOCAL.value:
        return None

    image = config.default_params.get(LOCAL_LAUNCH_IMAGE)
    print("config.provider", image)
    if not image:
        return None

    extra_args_raw = config.default_params.get(LOCAL_LAUNCH_EXTRA_ARGS, ())
    if isinstance(extra_args_raw, str):
        extra_args = (extra_args_raw,)
    else:
        extra_args = tuple(str(item) for item in extra_args_raw)

    env_raw = config.default_params.get(LOCAL_LAUNCH_ENV, {})
    env = {str(key): str(value) for key, value in dict(env_raw).items()}

    return ApptainerServerConfig(
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
                "/scratch1/bmahajan/.cache/huggingface",
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
    )


class ApptainerServerHandle:
    def __init__(self, config: ApptainerServerConfig) -> None:
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
                    f"apptainer server already running for {self._loaded_model!r}; "
                    f"stop it before switching to {model!r}"
                )
            return self.api_base

        self._process = await asyncio.create_subprocess_exec(
            *self._build_command(model),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._build_env(),
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
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        else:
            await process.wait()

    async def _wait_until_ready(self) -> None:
        deadline = monotonic() + self.config.startup_timeout_seconds
        models_url = self.api_base.removesuffix("/chat/completions").rstrip("/") + "/models"

        while monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    f"apptainer server exited before becoming ready with code {self._process.returncode}"
                )
            if await asyncio.to_thread(self._probe_ready, models_url):
                return
            await asyncio.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(
            f"apptainer server did not become ready within "
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
        if self.config.bind:
            bind_target = f"{self.config.bind}:{self.config.container_cache_dir}"
            command.extend(["--bind", bind_target])
        command.extend(
            [
                self.config.image,
                "--model",
                model,
                "--host",
                self.config.server_host,
                "--port",
                str(self.config.port),
            ]
        )
        command.extend(self.config.extra_args)
        return tuple(command)


@contextlib.asynccontextmanager
async def managed_local_provider_config(
    config: ProviderConfig,
    *,
    model: str,
):
    print(f"config:{config}, model:{model}")
    normalized_config = normalize_local_provider_config(config)
    launch_config = build_apptainer_server_config(config)
    print(f"launch_config:{launch_config}")
    if launch_config is None:
        yield normalized_config
        return

    handle = ApptainerServerHandle(launch_config)
    print(f"handle:{handle}")
    try:
        api_base = await handle.start(model)
        yield replace(normalized_config, api_base=api_base)
    finally:
        await handle.stop()


def _normalize_provider_name(provider: Provider | str) -> str:
    if isinstance(provider, Provider):
        return provider.value
    return str(provider).strip().lower()
