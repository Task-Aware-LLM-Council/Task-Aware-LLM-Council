from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable

from llm_gateway import Provider, ProviderConfig, create_client, managed_local_provider_config


@dataclass(slots=True)
class ResolvedProviderHandle:
    provider_config: ProviderConfig
    close: Callable[[], Awaitable[None]]


@asynccontextmanager
async def resolve_provider_config(
    provider_config: ProviderConfig,
    *,
    model: str,
):
    normalized_provider = _normalize_provider_name(provider_config.provider)
    if normalized_provider != Provider.LOCAL.value:
        yield ResolvedProviderHandle(provider_config=provider_config, close=_noop_close)
        return

    async with managed_local_provider_config(provider_config, model=model) as resolved:
        yield ResolvedProviderHandle(provider_config=resolved, close=_noop_close)


def build_client(provider_config: ProviderConfig):
    return create_client(provider_config)


async def _noop_close() -> None:
    return None


def _normalize_provider_name(provider: Provider | str) -> str:
    if isinstance(provider, Provider):
        return provider.value
    return str(provider).strip().lower()
