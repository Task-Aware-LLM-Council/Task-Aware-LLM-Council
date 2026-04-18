"""
Scoped fan-out session for multi-role dispatch with swap-safe execution.

Motivation
----------
`ModelOrchestrator` treats each role as an independently-lazy client (one
vLLM container per role, own port). That works when all containers can
co-reside (API providers, multi-GPU). On a single GPU it does not: the
weights of two roles cannot fit at once, so the orchestrator must serialize
load -> infer -> unload across roles.

`FanoutSession` is the thin layer that enforces this. A policy declares
which roles it intends to use for a given fan-out and then calls
`run_role` procedurally. The session keeps at most one role resident,
swapping on role change, and cleans up on exit.

Not intended for concurrent use within a single session.

See also
--------
`scratch/plan_dedicated_synthesizer.md` for the design discussion that
motivated this layer.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING

from llm_gateway import PromptRequest

from model_orchestration.models import OrchestratorResponse

if TYPE_CHECKING:
    from model_orchestration.orchestrator import ModelOrchestrator

logger = logging.getLogger(__name__)


class FanoutSession:
    """Scoped fan-out over a declared set of roles."""

    def __init__(
        self,
        orchestrator: "ModelOrchestrator",
        declared_roles: frozenset[str],
    ) -> None:
        self._orchestrator = orchestrator
        self._declared_roles = declared_roles
        self._current_role: str | None = None
        self._closed = False

    async def __aenter__(self) -> "FanoutSession":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc is None:
                await self._unload_current()
            else:
                await self._handle_exit_error(exc)
        finally:
            self._closed = True

    async def run_role(
        self,
        role: str,
        requests: list[PromptRequest],
    ) -> list[OrchestratorResponse]:
        """Run `requests` through `role`, swapping from the current role if needed."""
        if self._closed:
            raise RuntimeError("FanoutSession already closed")

        self._check_role_declared(role)

        if self._current_role is not None and self._current_role != role:
            await self._unload_current()

        self._current_role = role
        responses: list[OrchestratorResponse] = []
        for request in requests:
            response = await self._orchestrator.run(request, target=role)
            responses.append(response)
        return responses

    # ------------------------------------------------------------------ #
    # Decision points — see /scratch/plan_dedicated_synthesizer.md       #
    # ------------------------------------------------------------------ #

    def _check_role_declared(self, role: str) -> None:
        """Strict: the role must have been declared when the session opened."""
        canonical = self._orchestrator._canonical_role(role)
        if canonical not in self._declared_roles:
            raise ValueError(
                f"Role {role!r} was not declared for this FanoutSession "
                f"(declared: {sorted(self._declared_roles)}). "
                f"Pass it to orchestrator.fanout([...]) at session open."
            )

    async def _handle_exit_error(self, exc: BaseException) -> None:
        """Unload on error: next session starts cold with known-good state."""
        logger.warning(
            "FanoutSession exiting via %s; unloading role %r to avoid stuck state",
            type(exc).__name__, self._current_role,
        )
        await self._unload_current()

    # ------------------------------------------------------------------ #

    async def _unload_current(self) -> None:
        if self._current_role is None:
            return
        managed = self._orchestrator._managed_clients.get(self._current_role)
        if managed is not None:
            await managed.close()
            logger.debug("FanoutSession unloaded role %r", self._current_role)
        self._current_role = None
