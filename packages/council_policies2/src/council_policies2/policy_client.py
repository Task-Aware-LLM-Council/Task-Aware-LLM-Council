from __future__ import annotations

"""
PolicyClient — bridges council_policies2 and benchmarking_pipeline.

The benchmarking_pipeline's ``run_benchmark`` function accepts a
``client: BaseLLMClient``.  This module provides a ``PolicyClient`` that
implements that interface by delegating every ``generate`` call to a council
policy (P2, P3, or P4).

Usage
-----
    from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
    from council_policies2 import RuleBasedRoutingPolicy
    from council_policies2.policy_client import PolicyClient
    from benchmarking_pipeline import run_benchmark, BenchmarkRunConfig

    config = build_default_orchestrator_config(...)
    async with ModelOrchestrator(config) as orchestrator:
        policy  = RuleBasedRoutingPolicy(orchestrator)
        client  = PolicyClient(policy)
        result  = await run_benchmark(datasets, run_config, client=client)

The ``PolicyClient`` passes the raw ``PromptRequest`` straight to the policy's
``run()`` method and returns the winning ``PromptResponse`` so the rest of the
pipeline (scoring, storage, etc.) is completely unaware of the council layer.
"""

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from llm_gateway import BaseLLMClient, ClientInfo, PromptRequest, PromptResponse

logger = logging.getLogger(__name__)


@runtime_checkable
class CouncilPolicy(Protocol):
    """Structural protocol satisfied by FlatCouncilPolicy, RuleBasedRoutingPolicy, and LearnedRouterPolicy."""

    async def run(self, request: PromptRequest) -> Any:
        """Run the policy and return a CouncilResponse."""
        ...


class PolicyClient(BaseLLMClient):
    """
    A ``BaseLLMClient`` wrapper that routes every ``generate`` call through
    a council policy.

    This is the glue layer that connects ``council_policies2`` to the
    ``benchmarking_pipeline`` without modifying either package.

    Parameters
    ----------
    policy:
        Any policy object with an async ``run(request) -> CouncilResponse``
        method.  All three policy classes in this package qualify.
    provider_label:
        An optional string used in log messages and stored in
        ``PromptResponse.provider`` to identify which policy was active.
        Defaults to ``"council"``.
    """

    def __init__(
        self,
        policy: CouncilPolicy,
        *,
        provider_label: str = "council",
    ) -> None:
        super().__init__(
            ClientInfo(
                provider=provider_label,
                # No single default model — the policy resolves models internally.
                default_model=None,
                # The policy's constituent clients handle their own timeouts.
                timeout_seconds=0.0,
            )
        )
        self._policy = policy
        self._provider_label = provider_label

    def validate_request(self, request: PromptRequest) -> None:
        """
        Override the base validation.

        Unlike a raw LLM client, the PolicyClient doesn't require a ``model``
        field on the request — the policy picks the right model(s) itself.
        We only check that there is some prompt content to work with.
        """
        if not request.resolved_messages():
            from llm_gateway import LLMRequestError
            raise LLMRequestError("PolicyClient: request must include prompt content")

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """
        Route ``request`` through the council policy and return the winning
        ``PromptResponse``.

        The ``CouncilResponse.winner`` field holds an ``OrchestratorResponse``
        whose ``.prompt_response`` is a ``PromptResponse`` — exactly what the
        benchmarking pipeline expects.
        """
        self.ensure_open()
        council_response = await self._policy.run(request)
        winning: PromptResponse = council_response.winner.prompt_response

        # Annotate the response so downstream consumers know which policy ran.
        from dataclasses import replace
        metadata = dict(winning.metadata)
        metadata["council_policy"] = council_response.policy
        if council_response.task_type is not None:
            metadata["council_task_type"] = council_response.task_type.value
        if council_response.vote_tally:
            metadata["vote_tally"] = dict(council_response.vote_tally)
        metadata.update(council_response.metadata)

        return replace(
            winning,
            provider=self._provider_label,
            metadata=metadata,
        )
