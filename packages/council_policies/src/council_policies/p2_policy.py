from __future__ import annotations

import asyncio
import logging

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorResponse

from council_policies.models import CouncilResponse
from council_policies.voter import run_vote

logger = logging.getLogger(__name__)

_DEFAULT_COUNCIL_ROLES = ("qa", "reasoning", "general")


class FlatCouncilPolicy:
    """
    P2: Flat council — send the prompt to all member models in parallel,
    then run a majority vote to select the winning answer.
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        *,
        council_roles: tuple[str, ...] = _DEFAULT_COUNCIL_ROLES,
    ) -> None:
        self.orchestrator = orchestrator
        self.council_roles = council_roles

    async def run(self, request: PromptRequest) -> CouncilResponse:
        answer_tasks = [
            self.orchestrator.get_client(role).get_response(request)
            for role in self.council_roles
        ]
        results = await asyncio.gather(*answer_tasks, return_exceptions=True)

        candidates: dict[str, OrchestratorResponse] = {}
        for role, result in zip(self.council_roles, results):
            if isinstance(result, Exception):
                logger.warning("Council member %r failed: %s", role, result)
            else:
                candidates[role] = result

        if not candidates:
            raise RuntimeError("All council members failed to respond")

        if len(candidates) == 1:
            only_role = next(iter(candidates))
            return CouncilResponse(
                winner=candidates[only_role],
                policy="p2",
                candidates=tuple(candidates.values()),
                vote_tally={only_role: 1},
            )

        question = request.user_prompt or ""
        winner_role, tally = await run_vote(
            question=question,
            candidates=candidates,
            orchestrator=self.orchestrator,
            voter_roles=tuple(candidates.keys()),
        )

        return CouncilResponse(
            winner=candidates[winner_role],
            policy="p2",
            candidates=tuple(candidates.values()),
            vote_tally=tally,
        )
