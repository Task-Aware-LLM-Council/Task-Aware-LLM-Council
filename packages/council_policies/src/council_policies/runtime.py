from __future__ import annotations

import asyncio
from typing import Sequence

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator
from model_orchestration.models import OrchestratorConfig

from benchmarking_pipeline.models import BenchmarkExample

from council_policies.models import PolicyEvaluationResult
from council_policies.p2.policy import _majority, _parse_vote
from council_policies.p2.prompts import ROLE_TO_LABEL, build_vote_prompt

_ROLES = ("qa", "reasoning", "general")


class PolicyRuntime:
    """
    API-based parallel execution for P2.

    Phase 1 — all 3 specialist models answer all examples in parallel.
    Phase 2 — all 3 specialist models vote on all examples in parallel.

    The vote winner is written as the final benchmark output.
    """

    def __init__(
        self,
        specialist_config: OrchestratorConfig,
        synthesizer_config: OrchestratorConfig | None = None,
        max_concurrency: int = 10,
    ) -> None:
        self.specialist_config = specialist_config
        self.synthesizer_config = synthesizer_config
        self.max_concurrency = max_concurrency

    async def run_batch(
        self,
        examples: Sequence[BenchmarkExample],
        dataset_name: str,
    ) -> list[PolicyEvaluationResult]:
        examples = list(examples)
        sem = asyncio.Semaphore(self.max_concurrency)

        async with ModelOrchestrator(self.specialist_config) as specialist_orch:
            # Phase 1: All 3 models answer all examples in parallel
            role_answers = await _collect_all_answers(specialist_orch, examples, sem)

            # Build per-example {label: answer} mapping (A=qa, B=reasoning, C=general)
            label_to_answers = [
                {ROLE_TO_LABEL[role]: role_answers[role][i] for role in _ROLES}
                for i in range(len(examples))
            ]

            # Phase 2: All 3 models vote on all examples in parallel
            role_votes = await _collect_all_votes(specialist_orch, examples, label_to_answers, sem)

        # Determine the winning label per example
        per_example_votes = [
            {role: role_votes[role][i] for role in _ROLES}
            for i in range(len(examples))
        ]
        winners = [_majority(votes) for votes in per_example_votes]
        winning_answers = [label_to_answers[i][winners[i]] for i in range(len(examples))]

        return [
            PolicyEvaluationResult(
                example_id=examples[i].example_id,
                dataset_name=dataset_name,
                output=winning_answers[i],
                status="success",
                metadata={
                    "answer_a": label_to_answers[i]["A"],
                    "answer_b": label_to_answers[i]["B"],
                    "answer_c": label_to_answers[i]["C"],
                    "votes": per_example_votes[i],
                    "winning_label": winners[i],
                    "winning_answer": winning_answers[i],
                    "candidate_answers": label_to_answers[i],
                },
            )
            for i in range(len(examples))
        ]


async def _collect_all_answers(
    orch: ModelOrchestrator,
    examples: list[BenchmarkExample],
    sem: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Run all 3 specialist models in parallel, returning {role: [answers]}."""
    results = await asyncio.gather(*[
        _answers_for_role(orch, examples, role, sem) for role in _ROLES
    ])
    return dict(zip(_ROLES, results))


async def _answers_for_role(
    orch: ModelOrchestrator,
    examples: list[BenchmarkExample],
    role: str,
    sem: asyncio.Semaphore,
) -> list[str]:
    async def answer_one(ex: BenchmarkExample) -> str:
        async with sem:
            req = PromptRequest(
                user_prompt=ex.question or "",
                context=ex.context,
                system_prompt=ex.system_prompt,
                temperature=0.0,
            )
            try:
                resp = await orch.run(req, target=role)
                return resp.text.strip()
            except Exception:
                return ""

    return list(await asyncio.gather(*[answer_one(ex) for ex in examples]))


async def _collect_all_votes(
    orch: ModelOrchestrator,
    examples: list[BenchmarkExample],
    label_to_answers: list[dict[str, str]],
    sem: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Run all 3 specialist models in parallel to vote, returning {role: [vote labels]}."""
    results = await asyncio.gather(*[
        _votes_for_role(orch, examples, label_to_answers, role, sem) for role in _ROLES
    ])
    return dict(zip(_ROLES, results))


async def _votes_for_role(
    orch: ModelOrchestrator,
    examples: list[BenchmarkExample],
    label_to_answers: list[dict[str, str]],
    role: str,
    sem: asyncio.Semaphore,
) -> list[str]:
    async def vote_one(ex: BenchmarkExample, lta: dict[str, str]) -> str:
        async with sem:
            req = PromptRequest(
                user_prompt=build_vote_prompt(ex.question or "", lta),
                temperature=0.0,
            )
            try:
                resp = await orch.run(req, target=role)
                return _parse_vote(resp.text)
            except Exception:
                return "A"

    return list(await asyncio.gather(*[vote_one(ex, lta) for ex, lta in zip(examples, label_to_answers)]))
