"""
P4: learned-router policy with LLM decomposition + ordered synthesis.

Two-phase GPU lifecycle (per reviewer's callout on PR #10 — see
`test_orchestartor_client.py` at the repo root for the reference):

  Phase 1 ─ specialist dispatch ─────────────────────────────────┐
    async with ModelOrchestrator(specialist_config) as spec_orch: │
        await spec_orch.load_all(max_parallel=1)                  │  ≤ 3 models
        for req in batch:                                         │  GPU-resident
            decompose → route → group into runs                   │
            dispatch runs in parallel through spec_orch           │
  ← specialist orchestrator exits, GPU memory reclaimed ──────────┘

  Phase 2 ─ synthesis ───────────────────────────────────────────┐
    async with ModelOrchestrator(synthesizer_config) as synth:    │
        await synth.load_all(max_parallel=1)                      │  1 model
        for req in batch where len(runs) > 1:                     │  GPU-resident
            synthesize_ordered(…, orchestrator=synth)              │
  ← synthesizer orchestrator exits ──────────────────────────────┘

Peak GPU residency = 3 models (never 4). Single-run requests skip Phase 2
entirely via the short-circuit in `synthesize_ordered`.

`Router` and `Decomposer` are protocols (see router.py, decomposer.py);
this file depends on the interfaces only.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any, Callable

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorConfig, OrchestratorResponse

from council_policies.decomposer import Decomposer
from council_policies.models import CouncilResponse
from council_policies.router import DispatchRun, RoutingDecision, Router, Subtask
from council_policies.synthesis import SynthesisResult, synthesize_ordered

logger = logging.getLogger(__name__)


# Factory for tests: `orchestrator_factory(config)` must return an async
# context manager yielding something with `get_client(role)` and
# `load_all(max_parallel=…)`. In production this is `ModelOrchestrator`
# itself; tests pass a lambda returning a FakeOrchestrator.
OrchestratorFactory = Callable[[OrchestratorConfig], Any]


class LearnedRouterPolicy:
    """
    P4 policy — decompose, route, dispatch, synthesize.

    Parameters
    ----------
    specialist_config:
        `OrchestratorConfig` for the three specialist models. Must register
        every role the router can emit plus the `fallback_role`. Does
        *not* register the synthesizer.
    synthesizer_config:
        `OrchestratorConfig` for the synthesizer only. Kept separate so
        Phase 2 runs after Phase 1 has torn down.
    router:
        Any `Router` — `KeywordRouter` for fixtures/tests, `LearnedRouter`
        for real dispatch.
    decomposer:
        Any `Decomposer` — `PassthroughDecomposer` for single-skill parity
        tests, `LLMDecomposer` for real multi-skill fan-out.
    fallback_role:
        Dispatched when the router signals uncertainty
        (`RoutingDecision.fallback_used=True`). Can't be validated at
        construction (no orchestrator is live yet) — validation fires
        lazily when the specialist orchestrator is entered in `run_batch`.
    synthesizer_role:
        Referee role name inside `synthesizer_config`. Must not overlap
        with any role the router can emit (self-bias guard in
        `synthesize_ordered`).
    max_subtasks:
        Hard ceiling on subtasks per request. Decomposer output is
        truncated (in `order` sequence) if it exceeds this.
    specialist_load_parallel:
        Passed as `max_parallel` to `spec_orch.load_all()`. Default 1
        matches `test_orchestartor_client.py` — loading specialists
        sequentially keeps peak GPU memory bounded during warmup.
    orchestrator_factory:
        Test seam. Production passes `ModelOrchestrator` (the default);
        tests pass a factory returning a fake async context manager.
    """

    def __init__(
        self,
        *,
        specialist_config: OrchestratorConfig,
        synthesizer_config: OrchestratorConfig,
        router: Router,
        decomposer: Decomposer,
        fallback_role: str = "fact_general",
        synthesizer_role: str = "synthesizer",
        max_subtasks: int = 4,
        specialist_load_parallel: int = 1,
        orchestrator_factory: OrchestratorFactory = ModelOrchestrator,
    ) -> None:
        if max_subtasks < 1:
            raise ValueError(f"max_subtasks must be >= 1, got {max_subtasks}")
        if specialist_load_parallel < 1:
            raise ValueError(
                f"specialist_load_parallel must be >= 1, got {specialist_load_parallel}"
            )

        self.specialist_config = specialist_config
        self.synthesizer_config = synthesizer_config
        self.router = router
        self.decomposer = decomposer
        self.fallback_role = fallback_role
        self.synthesizer_role = synthesizer_role
        self.max_subtasks = max_subtasks
        self.specialist_load_parallel = specialist_load_parallel
        self._orchestrator_factory = orchestrator_factory

        # Fail fast on misconfigured role names — inspect the config specs
        # without starting any orchestrator. An orchestrator isn't live yet,
        # so we walk `config.models` to build the alias set. This is the
        # equivalent of the P3 constructor check, adapted for the two-config
        # world.
        self._assert_role_registered(
            specialist_config, "fallback_role", fallback_role
        )
        self._assert_role_registered(
            synthesizer_config, "synthesizer_role", synthesizer_role
        )

    async def run_council(self, request: PromptRequest) -> CouncilResponse:
        """Single-question path. Pays two cold-starts (specialist + synth).
        For throughput, use `run_batch` — it amortizes both phases across
        the batch."""
        responses = await self.run_batch_council([request])
        return responses[0]

    async def run(self, request: PromptRequest) -> OrchestratorResponse:
        return (await self.run_council(request)).winner

    async def run_batch(self, requests: list[PromptRequest]) -> list[OrchestratorResponse]:
        responses = await self.run_batch_council(requests)
        return [response.winner for response in responses]

    async def run_batch_council(self, requests: list[PromptRequest]) -> list[CouncilResponse]:
        """Batched two-phase execution; see module docstring."""
        if not requests:
            return []

        # ---- Phase 1: specialist dispatch ----
        batched_subtasks: list[list[Subtask]] = []
        batched_decisions: list[list[RoutingDecision]] = []
        batched_runs: list[list[DispatchRun]] = []

        async with self._orchestrator_factory(self.specialist_config) as spec_orch:
            await spec_orch.load_all(max_parallel=self.specialist_load_parallel)

            for req in requests:
                subs, decisions, runs = await self._dispatch_one(req, spec_orch)
                batched_subtasks.append(subs)
                batched_decisions.append(decisions)
                batched_runs.append(runs)
        # ← specialists released here

        # ---- Phase 2: synthesize multi-run requests ----
        needs_synth = [i for i, runs in enumerate(batched_runs) if len(runs) > 1]
        synth_results: dict[int, SynthesisResult] = {}

        if needs_synth:
            async with self._orchestrator_factory(self.synthesizer_config) as synth_orch:
                await synth_orch.load_all(max_parallel=1)

                for i in needs_synth:
                    synth_results[i] = await synthesize_ordered(
                        question=requests[i].user_prompt or "",
                        runs=batched_runs[i],
                        orchestrator=synth_orch,
                        synthesizer_role=self.synthesizer_role,
                    )
        # ← synthesizer released here

        # ---- Assemble CouncilResponses ----
        return [
            self._build_response(
                runs=batched_runs[i],
                subtasks=batched_subtasks[i],
                decisions=batched_decisions[i],
                synth_result=synth_results.get(i),
            )
            for i in range(len(requests))
        ]

    async def _dispatch_one(
        self,
        request: PromptRequest,
        orchestrator: ModelOrchestrator,
    ) -> tuple[list[Subtask], list[RoutingDecision], list[DispatchRun]]:
        prompt_text = request.user_prompt or ""
        context_text = request.context or ""

        # 1. Decompose.
        subtasks = await self.decomposer.decompose(prompt_text, context=context_text)
        if not subtasks:
            logger.warning("Decomposer returned no subtasks; falling back to single passthrough")
            subtasks = [Subtask(text=prompt_text, order=0)]

        # Sort by order (trust the field, not list position) and cap.
        subtasks = sorted(subtasks, key=lambda s: s.order)[: self.max_subtasks]

        # 2. Classify each subtask.
        decisions = [
            self.router.classify(s.text, context=context_text) for s in subtasks
        ]

        # 3. Group into contiguous same-role runs. Non-adjacent duplicates
        # stay in separate runs so interleaved [A, B, A] ordering survives.
        runs: list[DispatchRun] = []
        for sub, decision in zip(subtasks, decisions):
            if runs and runs[-1].role == decision.role:
                runs[-1].append(sub)
            else:
                run = DispatchRun(role=decision.role)
                run.append(sub)
                runs.append(run)

        # 4. Dispatch runs in parallel through the specialist orchestrator.
        async def _dispatch(run: DispatchRun) -> OrchestratorResponse:
            sub_request = replace(request, user_prompt=run.rendered_prompt())
            return await orchestrator.get_client(run.role).get_response(sub_request)

        responses = await asyncio.gather(*(_dispatch(r) for r in runs))
        for run, resp in zip(runs, responses):
            run.response = resp

        return subtasks, decisions, runs

    def _build_response(
        self,
        *,
        runs: list[DispatchRun],
        subtasks: list[Subtask],
        decisions: list[RoutingDecision],
        synth_result: SynthesisResult | None,
    ) -> CouncilResponse:
        # Single-run: short-circuit — specialist response is the answer,
        # no synthesizer call was made.
        if synth_result is None:
            winner = runs[0].response
            synth_short_circuit: str | None = "single_run"
            synth_used_fallback = False
        else:
            winner = synth_result.response
            synth_short_circuit = synth_result.metadata.get("short_circuit")
            synth_used_fallback = synth_result.used_fallback

        specialist_responses = tuple(
            r.response for r in runs if r.response is not None
        )

        return CouncilResponse(
            winner=winner,  # type: ignore[arg-type]  # None only if dispatch failed, which would have raised
            policy="p4",
            task_type=None,
            candidates=specialist_responses,
            metadata={
                "subtasks": [
                    {"order": s.order, "text": s.text} for s in subtasks
                ],
                "runs": [
                    {
                        "role": r.role,
                        "subtask_orders": [s.order for s in r.subtasks],
                    }
                    for r in runs
                ],
                "router_scores": [d.scores for d in decisions],
                "router_confidence": [d.confidence for d in decisions],
                "router_fallback_used": [d.fallback_used for d in decisions],
                "synthesizer_role": self.synthesizer_role,
                "synthesis_short_circuit": synth_short_circuit,
                "synthesis_used_fallback": synth_used_fallback,
            },
        )

    @staticmethod
    def _assert_role_registered(
        config: OrchestratorConfig,
        field_name: str,
        role: str,
    ) -> None:
        """Verify a role is present in a config's specs (role name OR alias)
        without starting the orchestrator. Mirrors the old P3 constructor
        check adapted for the specialist/synthesizer split."""
        known: set[str] = set()
        for spec in config.models:
            known.add(str(spec.role))
            known.update(str(a) for a in spec.aliases)
        if role not in known:
            raise ValueError(
                f"LearnedRouterPolicy: {field_name} {role!r} is not "
                f"registered in the provided config. Known roles/aliases: "
                f"{sorted(known) or '(none)'}."
            )
