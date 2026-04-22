"""
P4: learned-router policy with LLM decomposition + ordered synthesis.

Adapter-shaped. The policy exposes `plan()` / `finalize()` so
`CouncilBenchmarkRunner` can batch its specialist dispatch alongside
other policies in Phase 1, then drive synthesis in Phase 2.

Two-phase GPU residency is enforced by the runner, not this class:

  Phase 1 ─ specialist dispatch (runner opens specialist_orchestrator) ─┐
    plan() declares SpecialistRequests → runner executes them in        │
    parallel across every policy in the batch → runner closes the       │  ≤ N specialist models
    specialist orchestrator, reclaiming GPU memory.                     │  GPU-resident
  ← specialist orchestrator closed ───────────────────────────────────┘

  Phase 2 ─ synthesis (runner only opens synthesizer if needed) ────────┐
    finalize() calls runtime.get_synthesizer_orchestrator() only when   │
    len(runs) > 1 — single-run requests short-circuit and never touch   │  1 synthesizer model
    the synthesizer. Multiple multi-run requests share the one opened   │  GPU-resident
    synthesizer orchestrator.                                           │
  ← synthesizer orchestrator closed ──────────────────────────────────┘

Peak GPU residency = specialist config's worth of models (never the
synthesizer simultaneously).

Role validation is *lazy* and lives in `plan()`:
  - primary router role absent + fallback_role also absent → RuntimeError
  - synthesizer_role absent AND len(runs) > 1 → RuntimeError

This replaces the old eager __init__ check so construction can happen
before any orchestrator config is fully wired up. Regression guard
ported from the P3 fix in commit 053ce82.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from llm_gateway import PromptRequest

from council_policies.decomposer import Decomposer, PassthroughDecomposer
from council_policies.policy_runner import (
    BasePolicyAdapter,
    PolicyExecutionState,
    PolicyMetrics,
    PolicyResult,
    PolicyRuntime,
    SpecialistCache,
    SpecialistRequest,
    aggregate_specialist_metrics,
    build_run_from_cached_response,
    build_specialist_cache_key,
    make_policy_result,
    orchestrator_to_prompt_response,
    sum_usage,
)
from council_policies.router import DispatchRun, RoutingDecision, Router, Subtask
from council_policies.router_labels import ROLE_LABELS
from council_policies.synthesis import synthesize_ordered

logger = logging.getLogger(__name__)


class LearnedRouterPolicy(BasePolicyAdapter):
    """
    P4 — learned-router policy with decomposition + ordered synthesis.

    Parameters
    ----------
    router:
        Any `Router`. `KeywordRouter` for fixtures/tests,
        `LearnedRouter` for real dispatch.
    decomposer:
        Any `Decomposer`. Defaults to `PassthroughDecomposer` so a
        caller with no multi-skill decomposition still gets P3-like
        single-subtask behavior.
    fallback_role:
        Dispatched when the router's role isn't registered on the
        specialist orchestrator. If the fallback is also missing,
        `plan()` raises `RuntimeError` naming both roles.
    synthesizer_role:
        Referee role on the synthesizer orchestrator. Must not collide
        with any role the router can emit (self-bias guard in
        `synthesize_ordered`). Checked lazily in `plan()` only when
        synthesis will actually fire (len(runs) > 1).
    max_subtasks:
        Hard ceiling on subtasks per request. Decomposer output is
        sorted by `order` then truncated if it exceeds this.
    """

    policy_id = "p4"

    def __init__(
        self,
        *,
        router: Router | None = None,
        decomposer: Decomposer | None = None,
        fallback_role: str = "fact_general",
        synthesizer_role: str = "synthesizer",
        max_subtasks: int = 4,
        use_joint_model: bool = False,
    ) -> None:
        if max_subtasks < 1:
            raise ValueError(f"max_subtasks must be >= 1, got {max_subtasks}")
        if use_joint_model:
            if router is not None:
                raise ValueError(
                    "LearnedRouterPolicy(use_joint_model=True): `router` "
                    "must be None — the joint decomposer emits routing "
                    "decisions directly via Subtask.suggested_role."
                )
            if decomposer is None:
                raise ValueError(
                    "LearnedRouterPolicy(use_joint_model=True): `decomposer` "
                    "is required — pass a Seq2SeqDecomposerRouter (or any "
                    "Decomposer that populates Subtask.suggested_role)."
                )
        else:
            if router is None:
                raise ValueError(
                    "LearnedRouterPolicy(use_joint_model=False): `router` "
                    "is required. Pass `use_joint_model=True` only when "
                    "routing is handled by the decomposer itself."
                )
        self.router = router
        self.decomposer = decomposer or PassthroughDecomposer()
        self.fallback_role = fallback_role
        self.synthesizer_role = synthesizer_role
        self.max_subtasks = max_subtasks
        self.use_joint_model = use_joint_model

    async def plan(
        self,
        request: PromptRequest,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState:
        prompt_text = request.user_prompt or ""
        context_text = request.context or ""

        subtasks = await self.decomposer.decompose(prompt_text, context=context_text)
        if not subtasks:
            logger.warning(
                "Decomposer returned no subtasks; falling back to single passthrough"
            )
            subtasks = [Subtask(text=prompt_text, order=0)]
        subtasks = sorted(subtasks, key=lambda s: s.order)[: self.max_subtasks]

        decisions: list[RoutingDecision] = []
        runs: list[DispatchRun] = []
        for subtask in subtasks:
            if self.use_joint_model:
                decision = self._decision_from_joint_model(subtask)
            else:
                decision = self.router.classify(subtask.text, context=context_text)
            role = self._resolve_role(decision.role, runtime)
            resolved_decision = RoutingDecision(
                role=role,
                scores=decision.scores,
                confidence=decision.confidence,
                fallback_used=decision.fallback_used or role == self.fallback_role,
                fallback_reason=decision.fallback_reason,
            )
            decisions.append(resolved_decision)
            if runs and runs[-1].role == role:
                runs[-1].append(subtask)
            else:
                run = DispatchRun(role=role)
                run.append(subtask)
                runs.append(run)

        if len(runs) > 1 and not runtime.has_synthesizer_role(self.synthesizer_role):
            raise RuntimeError(
                f"P4 LearnedRouterPolicy: synthesizer_role "
                f"{self.synthesizer_role!r} is not registered in the "
                f"synthesizer config, but this request requires synthesis "
                f"({len(runs)} runs)."
            )

        specialist_prompt_template = (request.metadata or {}).get(
            "specialist_prompt_template"
        )
        specialist_requests: list[SpecialistRequest] = []
        run_payloads: list[dict[str, Any]] = []
        for index, run in enumerate(runs):
            subtask_prompt = run.rendered_prompt()
            if specialist_prompt_template:
                wrapped = specialist_prompt_template.format(
                    context=context_text, question=subtask_prompt,
                )
                sub_request = replace(
                    request, user_prompt=wrapped, context=None,
                )
            else:
                sub_request = replace(request, user_prompt=subtask_prompt)
            cache_key = build_specialist_cache_key(
                self.policy_id,
                request,
                run.role,
                phase="specialist",
                prompt_override=sub_request,
                extra=f"run-{index}",
            )
            specialist_requests.append(
                SpecialistRequest(
                    cache_key=cache_key,
                    role=run.role,
                    request=sub_request,
                )
            )
            run_payloads.append(
                {
                    "role": run.role,
                    "cache_key": cache_key,
                    "subtasks": [
                        {"text": subtask.text, "order": subtask.order}
                        for subtask in run.subtasks
                    ],
                }
            )

        return PolicyExecutionState(
            policy_id=self.policy_id,
            request=request,
            specialist_requests=specialist_requests,
            metadata={
                "subtasks": [
                    {"text": subtask.text, "order": subtask.order}
                    for subtask in subtasks
                ],
                "runs": run_payloads,
                "router_scores": [d.scores for d in decisions],
                "router_confidence": [d.confidence for d in decisions],
                "router_fallback_used": [d.fallback_used for d in decisions],
                "synthesizer_role": self.synthesizer_role,
                "synthesis_used": len(run_payloads) > 1,
                "predicted_route": [r.role for r in runs],
                "router_type": (
                    "Seq2SeqDecomposerRouter"
                    if self.use_joint_model
                    else type(self.router).__name__
                ),
            },
        )

    async def finalize(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyResult:
        run_payloads = list(state.metadata["runs"])
        confidence = _mean_or_none(state.metadata.get("router_confidence") or [])

        if len(run_payloads) == 1:
            response = cache.get(run_payloads[0]["cache_key"])
            metrics = _build_metrics(
                specialist_responses={run_payloads[0]["role"]: [response]},
                confidence=confidence,
            )
            return make_policy_result(
                state,
                response=orchestrator_to_prompt_response(
                    response,
                    metadata={
                        "synthesis_short_circuit": "single_run",
                        "synthesis_used_fallback": False,
                    },
                ),
                extra_metadata={
                    "synthesis_short_circuit": "single_run",
                    "synthesis_used_fallback": False,
                },
                metrics=metrics,
            )

        runs: list[DispatchRun] = []
        specialist_by_role: dict[str, list] = {}
        for run_payload in run_payloads:
            response = cache.get(run_payload["cache_key"])
            specialist_by_role.setdefault(run_payload["role"], []).append(response)
            runs.append(
                build_run_from_cached_response(
                    run_payload["role"],
                    run_payload["subtasks"],
                    response,
                )
            )

        synthesizer_orchestrator = await runtime.get_synthesizer_orchestrator()
        synthesis_result = await synthesize_ordered(
            question=state.request.user_prompt or "",
            runs=runs,
            orchestrator=synthesizer_orchestrator,
            synthesizer_role=self.synthesizer_role,
        )
        short_circuit = synthesis_result.metadata.get("short_circuit")
        specialist_by_role[self.synthesizer_role] = [synthesis_result.response]
        metrics = _build_metrics(
            specialist_responses=specialist_by_role,
            confidence=confidence,
        )
        return make_policy_result(
            state,
            response=orchestrator_to_prompt_response(
                synthesis_result.response,
                metadata={
                    "synthesis_short_circuit": short_circuit,
                    "synthesis_used_fallback": synthesis_result.used_fallback,
                },
            ),
            extra_metadata={
                "synthesis_short_circuit": short_circuit,
                "synthesis_used_fallback": synthesis_result.used_fallback,
            },
            metrics=metrics,
        )

    def _decision_from_joint_model(self, subtask: Subtask) -> RoutingDecision:
        hint = (subtask.suggested_role or "").strip()
        if not hint:
            return RoutingDecision(
                role=self.fallback_role,
                scores={},
                confidence=0.0,
                fallback_used=True,
                fallback_reason="joint_model_missing_role",
            )
        if hint not in ROLE_LABELS:
            return RoutingDecision(
                role=self.fallback_role,
                scores={},
                confidence=0.0,
                fallback_used=True,
                fallback_reason="joint_model_unknown_role",
            )
        return RoutingDecision(
            role=hint,
            scores={hint: 1.0},
            confidence=1.0,
            fallback_used=False,
            fallback_reason=None,
        )

    def _resolve_role(self, router_role: str, runtime: PolicyRuntime) -> str:
        if runtime.has_specialist_role(router_role):
            return router_role
        if not runtime.has_specialist_role(self.fallback_role):
            raise RuntimeError(
                f"P4 LearnedRouterPolicy: router role {router_role!r} is "
                f"not registered and fallback_role {self.fallback_role!r} "
                f"is also missing from the specialist config."
            )
        return self.fallback_role


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _build_metrics(
    *,
    specialist_responses: dict[str, list],
    confidence: float | None,
) -> PolicyMetrics:
    latency_by_role, usage_by_role = aggregate_specialist_metrics(
        specialist_responses
    )
    total_latency = sum(latency_by_role.values())
    total_usage = sum_usage(list(usage_by_role.values()))
    return PolicyMetrics(
        latency_ms=total_latency,
        token_usage=total_usage,
        confidence_score=confidence,
        specialist_latency_ms=latency_by_role,
        specialist_token_usage=usage_by_role,
    )
