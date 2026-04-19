from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from llm_gateway import PromptRequest

from council_policies.models import TASK_TO_ROLE
from council_policies.p3_policy import classify_task
from council_policies.policy_runner import (
    BasePolicyAdapter,
    PolicyExecutionState,
    PolicyResult,
    PolicyRuntime,
    SpecialistCache,
    SpecialistRequest,
    build_run_from_cached_response,
    build_specialist_cache_key,
    make_policy_result,
    orchestrator_to_prompt_response,
)
from council_policies.prompts import (
    TIEBREAK_SYSTEM_PROMPT,
    VOTER_SYSTEM_PROMPT,
    build_tiebreak_prompt,
    build_voter_prompt,
    parse_vote,
)
from council_policies.router import DispatchRun, Router, RoutingDecision, Subtask
from council_policies.synthesis import synthesize_ordered
from council_policies.decomposer import Decomposer, PassthroughDecomposer


_VOTE_LABELS = list("ABCDEFGHIJ")


class P3Adapter(BasePolicyAdapter):
    policy_id = "p3"

    def __init__(self, *, fallback_role: str = "general") -> None:
        self.fallback_role = fallback_role

    async def plan(
        self,
        request: PromptRequest,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState:
        task_type = classify_task(request.user_prompt or "")
        role = TASK_TO_ROLE[task_type]
        if not runtime.has_specialist_role(role):
            role = self.fallback_role
        specialist_request = SpecialistRequest(
            cache_key=build_specialist_cache_key(
                self.policy_id,
                request,
                role,
                phase="specialist",
            ),
            role=role,
            request=request,
        )
        return PolicyExecutionState(
            policy_id=self.policy_id,
            request=request,
            specialist_requests=[specialist_request],
            metadata={
                "task_type": task_type.value,
                "routed_role": role,
                "synthesis_used": False,
            },
        )

    async def finalize(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyResult:
        del runtime
        specialist_request = state.specialist_requests[0]
        response = cache.get(specialist_request.cache_key)
        return make_policy_result(
            state,
            response=orchestrator_to_prompt_response(response),
        )


class P4Adapter(BasePolicyAdapter):
    policy_id = "p4"

    def __init__(
        self,
        *,
        router: Router,
        decomposer: Decomposer | None = None,
        fallback_role: str = "fact_general",
        synthesizer_role: str = "synthesizer",
        max_subtasks: int = 4,
    ) -> None:
        self.router = router
        self.decomposer = decomposer or PassthroughDecomposer()
        self.fallback_role = fallback_role
        self.synthesizer_role = synthesizer_role
        self.max_subtasks = max_subtasks

    async def plan(
        self,
        request: PromptRequest,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState:
        prompt_text = request.user_prompt or ""
        context_text = request.context or ""
        subtasks = await self.decomposer.decompose(prompt_text, context=context_text)
        if not subtasks:
            subtasks = [Subtask(text=prompt_text, order=0)]
        subtasks = sorted(subtasks, key=lambda subtask: subtask.order)[: self.max_subtasks]

        decisions: list[RoutingDecision] = []
        runs: list[DispatchRun] = []
        specialist_requests: list[SpecialistRequest] = []
        run_payloads: list[dict[str, Any]] = []

        for subtask in subtasks:
            decision = self.router.classify(subtask.text, context=context_text)
            role = decision.role if runtime.has_specialist_role(decision.role) else self.fallback_role
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

        for index, run in enumerate(runs):
            sub_request = replace(request, user_prompt=run.rendered_prompt())
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
                "subtasks": [{"text": subtask.text, "order": subtask.order} for subtask in subtasks],
                "runs": run_payloads,
                "router_scores": [decision.scores for decision in decisions],
                "router_confidence": [decision.confidence for decision in decisions],
                "router_fallback_used": [decision.fallback_used for decision in decisions],
                "synthesizer_role": self.synthesizer_role,
                "synthesis_used": len(run_payloads) > 1,
            },
        )

    async def finalize(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyResult:
        run_payloads = list(state.metadata["runs"])
        if len(run_payloads) == 1:
            response = cache.get(run_payloads[0]["cache_key"])
            return make_policy_result(
                state,
                response=orchestrator_to_prompt_response(
                    response,
                    metadata={"synthesis_short_circuit": "single_run"},
                ),
                extra_metadata={"synthesis_short_circuit": "single_run"},
            )

        runs: list[DispatchRun] = []
        for run_payload in run_payloads:
            response = cache.get(run_payload["cache_key"])
            runs.append(
                build_run_from_cached_response(
                    run_payload["role"],
                    run_payload["subtasks"],
                    response,
                )
            )

        orchestrator = await runtime.get_synthesizer_orchestrator()
        synthesis_result = await synthesize_ordered(
            question=state.request.user_prompt or "",
            runs=runs,
            orchestrator=orchestrator,
            synthesizer_role=self.synthesizer_role,
        )
        return make_policy_result(
            state,
            response=orchestrator_to_prompt_response(
                synthesis_result.response,
                metadata={
                    "synthesis_short_circuit": synthesis_result.metadata.get("short_circuit"),
                    "synthesis_used_fallback": synthesis_result.used_fallback,
                },
            ),
            extra_metadata={
                "synthesis_short_circuit": synthesis_result.metadata.get("short_circuit"),
                "synthesis_used_fallback": synthesis_result.used_fallback,
            },
        )


class P2PromptAdapter(BasePolicyAdapter):
    policy_id = "p2_prompt"

    def __init__(
        self,
        *,
        council_roles: tuple[str, ...] = ("qa", "reasoning", "general"),
        voter_roles: tuple[str, ...] | None = None,
        seed: int = 42,
    ) -> None:
        self.council_roles = council_roles
        self.voter_roles = voter_roles or council_roles
        self._rng = random.Random(seed)

    async def plan(
        self,
        request: PromptRequest,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState:
        specialist_requests: list[SpecialistRequest] = []
        candidate_payloads: list[dict[str, str]] = []
        for role in self.council_roles:
            if not runtime.has_specialist_role(role):
                continue
            cache_key = build_specialist_cache_key(
                self.policy_id,
                request,
                role,
                phase="specialist",
            )
            specialist_requests.append(
                SpecialistRequest(cache_key=cache_key, role=role, request=request)
            )
            candidate_payloads.append({"role": role, "cache_key": cache_key})

        if not specialist_requests:
            raise ValueError("P2PromptAdapter requires at least one registered council role")

        return PolicyExecutionState(
            policy_id=self.policy_id,
            request=request,
            specialist_requests=specialist_requests,
            metadata={
                "candidate_roles": [item["role"] for item in candidate_payloads],
                "candidate_cache_keys": [item["cache_key"] for item in candidate_payloads],
                "vote_tally": {},
                "winner_role": None,
                "synthesis_used": False,
            },
        )

    async def complete_specialist_phase(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState:
        candidates = {
            request.role: cache.get(request.cache_key)
            for request in state.specialist_requests
        }
        winner_role, vote_tally = await self._run_vote(
            question=state.request.user_prompt or "",
            candidates=candidates,
            runtime=runtime,
        )
        state.metadata["vote_tally"] = vote_tally
        state.metadata["winner_role"] = winner_role
        state.winner_response = candidates[winner_role]
        return state

    async def finalize(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyResult:
        del cache, runtime
        if state.winner_response is None:
            raise RuntimeError("P2PromptAdapter winner_response missing after specialist phase")
        return make_policy_result(
            state,
            response=orchestrator_to_prompt_response(state.winner_response),
        )

    async def _run_vote(
        self,
        *,
        question: str,
        candidates: dict[str, Any],
        runtime: PolicyRuntime,
    ) -> tuple[str, dict[str, int]]:
        keys = list(candidates)
        labeled_answers = {_VOTE_LABELS[i]: candidates[key].text for i, key in enumerate(keys)}
        label_list = [_VOTE_LABELS[i] for i in range(len(keys))]
        label_to_key = {_VOTE_LABELS[i]: key for i, key in enumerate(keys)}
        voter_request = PromptRequest(
            system_prompt=VOTER_SYSTEM_PROMPT,
            user_prompt=build_voter_prompt(question, labeled_answers),
        )

        results = await runtime.execute_vote_requests(
            voter_roles=self.voter_roles,
            request=voter_request,
        )

        tally: dict[str, int] = {}
        for result in results:
            label = parse_vote(result.text, label_list)
            if not label:
                continue
            key = label_to_key[label]
            tally[key] = tally.get(key, 0) + 1

        if not tally:
            return keys[0], {}

        top_count = max(tally.values())
        leaders = [key for key, count in tally.items() if count == top_count]
        if len(leaders) == 1:
            return leaders[0], tally

        tie_break_request = PromptRequest(
            system_prompt=TIEBREAK_SYSTEM_PROMPT,
            user_prompt=build_tiebreak_prompt(
                question,
                {label: labeled_answers[label] for label in label_list if label_to_key[label] in leaders},
                {
                    label: tally[label_to_key[label]]
                    for label in label_list
                    if label_to_key[label] in leaders
                },
            ),
        )
        tie_break_result = await runtime.execute_vote_request(
            role="general" if runtime.has_specialist_role("general") else self.voter_roles[0],
            request=tie_break_request,
        )
        winning_label = parse_vote(
            tie_break_result.text,
            [label for label in label_list if label_to_key[label] in leaders],
        )
        if winning_label is None:
            return leaders[0], tally
        return label_to_key[winning_label], tally
