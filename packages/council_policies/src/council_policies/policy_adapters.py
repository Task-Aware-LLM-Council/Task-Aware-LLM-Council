from __future__ import annotations

import random
from typing import Any

from llm_gateway import PromptRequest

from council_policies.models import TASK_TO_ROLE
from council_policies.p3_policy import classify_task
from council_policies.policy_runner import (
    BasePolicyAdapter,
    PolicyExecutionState,
    PolicyMetrics,
    PolicyResult,
    PolicyRuntime,
    SpecialistCache,
    SpecialistRequest,
    aggregate_specialist_metrics,
    build_specialist_cache_key,
    make_policy_result,
    orchestrator_to_prompt_response,
    sum_usage,
)
from council_policies.prompts import (
    TIEBREAK_SYSTEM_PROMPT,
    VOTER_SYSTEM_PROMPT,
    build_tiebreak_prompt,
    build_voter_prompt,
    parse_vote,
)


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
        primary_role = TASK_TO_ROLE[task_type]
        fallback_used = False
        if runtime.has_specialist_role(primary_role):
            role = primary_role
        else:
            role = self.fallback_role
            fallback_used = True
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
                "router_fallback_used": fallback_used,
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
        metrics = _single_specialist_metrics(
            role=specialist_request.role,
            response=response,
            confidence=(
                0.0 if state.metadata.get("router_fallback_used") else 1.0
            ),
        )
        return make_policy_result(
            state,
            response=orchestrator_to_prompt_response(response),
            metrics=metrics,
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
        winner_role, vote_tally, voter_responses = await self._run_vote(
            question=state.request.user_prompt or "",
            candidates=candidates,
            runtime=runtime,
        )
        state.metadata["vote_tally"] = vote_tally
        state.metadata["winner_role"] = winner_role
        state.metadata["_voter_responses"] = voter_responses
        state.metadata["_candidate_responses"] = candidates
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

        candidate_responses = state.metadata.pop("_candidate_responses", {})
        voter_responses = state.metadata.pop("_voter_responses", [])
        responses_by_role: dict[str, list] = {
            role: [resp] for role, resp in candidate_responses.items()
        }
        if voter_responses:
            responses_by_role["_voters"] = list(voter_responses)

        tally = state.metadata.get("vote_tally") or {}
        winner = state.metadata.get("winner_role")
        total = sum(tally.values())
        confidence = (tally.get(winner, 0) / total) if total else None

        return make_policy_result(
            state,
            response=orchestrator_to_prompt_response(state.winner_response),
            metrics=_multi_specialist_metrics(
                responses_by_role=responses_by_role,
                confidence=confidence,
            ),
        )

    async def _run_vote(
        self,
        *,
        question: str,
        candidates: dict[str, Any],
        runtime: PolicyRuntime,
    ) -> tuple[str, dict[str, int], list]:
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

        voter_responses = list(results)

        tally: dict[str, int] = {}
        for result in results:
            label = parse_vote(result.text, label_list)
            if not label:
                continue
            key = label_to_key[label]
            tally[key] = tally.get(key, 0) + 1

        if not tally:
            return keys[0], {}, voter_responses

        top_count = max(tally.values())
        leaders = [key for key, count in tally.items() if count == top_count]
        if len(leaders) == 1:
            return leaders[0], tally, voter_responses

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
        voter_responses.append(tie_break_result)
        winning_label = parse_vote(
            tie_break_result.text,
            [label for label in label_list if label_to_key[label] in leaders],
        )
        if winning_label is None:
            return leaders[0], tally, voter_responses
        return label_to_key[winning_label], tally, voter_responses


def _single_specialist_metrics(
    *,
    role: str,
    response: Any,
    confidence: float | None,
) -> PolicyMetrics:
    latency_by_role, usage_by_role = aggregate_specialist_metrics(
        {role: [response]}
    )
    return PolicyMetrics(
        latency_ms=sum(latency_by_role.values()),
        token_usage=sum_usage(list(usage_by_role.values())),
        confidence_score=confidence,
        specialist_latency_ms=latency_by_role,
        specialist_token_usage=usage_by_role,
    )


def _multi_specialist_metrics(
    *,
    responses_by_role: dict[str, list],
    confidence: float | None,
) -> PolicyMetrics:
    latency_by_role, usage_by_role = aggregate_specialist_metrics(
        responses_by_role
    )
    return PolicyMetrics(
        latency_ms=sum(latency_by_role.values()),
        token_usage=sum_usage(list(usage_by_role.values())),
        confidence_score=confidence,
        specialist_latency_ms=latency_by_role,
        specialist_token_usage=usage_by_role,
    )
