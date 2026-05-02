from __future__ import annotations

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
