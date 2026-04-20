from __future__ import annotations

from llm_gateway.base import BaseLLMClient, ClientInfo
from llm_gateway.models import PromptRequest, PromptResponse


class PolicyClient(BaseLLMClient):
    """
    Adapter that allows `benchmarking_pipeline` to use a council policy
    as if it were a single standard LLM model.
    """

    def __init__(self, policy, model_name: str = "p3-router") -> None:
        self.policy = policy
        info = ClientInfo(
            provider="council_policy",
            default_model=model_name,
            timeout_seconds=900,
            max_retries=0,
            api_base=None,
        )
        super().__init__(info)

    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Forward a single request through the policy's routing logic."""
        from council_policies.p3_policy import classify_task
        
        # 1. Classify the user prompt
        prompt = request.user_prompt or ""
        task_type = classify_task(prompt)
        
        # 2. Route to the appropriate specialist
        routed_role, orchestrator_res = await self.policy._dispatch(task_type, request)
        
        # 3. Safely wrap usage data
        usage_metadata = {}
        if orchestrator_res.usage:
            usage_metadata = {
                "input_tokens": orchestrator_res.usage.input_tokens,
                "output_tokens": orchestrator_res.usage.output_tokens,
                "total_tokens": orchestrator_res.usage.total_tokens,
            }

        return PromptResponse(
            model=orchestrator_res.model,
            text=orchestrator_res.text,
            latency_ms=orchestrator_res.latency_ms,
            usage=orchestrator_res.usage,
            finish_reason=orchestrator_res.finish_reason,
            provider=orchestrator_res.provider,
            request_id=orchestrator_res.request_id,
            raw_response=orchestrator_res.raw_response,
            metadata={
                **orchestrator_res.metadata,
                "policy": "p3_rule_based",
                "task_type": task_type.value,
                "routed_role": routed_role,
                "usage": usage_metadata,
            },
        )
