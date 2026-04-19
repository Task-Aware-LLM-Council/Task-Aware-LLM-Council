from __future__ import annotations

import pytest
from llm_gateway import PromptRequest

from council_policies import DatasetCouncilPolicy, LearnedRouterPolicy, RuleBasedRoutingPolicy
from council_policies.prompts import RATER_SYSTEM_PROMPT

from conftest import ConfigSentinel, FakeClient, FakeOrchestrator


@pytest.mark.asyncio
async def test_p3_run_returns_orchestrator_response() -> None:
    orchestrator = FakeOrchestrator(
        {
            "qa": FakeClient("qa"),
            "reasoning": FakeClient("reasoning"),
            "general": FakeClient("general"),
            "synthesizer": FakeClient("synthesizer", text="SYNTHESIZED"),
        }
    )
    policy = RuleBasedRoutingPolicy(orchestrator)

    response = await policy.run(PromptRequest(user_prompt="what is the capital of france"))

    assert response.target == "qa"
    assert response.text == "response-from-qa"


@pytest.mark.asyncio
async def test_p2_run_returns_selected_winner_response() -> None:
    def _response_fn(role: str):
        def _inner(request: PromptRequest) -> str:
            if request.system_prompt == RATER_SYSTEM_PROMPT:
                if role == "qa":
                    return '{"A": {"score": 9}, "B": {"score": 5}, "C": {"score": 4}}'
                if role == "reasoning":
                    return '{"A": {"score": 4}, "B": {"score": 9}, "C": {"score": 5}}'
                return '{"A": {"score": 5}, "B": {"score": 4}, "C": {"score": 9}}'
            return f"{role}-answer"

        return _inner

    orchestrator = FakeOrchestrator(
        {
            "qa": FakeClient("qa", response_fn=_response_fn("qa")),
            "reasoning": FakeClient("reasoning", response_fn=_response_fn("reasoning")),
            "general": FakeClient("general", response_fn=_response_fn("general")),
        }
    )
    request = PromptRequest(user_prompt="pick the best answer")
    question_policy = DatasetCouncilPolicy(orchestrator, seed=0)
    run_policy = DatasetCouncilPolicy(orchestrator, seed=0)

    question_result = await question_policy.run_question(request)
    response = await run_policy.run(request)

    assert question_result.best_answer is not None
    assert question_result.best_answer.response is not None
    assert response.target == question_result.best_answer.response.target
    assert response.text == question_result.best_answer.response.text


@pytest.mark.asyncio
async def test_p4_run_returns_orchestrator_response() -> None:
    specialist_config = ConfigSentinel(("math_code", "qa_reasoning", "fact_general"))
    synthesizer_config = ConfigSentinel(("synthesizer",))
    specialist_orch = FakeOrchestrator(
        {
            "math_code": FakeClient("math_code"),
            "qa_reasoning": FakeClient("qa_reasoning"),
            "fact_general": FakeClient("fact_general"),
        }
    )
    synthesizer_orch = FakeOrchestrator(
        {"synthesizer": FakeClient("synthesizer", text="SYNTHESIZED")}
    )

    def orchestrator_factory(config):
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch
        raise AssertionError(f"unexpected config passed to factory: {config!r}")

    class _Router:
        def classify(self, prompt: str, *, context: str = ""):
            del context
            if "math" in prompt:
                from council_policies import RoutingDecision

                return RoutingDecision(role="math_code", scores={"math_code": 1.0}, confidence=1.0)
            from council_policies import RoutingDecision

            return RoutingDecision(role="qa_reasoning", scores={"qa_reasoning": 1.0}, confidence=1.0)

    class _Decomposer:
        async def decompose(self, prompt: str, context: str = ""):
            del context
            from council_policies import Subtask

            return [Subtask(text="math step", order=0), Subtask(text="qa step", order=1)]

    policy = LearnedRouterPolicy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        router=_Router(),
        decomposer=_Decomposer(),
        orchestrator_factory=orchestrator_factory,
    )

    response = await policy.run(PromptRequest(user_prompt="mixed task"))

    assert response.target == "synthesizer"
    assert response.text == "SYNTHESIZED"
