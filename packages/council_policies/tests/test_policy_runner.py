from __future__ import annotations

from dataclasses import dataclass

import pytest
from llm_gateway import PromptRequest

from council_policies import (
    CouncilBenchmarkRunner,
    LearnedRouterPolicy,
    P3Adapter,
    PolicyRuntime,
    Subtask,
)

from conftest import ConfigSentinel, FakeClient, FakeOrchestrator


@dataclass
class StubRouter:
    role_map: dict[str, str]
    default_role: str = "fact_general"

    def classify(self, prompt: str, *, context: str = ""):
        del context
        for needle, role in self.role_map.items():
            if needle in prompt:
                from council_policies.p4.router import RoutingDecision

                return RoutingDecision(role=role, scores={role: 1.0}, confidence=1.0)

        from council_policies.p4.router import RoutingDecision

        return RoutingDecision(
            role=self.default_role,
            scores={self.default_role: 0.0},
            confidence=0.0,
            fallback_used=True,
            fallback_reason="no_match",
        )


class StubDecomposer:
    def __init__(self, subtasks: list[Subtask]) -> None:
        self._subtasks = subtasks

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]:
        del prompt, context
        return list(self._subtasks)


class TrackingOrchestrator(FakeOrchestrator):
    def __init__(self, name: str, clients: dict[str, FakeClient], events: list[str]) -> None:
        super().__init__(clients)
        self.name = name
        self._events = events

    async def load_all(self, targets=None, *, max_parallel=None) -> None:
        del targets
        self._events.append(f"{self.name}:load:{max_parallel}")
        await super().load_all(max_parallel=max_parallel)

    async def __aenter__(self):
        self._events.append(f"{self.name}:enter")
        return await super().__aenter__()

    async def __aexit__(self, *args):
        self._events.append(f"{self.name}:exit")
        await super().__aexit__(*args)


def _make_runtime(
    specialist_orch: FakeOrchestrator,
    synthesizer_orch: FakeOrchestrator,
    specialist_config: ConfigSentinel,
    synthesizer_config: ConfigSentinel,
) -> PolicyRuntime:
    def _factory(config):
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch
        raise AssertionError(f"unexpected config: {config!r}")

    return PolicyRuntime(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=_factory,
    )


@pytest.mark.asyncio
async def test_runner_batches_specialists_then_synthesizes_in_second_phase():
    events: list[str] = []
    specialist_config = ConfigSentinel(
        ("qa", "reasoning", "general", "math_code", "qa_reasoning", "fact_general")
    )
    synthesizer_config = ConfigSentinel(("synthesizer",))
    specialist_orch = TrackingOrchestrator(
        "specialist",
        {
            "qa": FakeClient("qa", text="P3-QA"),
            "reasoning": FakeClient("reasoning"),
            "general": FakeClient("general"),
            "math_code": FakeClient("math_code", text="math partial"),
            "qa_reasoning": FakeClient("qa_reasoning", text="qa partial"),
            "fact_general": FakeClient("fact_general"),
        },
        events,
    )
    synthesizer_orch = TrackingOrchestrator(
        "synth",
        {"synthesizer": FakeClient("synthesizer", text="SYNTHESIZED")},
        events,
    )
    runtime = _make_runtime(
        specialist_orch,
        synthesizer_orch,
        specialist_config,
        synthesizer_config,
    )
    runner = CouncilBenchmarkRunner(
        policies=(
            P3Adapter(),
            LearnedRouterPolicy(
                router=StubRouter({"math": "math_code", "qa": "qa_reasoning"}),
                decomposer=StubDecomposer(
                    [Subtask("math step", 0), Subtask("qa step", 1)]
                ),
            ),
        ),
        runtime=runtime,
    )

    result = await runner.run([PromptRequest(user_prompt="What is France?")])

    assert events == [
        "specialist:enter",
        "specialist:load:1",
        "specialist:exit",
        "synth:enter",
        "synth:load:1",
        "synth:exit",
    ]
    assert len(result.results) == 2
    by_policy = {item.policy_id: item for item in result.results}
    assert by_policy["p3"].response.text == "P3-QA"
    assert by_policy["p4"].response.text == "SYNTHESIZED"
    assert len(specialist_orch.get_client("qa").requests) == 1
    assert len(specialist_orch.get_client("math_code").requests) == 1
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 1
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 1


@pytest.mark.asyncio
async def test_runner_reuses_cached_specialist_calls_for_identical_requests():
    specialist_config = ConfigSentinel(("qa", "general"))
    synthesizer_config = ConfigSentinel(("synthesizer",))
    specialist_orch = FakeOrchestrator(
        {
            "qa": FakeClient("qa", text="same answer"),
            "general": FakeClient("general", text="fallback"),
        }
    )
    synthesizer_orch = FakeOrchestrator(
        {"synthesizer": FakeClient("synthesizer", text="unused")}
    )
    runtime = _make_runtime(
        specialist_orch,
        synthesizer_orch,
        specialist_config,
        synthesizer_config,
    )
    runner = CouncilBenchmarkRunner(policies=(P3Adapter(),), runtime=runtime)

    requests = [
        PromptRequest(user_prompt="What is the capital of France?"),
        PromptRequest(user_prompt="What is the capital of France?"),
    ]
    result = await runner.run(requests)

    assert len(result.results) == 2
    assert len(result.specialist_cache_keys) == 1
    assert len(specialist_orch.get_client("qa").requests) == 1
    assert synthesizer_orch.enter_count == 0
