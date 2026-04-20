from __future__ import annotations

import asyncio

from llm_gateway import PromptRequest
from llm_gateway.models import PromptResponse, Usage
from model_orchestration import ModelOrchestrator

from council_policies.p2.policy import _majority, _parse_vote
from council_policies.p2.prompts import ROLE_TO_LABEL, build_vote_prompt, build_synthesis_prompt

_ROLES = ("qa", "reasoning", "general")
_CALL_TIMEOUT_SECONDS = 90


class P2PolicyClient:
    """
    Wraps the P2 council policy as a drop-in generate() client for run_benchmark.

    Each call to generate() runs the full P2 flow for one example:
      1. All 3 specialists answer in parallel.
      2. All 3 specialists vote on A/B/C in parallel.
      3. The general model synthesizes a final answer.

    The orchestrator is kept open for the lifetime of the benchmark run,
    so all examples share the same model connections.

    Usage:
        async with ModelOrchestrator(config) as orch:
            client = P2PolicyClient(orch)

            async def pipeline_runner(datasets, pipeline_config):
                return await run_benchmark(datasets, pipeline_config, client=client)

            await run_registered_benchmark_suite(datasets, spec, pipeline_runner=pipeline_runner)
    """

    def __init__(self, orchestrator: ModelOrchestrator, model_name: str = "p2_council") -> None:
        self._orch = orchestrator
        self.model_name = model_name

    async def generate(self, request: PromptRequest) -> PromptResponse:
        final_text, usage = await _run_p2_for_one(self._orch, request)
        return PromptResponse(model=self.model_name, text=final_text, usage=usage)

    async def close(self) -> None:
        pass  # orchestrator lifecycle is managed by the caller


def _add_usage(total: Usage, resp_usage: Usage) -> Usage:
    """Accumulate token counts from one response into a running total."""
    def _add(a, b):
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)
    return Usage(
        input_tokens=_add(total.input_tokens, resp_usage.input_tokens),
        output_tokens=_add(total.output_tokens, resp_usage.output_tokens),
        total_tokens=_add(total.total_tokens, resp_usage.total_tokens),
    )


async def _run_p2_for_one(orch: ModelOrchestrator, request: PromptRequest) -> tuple[str, Usage]:
    """
    Full P2 flow for a single example.
    Returns (synthesized_answer, aggregated_usage_across_all_calls).
    """
    question = request.user_prompt or ""
    total_usage = Usage()

    async def _run(req, *, target):
        return await asyncio.wait_for(orch.run(req, target=target), timeout=_CALL_TIMEOUT_SECONDS)

    # Stage 1: All 3 models answer the question (sequential to avoid rate limits)
    answer_req = PromptRequest(
        user_prompt=question,
        context=request.context,
        system_prompt=request.system_prompt,
        temperature=0.0,
    )
    answers = {}
    for role in _ROLES:
        try:
            resp = await _run(answer_req, target=role)
            answers[role] = resp.text.strip()
            total_usage = _add_usage(total_usage, resp.prompt_response.usage)
        except Exception as exc:
            print(f"[P2] answer error ({role}): {type(exc).__name__}: {exc}")
            answers[role] = ""
    label_to_answer = {ROLE_TO_LABEL[role]: answers[role] for role in _ROLES}
    print(f"[P2] answers: A={answers['qa'][:60]!r}  B={answers['reasoning'][:60]!r}  C={answers['general'][:60]!r}")

    # Stage 2: All 3 models vote on A/B/C (sequential)
    vote_req = PromptRequest(
        user_prompt=build_vote_prompt(question, label_to_answer),
        temperature=0.0,
    )
    votes = {}
    for role in _ROLES:
        try:
            resp = await _run(vote_req, target=role)
            votes[role] = _parse_vote(resp.text)
            total_usage = _add_usage(total_usage, resp.prompt_response.usage)
        except Exception as exc:
            print(f"[P2] vote error ({role}): {type(exc).__name__}: {exc}")
            votes[role] = "A"
    print(f"[P2] votes: {votes}  → winner: {_majority(votes)}")

    # Stage 3: Synthesizer (general model) writes the final improved answer
    winning_label = _majority(votes)
    winning_answer = label_to_answer[winning_label]
    other_answers = [text for lbl, text in label_to_answer.items() if lbl != winning_label]

    synth_req = PromptRequest(
        user_prompt=build_synthesis_prompt(question, winning_answer, other_answers),
        temperature=0.3,
    )
    try:
        synth_resp = await _run(synth_req, target="general")
        total_usage = _add_usage(total_usage, synth_resp.prompt_response.usage)
        return synth_resp.text.strip(), total_usage
    except Exception as exc:
        print(f"[P2] synthesis error: {type(exc).__name__}: {exc}")
        return winning_answer, total_usage  # fallback to the vote winner
