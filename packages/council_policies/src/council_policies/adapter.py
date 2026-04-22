from __future__ import annotations

import asyncio
import re
import time

from llm_gateway import PromptRequest
from llm_gateway.models import PromptResponse, Usage
from model_orchestration import ModelOrchestrator

from council_policies.p2.policy import _majority, _parse_vote
from council_policies.p2.prompts import ROLE_TO_LABEL, build_vote_prompt, build_synthesis_prompt

_ROLES = ("qa", "reasoning", "general")
_CALL_TIMEOUT_SECONDS = 3600  # 1 hour — covers vLLM model load + torch.compile + inference on single GPU
_MAX_ANSWER_CHARS = 2000  # cap each answer in synthesis to avoid context overflow


def _clean_answer(text: str) -> str:
    """Strip reasoning blocks and truncate to avoid context overflow."""
    # Strip internal reasoning tags (DeepSeek-R1 think, musique scratchpad, etc.)
    for tag in ("think", "scratchpad", "reasoning"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
    text = text.strip()
    if len(text) > _MAX_ANSWER_CHARS:
        text = text[:_MAX_ANSWER_CHARS] + "..."
    return text


class P2PolicyClient:
    """
    Wraps the P2 council policy as a drop-in generate() client for run_benchmark.

    Each call to generate() runs the full P2 flow for one example:
      1. All 3 specialists answer in parallel.
      2. All 3 specialists vote on A/B/C in parallel.
      3. The general model synthesizes a final answer.

    The orchestrator is kept open for the lifetime of the benchmark run,
    so all examples share the same model connections.

    Tracks per-example latency and token usage; access via .stats after the run.
    """

    def __init__(self, orchestrator: ModelOrchestrator, model_name: str = "p2_council") -> None:
        self._orch = orchestrator
        self.model_name = model_name
        # accumulated stats — read after run completes
        self._latencies: list[float] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._lock = asyncio.Lock()

    async def generate(self, request: PromptRequest) -> PromptResponse:
        t0 = time.monotonic()
        final_text, usage = await _run_p2_for_one(self._orch, request)
        latency = time.monotonic() - t0
        async with self._lock:
            self._latencies.append(latency)
            self._total_input_tokens += usage.input_tokens or 0
            self._total_output_tokens += usage.output_tokens or 0
        return PromptResponse(model=self.model_name, text=final_text, usage=usage, latency_ms=latency * 1000)

    @property
    def stats(self) -> dict:
        n = len(self._latencies)
        return {
            "examples_completed": n,
            "avg_latency_s": sum(self._latencies) / n if n else 0.0,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

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
            answers[role] = _clean_answer(resp.text)
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
        system_prompt=request.system_prompt,  # preserve format instructions (e.g. "Final Answer: ...")
        # context intentionally omitted — already incorporated in Stage 1 answers; re-adding bloats tokens
        temperature=0.3,
    )
    try:
        synth_resp = await _run(synth_req, target="general")
        total_usage = _add_usage(total_usage, synth_resp.prompt_response.usage)
        return synth_resp.text.strip(), total_usage
    except Exception as exc:
        print(f"[P2] synthesis error: {type(exc).__name__}: {exc}")
        return winning_answer, total_usage  # fallback to the vote winner
