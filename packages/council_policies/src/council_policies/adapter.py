from __future__ import annotations

import asyncio
import re
import time

from llm_gateway import PromptRequest
from llm_gateway.models import PromptResponse, Usage
from model_orchestration import ModelOrchestrator

from council_policies.models import P2CouncilDecision, P2RoleResult, P2SystemMetrics
from council_policies.p2.policy import _majority, _parse_vote
from council_policies.p2.prompts import LABEL_TO_ROLE, ROLE_TO_LABEL, build_vote_prompt

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
      3. The vote winner is returned as the benchmark output.

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
        decision = await _run_p2_for_one(self._orch, request)
        latency = time.monotonic() - t0
        async with self._lock:
            self._latencies.append(latency)
            self._total_input_tokens += decision.aggregated_usage.input_tokens or 0
            self._total_output_tokens += decision.aggregated_usage.output_tokens or 0
        return _decision_to_prompt_response(
            decision,
            latency_ms=latency * 1000,
            model_name=self.model_name,
        )

    async def generate_decision(
        self,
        request: PromptRequest,
        *,
        example_id: str,
        dataset_name: str,
    ) -> P2CouncilDecision:
        t0 = time.monotonic()
        decision = await _run_p2_for_one(
            self._orch,
            request,
            example_id=example_id,
            dataset_name=dataset_name,
        )
        latency = time.monotonic() - t0
        async with self._lock:
            self._latencies.append(latency)
            self._total_input_tokens += decision.aggregated_usage.input_tokens or 0
            self._total_output_tokens += decision.aggregated_usage.output_tokens or 0
        return decision

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


def _decision_to_prompt_response(
    decision: P2CouncilDecision,
    *,
    latency_ms: float,
    model_name: str,
) -> PromptResponse:
    return PromptResponse(
        model=model_name,
        text=decision.winning_answer,
        usage=decision.aggregated_usage,
        latency_ms=latency_ms,
        metadata={
            **decision.metadata,
            "votes": dict(decision.votes),
            "winning_label": decision.winning_label,
            "winning_role": decision.winning_role,
            "winning_model": decision.winning_model,
            "winning_answer": decision.winning_answer,
            "system_metrics": {
                "answer_usage": _usage_to_dict(decision.system_metrics.answer_usage),
                "vote_usage": _usage_to_dict(decision.system_metrics.vote_usage),
                "specialist_usage": _usage_to_dict(decision.system_metrics.specialist_usage),
                "answer_latency_ms_total": decision.system_metrics.answer_latency_ms_total,
                "vote_latency_ms_total": decision.system_metrics.vote_latency_ms_total,
                "specialist_latency_ms_total": decision.system_metrics.specialist_latency_ms_total,
                "wall_clock_latency_ms": decision.system_metrics.wall_clock_latency_ms,
            },
            "role_results": {
                role: _role_result_metadata(result)
                for role, result in decision.role_results.items()
            },
        },
    )


def _role_result_metadata(result: P2RoleResult) -> dict[str, object]:
    return {
        "role": result.role,
        "model": result.model,
        "provider": result.provider,
        "provider_mode": result.provider_mode,
        "latency_ms": result.latency_ms,
        "usage": _usage_to_dict(result.usage),
        "request_id": result.request_id,
        "finish_reason": result.finish_reason,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


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


def _response_usage(response) -> Usage:
    prompt_response = getattr(response, "prompt_response", None)
    usage = getattr(prompt_response, "usage", None)
    return usage if isinstance(usage, Usage) else Usage()


def _usage_to_dict(usage: Usage) -> dict[str, int | float | str | None]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cost": usage.cost,
        "currency": usage.currency,
    }


def _sum_latency_ms(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) if numeric else None


def _role_to_model(orch: ModelOrchestrator, role: str) -> str:
    for spec in orch.config.models:
        if spec.role == role:
            return spec.model
    return role


def _result_from_response(role: str, response) -> P2RoleResult:
    prompt_response = getattr(response, "prompt_response", None)
    model = getattr(response, "model", None)
    return P2RoleResult(
        role=role,
        model=model if isinstance(model, str) and model else role,
        provider=getattr(response, "provider", None),
        provider_mode=getattr(response, "provider_mode", None),
        text=_clean_answer(response.text),
        latency_ms=getattr(response, "latency_ms", None),
        usage=_response_usage(response),
        request_id=getattr(prompt_response, "request_id", None),
        finish_reason=getattr(prompt_response, "finish_reason", None),
    )


def _error_result(orch: ModelOrchestrator, role: str, exc: Exception) -> P2RoleResult:
    return P2RoleResult(
        role=role,
        model=_role_to_model(orch, role),
        text="",
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


async def _run_p2_for_one(
    orch: ModelOrchestrator,
    request: PromptRequest,
    *,
    example_id: str = "unknown",
    dataset_name: str = "router_dataset",
) -> P2CouncilDecision:
    """
    Full P2 flow for a single example, returning a rich council decision object.
    """
    question = request.user_prompt or ""
    total_usage = Usage()
    answer_usage = Usage()
    vote_usage = Usage()
    role_results: dict[str, P2RoleResult] = {}
    started = time.monotonic()

    async def _run(req, *, target):
        return await asyncio.wait_for(orch.run(req, target=target), timeout=_CALL_TIMEOUT_SECONDS)

    answer_req = PromptRequest(
        user_prompt=question,
        context=request.context,
        system_prompt=request.system_prompt,
        temperature=0.0,
    )

    async def answer_role(role: str) -> tuple[str, P2RoleResult]:
        try:
            resp = await _run(answer_req, target=role)
            return role, _result_from_response(role, resp)
        except Exception as exc:
            print(f"[P2] answer error ({role}): {type(exc).__name__}: {exc}")
            return role, _error_result(orch, role, exc)

    answers = {}
    for role, role_result in await asyncio.gather(*[answer_role(role) for role in _ROLES]):
        role_results[role] = role_result
        answers[role] = role_result.text
        total_usage = _add_usage(total_usage, role_result.usage)
        answer_usage = _add_usage(answer_usage, role_result.usage)
    label_to_answer = {ROLE_TO_LABEL[role]: answers[role] for role in _ROLES}
    print(f"[P2] answers: A={answers['qa'][:60]!r}  B={answers['reasoning'][:60]!r}  C={answers['general'][:60]!r}")

    vote_req = PromptRequest(
        user_prompt=build_vote_prompt(question, label_to_answer),
        temperature=0.0,
    )

    async def vote_role(role: str) -> tuple[str, str, Usage, float | None]:
        try:
            resp = await _run(vote_req, target=role)
            return role, _parse_vote(resp.text), _response_usage(resp), getattr(resp, "latency_ms", None)
        except Exception as exc:
            print(f"[P2] vote error ({role}): {type(exc).__name__}: {exc}")
            return role, "A", Usage(), None

    votes = {}
    vote_latencies: list[float | None] = []
    for role, vote, usage, vote_latency_ms in await asyncio.gather(*[vote_role(role) for role in _ROLES]):
        votes[role] = vote
        total_usage = _add_usage(total_usage, usage)
        vote_usage = _add_usage(vote_usage, usage)
        vote_latencies.append(vote_latency_ms)
    print(f"[P2] votes: {votes}  → winner: {_majority(votes)}")

    winning_label = _majority(votes)
    winning_role = LABEL_TO_ROLE[winning_label]
    winning_model = role_results[winning_role].model
    winning_answer = label_to_answer[winning_label]
    return P2CouncilDecision(
        example_id=example_id,
        dataset_name=dataset_name,
        request=request,
        winning_label=winning_label,
        winning_role=winning_role,
        winning_model=winning_model,
        winning_answer=winning_answer,
        votes=votes,
        role_results=role_results,
        aggregated_usage=total_usage,
        system_metrics=P2SystemMetrics(
            answer_usage=answer_usage,
            vote_usage=vote_usage,
            specialist_usage=total_usage,
            answer_latency_ms_total=_sum_latency_ms([result.latency_ms for result in role_results.values()]),
            vote_latency_ms_total=_sum_latency_ms(vote_latencies),
            specialist_latency_ms_total=_sum_latency_ms(
                [result.latency_ms for result in role_results.values()] + vote_latencies
            ),
            wall_clock_latency_ms=(time.monotonic() - started) * 1000,
        ),
        metadata={
            "answer_a": label_to_answer["A"],
            "answer_b": label_to_answer["B"],
            "answer_c": label_to_answer["C"],
            "candidate_answers": label_to_answer,
        },
    )
