"""
Synthesis aggregator for P3/P4 multi-skill dispatch.

Used when a policy fans a task out to multiple specialists (e.g. math + qa)
and needs to combine their partial outputs into one final answer.

Not to be confused with voter.py (P2's selection-from-redundant-attempts)
or aggregation.py (deterministic selection strategies). Synthesis is
*combination*: the output is a new answer produced by a synthesizer LLM,
not a pick from the inputs.

Design (per Aniruth, 2026-04-17; revised 2026-04-18 for two-phase GPU):
  - Input is partials from specialists (full text, not extracted).
  - A single synthesizer LLM call fuses them; no deterministic path.
  - The `orchestrator` argument is the **synthesizer-phase** orchestrator
    — a separate `ModelOrchestrator` instance built from `synthesizer_config`,
    entered via `async with` AFTER the specialist orchestrator has exited
    (so at most three models are GPU-resident at any time). See
    `test_orchestartor_client.py` at the repo root for the reference pattern.
  - Synthesizer role is the dedicated 'synthesizer' role. It is excluded
    from TASK_TO_ROLE so rule-based routing can never dispatch a user task
    to it, preserving the referee property.
  - Self-bias guard: refuse to synthesize if synthesizer_role also appears
    as a key in partials (would over-weight its own partial).
  - On synthesizer failure we fall back to the first partial rather than
    raising, so the surrounding policy can still return *something*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorResponse

from council_policies.prompts import (
    ORDERED_SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_ordered_synthesis_prompt,
    build_synthesis_prompt,
)
from council_policies.router import DispatchRun

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class SynthesisResult:
    """Outcome of synthesize(): the fused response + provenance."""

    response: OrchestratorResponse
    synthesizer_role: str
    partials: dict[str, OrchestratorResponse]  # keyed by specialist role
    used_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.response.text


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def synthesize(
    *,
    question: str,
    partials: dict[str, OrchestratorResponse],
    orchestrator: ModelOrchestrator,
    synthesizer_role: str = "synthesizer",
) -> SynthesisResult:
    """
    Fuse per-specialist partial answers into one final response.

    Parameters
    ----------
    question
        The original user prompt (pre-decomposition). Gives the synthesizer
        the overall target so it knows what the partials are *for*.
    partials
        role name -> full OrchestratorResponse. Must be non-empty.
    orchestrator
        The **synthesizer-phase** ModelOrchestrator (not the specialist
        one). Caller is responsible for entering it via `async with` and
        calling `load_all()` before this function, and for tearing it
        down after. See `test_orchestartor_client.py`.
    synthesizer_role
        Which orchestrator role performs the fusion. Defaults to the
        dedicated 'synthesizer' role; override only if you have a well-
        justified reason. Must not appear as a key in ``partials``.
    """
    if not partials:
        raise ValueError("synthesize() requires at least one partial response")

    if synthesizer_role in partials:
        raise ValueError(
            f"synthesizer_role {synthesizer_role!r} also appears as a specialist "
            f"in partials — refusing to synthesize to avoid self-bias. Use a "
            f"role that is not in TASK_TO_ROLE (the dedicated 'synthesizer' role "
            f"registered in model-orchestration satisfies this)."
        )

    # Short-circuit: one specialist = nothing to synthesize. Return its
    # response verbatim so single-skill P3 doesn't pay an extra LLM hop.
    if len(partials) == 1:
        only_role, only_resp = next(iter(partials.items()))
        return SynthesisResult(
            response=only_resp,
            synthesizer_role=only_role,
            partials=partials,
            used_fallback=False,
            metadata={"short_circuit": "single_specialist"},
        )

    partial_texts = {role: resp.text for role, resp in partials.items()}

    user_prompt = build_synthesis_prompt(question, partial_texts)
    request = PromptRequest(
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    try:
        fused = await orchestrator.get_client(synthesizer_role).get_response(request)
    except Exception as exc:
        logger.warning(
            "Synthesizer role %r failed (%s); falling back to first partial",
            synthesizer_role, exc,
        )
        fallback_role, fallback_resp = next(iter(partials.items()))
        return SynthesisResult(
            response=fallback_resp,
            synthesizer_role=fallback_role,
            partials=partials,
            used_fallback=True,
            metadata={"fallback_reason": str(exc)},
        )

    return SynthesisResult(
        response=fused,
        synthesizer_role=synthesizer_role,
        partials=partials,
        used_fallback=False,
        metadata={"specialist_roles": tuple(partials)},
    )


# --------------------------------------------------------------------------- #
# Ordered synthesis (P4)
# --------------------------------------------------------------------------- #


async def synthesize_ordered(
    *,
    question: str,
    runs: list[DispatchRun],
    orchestrator: ModelOrchestrator,
    synthesizer_role: str = "synthesizer",
) -> SynthesisResult:
    """
    Fuse an ordered list of `DispatchRun`s into one final response.

    Differs from `synthesize()` (above) in two ways:
      1. Input is a **list** of runs, not a role-keyed dict. The same role
         may appear in multiple runs (the non-adjacent [A, B, A] case) —
         a dict would collapse them and scramble ordering.
      2. The synthesizer is prompted to preserve step sequence, not just
         merge claims, via `ORDERED_SYNTHESIZER_SYSTEM_PROMPT`.

    Each run must have had its `response` assigned before calling this —
    typically by `LearnedRouterPolicy.run()` after awaiting the specialist
    dispatch gather.
    """
    if not runs:
        raise ValueError("synthesize_ordered() requires at least one run")

    missing = [i for i, r in enumerate(runs) if r.response is None]
    if missing:
        raise ValueError(
            f"runs at indices {missing} have no response — dispatch before "
            f"calling synthesize_ordered()"
        )

    # Self-bias guard: synthesizer must not also be a specialist in any run.
    run_roles = {r.role for r in runs}
    if synthesizer_role in run_roles:
        raise ValueError(
            f"synthesizer_role {synthesizer_role!r} also appears as a "
            f"specialist role in runs — refusing to synthesize to avoid "
            f"self-bias. Use a role that is not in TASK_TO_ROLE."
        )

    # Short-circuit: one run = nothing to synthesize.
    if len(runs) == 1:
        only = runs[0]
        return SynthesisResult(
            response=only.response,  # type: ignore[arg-type]  # not-None checked above
            synthesizer_role=only.role,
            partials={only.role: only.response},  # type: ignore[dict-item]
            used_fallback=False,
            metadata={"short_circuit": "single_run"},
        )

    ordered_partials = [
        (i, r.role, r.response.text)  # type: ignore[union-attr]
        for i, r in enumerate(runs)
    ]
    user_prompt = build_ordered_synthesis_prompt(question, ordered_partials)
    request = PromptRequest(
        system_prompt=ORDERED_SYNTHESIZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # partials view is best-effort (collapses same-role runs) — full
    # run ordering is preserved in metadata.
    partials_view: dict[str, OrchestratorResponse] = {
        r.role: r.response for r in runs  # type: ignore[misc]
    }

    try:
        fused = await orchestrator.get_client(synthesizer_role).get_response(request)
    except Exception as exc:
        logger.warning(
            "Ordered synthesizer role %r failed (%s); falling back to first run",
            synthesizer_role, exc,
        )
        first = runs[0]
        return SynthesisResult(
            response=first.response,  # type: ignore[arg-type]
            synthesizer_role=first.role,
            partials=partials_view,
            used_fallback=True,
            metadata={
                "fallback_reason": str(exc),
                "run_order": [r.role for r in runs],
            },
        )

    return SynthesisResult(
        response=fused,
        synthesizer_role=synthesizer_role,
        partials=partials_view,
        used_fallback=False,
        metadata={
            "run_count": len(runs),
            "run_order": [r.role for r in runs],
        },
    )
