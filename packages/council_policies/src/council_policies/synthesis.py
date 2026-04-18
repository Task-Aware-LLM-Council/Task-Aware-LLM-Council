"""
Synthesis aggregator for P3/P4 multi-skill dispatch.

Used when a policy fans a task out to multiple specialists (e.g. math + qa)
and needs to combine their partial outputs into one final answer.

Not to be confused with voter.py (P2's selection-from-redundant-attempts)
or aggregation.py (deterministic selection strategies). Synthesis is
*combination*: the output is a new answer produced by a synthesizer LLM,
not a pick from the inputs.

Design (per Aniruth, 2026-04-17):
  - Input is a map role -> OrchestratorResponse (full text, not extracted).
  - A single synthesizer LLM call fuses them; no deterministic path.
  - Synthesizer role defaults to the dedicated 'synthesizer' role registered
    in model-orchestration. It is excluded from TASK_TO_ROLE so rule-based
    routing can never dispatch a user task to it, preserving the referee
    property.
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
    SYNTHESIZER_SYSTEM_PROMPT,
    build_synthesis_prompt,
)

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
        Used to invoke the synthesizer.
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
