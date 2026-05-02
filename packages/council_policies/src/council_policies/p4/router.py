"""
P4 router types + fixtures.

Three data types and one protocol, colocated because they form a single
contract surface:

  prompt/context ──► Router.classify ──► RoutingDecision
                                            │
                  (grouped into runs by the policy)
                                            ▼
                     Subtask ──► DispatchRun.append
                                            │
                                            ▼
                     Specialist call ──► DispatchRun.response

Two `Router` implementations ship in this module:
  * `KeywordRouter` — wraps P3's regex classifier. Fast, deterministic,
    zero-dep; the permanent test fixture.
  * `LearnedRouter` — fine-tuned distilroberta classifier. Pure decision
    logic, backed by an injectable `ScoreFn` so the heavy HF model load
    stays out of unit tests. The HF-backed `ScoreFn` adapter lands with
    the training script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from model_orchestration import OrchestratorResponse

from council_policies.p4.types import TaskType
from council_policies.p4.p3_policy import classify_task
from council_policies.p4.router_featurize import DEFAULT_CONTEXT_CHAR_CAP, featurize

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Placeholder TaskType → role collapse
# --------------------------------------------------------------------------- #
# The real map lives in models.py (`TASK_TO_ROLE`) and gets collapsed to
# three roles in Step 0. Until Step 0 lands (blocked on owner confirming
# final role names), the `KeywordRouter` uses this local placeholder map
# so we can exercise the 3-role dispatch path today. When Step 0 ships,
# delete this constant and import `TASK_TO_ROLE` instead.
_PLACEHOLDER_TASKTYPE_TO_ROLE: dict[TaskType, str] = {
    TaskType.MATH: "math_code",
    TaskType.CODE: "math_code",
    TaskType.QA: "qa_reasoning",
    TaskType.REASONING: "qa_reasoning",
    TaskType.FEVER: "fact_general",
    TaskType.GENERAL: "fact_general",
}


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class Subtask:
    """One decomposed step. `order` is the logical execution index —
    trust it over list position, since decomposer output may not be
    pre-sorted.

    `suggested_role` is an optional hint emitted by the `LLMDecomposer`
    (one of e.g. `math`, `code`, `fact_verify`, `qa_multihop`,
    `qa_longctx` — the decomposer-level role vocabulary, not the
    orchestrator role name). The `Router` is free to ignore it or use
    it as a prior; today's `KeywordRouter` ignores it and classifies
    from `text` independently."""

    text: str
    order: int
    suggested_role: str | None = None


@dataclass(slots=True, frozen=True)
class RoutingDecision:
    """Result of routing one subtask to one specialist role."""

    role: str
    scores: dict[str, float]
    confidence: float
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass(slots=True)
class DispatchRun:
    """A contiguous run of same-role subtasks that will become one
    specialist call.

    Mutable by design — the policy appends subtasks while grouping, then
    assigns `response` after awaiting the specialist. `synthesize_ordered`
    reads `response` + the subtask order for ordered trace assembly.
    """

    role: str
    subtasks: list[Subtask] = field(default_factory=list)
    response: OrchestratorResponse | None = None

    def append(self, subtask: Subtask) -> None:
        self.subtasks.append(subtask)

    def rendered_prompt(self) -> str:
        """Render this run's subtasks as one prompt for the specialist.

        Single-subtask runs return the subtask text verbatim — the
        specialist shouldn't know it was routed through a decomposer
        when there's nothing to decompose.

        Multi-subtask runs use a framing line + `Step N:` labels keyed
        by `Subtask.order + 1`. The framing tells the specialist to
        preserve the label structure in its answer, which is what
        `synthesize_ordered()` relies on to reassemble partials in
        order. `Step N:` (not `[Step N]`) is chosen because bracketed
        markers sometimes get echoed back as literals or mistaken for
        tool-call syntax by instruction-tuned models; a plain
        `Label: ...` pattern is the most consistent across the three
        specialists in the roster.
        """
        if len(self.subtasks) == 1:
            return self.subtasks[0].text

        header = (
            f"The following request has {len(self.subtasks)} sub-tasks. "
            "Answer each in order, and label each answer with the "
            "matching `Step N:` prefix so the structure is preserved."
        )
        body = "\n\n".join(
            f"Step {s.order + 1}: {s.text}" for s in self.subtasks
        )
        return f"{header}\n\n{body}"


# --------------------------------------------------------------------------- #
# Router protocol + keyword fixture
# --------------------------------------------------------------------------- #


class Router(Protocol):
    """Pure classifier: subtask text (+ optional context) → routing
    decision. No LLM calls, no orchestrator dependency — trainable and
    testable in isolation."""

    def classify(self, prompt: str, *, context: str = "") -> RoutingDecision: ...


class KeywordRouter:
    """Wraps P3's `classify_task` regex set, collapsing its 6-class
    `TaskType` output to the 3-role action space via the placeholder
    map above.

    Not a learned router — exists so Steps 2, 3, and 4b can depend on
    `Router` without waiting for Step 4/5. Also becomes the permanent
    unit-test fixture for the policy.

    Confidence is always 1.0 (regex hit) or a fallback signal; this is
    intentionally coarse — callers that need calibrated scores should
    use `LearnedRouter`.
    """

    def __init__(
        self,
        *,
        fallback_role: str = "fact_general",
        tasktype_to_role: dict[TaskType, str] | None = None,
    ) -> None:
        self.fallback_role = fallback_role
        self._tasktype_to_role = tasktype_to_role or _PLACEHOLDER_TASKTYPE_TO_ROLE

    def classify(self, prompt: str, *, context: str = "") -> RoutingDecision:
        # Context is ignored by the keyword fixture — `LearnedRouter`
        # will use it; we keep the parameter for protocol parity.
        del context

        task_type = classify_task(prompt)
        role = self._tasktype_to_role.get(task_type)
        if role is None:
            # TaskType not in the placeholder map — treat as low-confidence
            # fallback rather than raising, so malformed orchestrator state
            # degrades instead of crashing the whole run.
            return RoutingDecision(
                role=self.fallback_role,
                scores={self.fallback_role: 0.0},
                confidence=0.0,
                fallback_used=True,
                fallback_reason=f"tasktype_unmapped:{task_type.value}",
            )

        return RoutingDecision(
            role=role,
            scores={role: 1.0},
            confidence=1.0,
        )


# --------------------------------------------------------------------------- #
# Learned router
# --------------------------------------------------------------------------- #


class ScoreFn(Protocol):
    """Adapter seam between the decision logic and the ML model.

    One method — text in, role→probability map out. The HF-backed
    implementation (distilroberta fine-tune loaded via
    `transformers.from_pretrained`) ships alongside the training script;
    unit tests inject a deterministic fake so the router logic can be
    exercised without torch.

    Probabilities SHOULD sum to ~1.0 (i.e. a softmax over the role
    action space) but `LearnedRouter` does not re-normalize — it trusts
    the adapter. Calibration is the adapter's problem, not the
    router's.
    """

    def __call__(self, text: str) -> dict[str, float]: ...


class LearnedRouter:
    """
    Fine-tuned classifier-based router. Pure decision logic:
    tokenized-input formatting → `ScoreFn` call → floor → dispatch
    decision. No model IO here.

    Parameters
    ----------
    score_fn:
        Any `ScoreFn`. Production wires the HF adapter; tests pass a
        fake that returns canned dicts.
    fallback_role:
        Role used when top-1 probability is below `floor`. Must be a
        key that the orchestrator registered — validated lazily in the
        policy, not here (the router has no orchestrator handle).
    floor:
        Minimum top-1 probability required to route to the learned
        argmax. Below this, we route to `fallback_role` and flag
        `fallback_used=True`. Default 0.3 matches the training plan;
        calibrate from dev-set ROC once training lands.
    context_char_cap:
        Hard cap on context text fed to the model, before tokenization.
        The tokenizer truncates again to its own max length — this is
        an upstream guard so we don't waste tokenizer time on text
        that will be dropped. Per the P4 plan: 2048 chars.

    Scoring on fallback
    -------------------
    When confidence < floor, we KEEP the learned score distribution
    and report `confidence=max(scores.values())`. Rationale: the
    learned scores are the most useful telemetry for calibrating
    `floor` later. Overwriting with `{fallback_role: 0.0}` would match
    `KeywordRouter`'s fallback shape but discards the "how uncertain"
    signal that the learned model actually produced.

    Callers distinguish fallback-vs-routed via `fallback_used`, not
    via inspecting `scores`.
    """

    def __init__(
        self,
        *,
        score_fn: ScoreFn,
        fallback_role: str,
        floor: float = 0.3,
        context_char_cap: int = DEFAULT_CONTEXT_CHAR_CAP,
    ) -> None:
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {floor}")
        if context_char_cap < 0:
            raise ValueError(
                f"context_char_cap must be >= 0, got {context_char_cap}"
            )
        self._score_fn = score_fn
        self.fallback_role = fallback_role
        self.floor = floor
        self.context_char_cap = context_char_cap

    def classify(self, prompt: str, *, context: str = "") -> RoutingDecision:
        text = self._format_input(prompt, context)
        scores = self._score_fn(text)

        if not scores:
            logger.warning(
                "LearnedRouter: score_fn returned empty dict; "
                "falling back to %s",
                self.fallback_role,
            )
            return RoutingDecision(
                role=self.fallback_role,
                scores={},
                confidence=0.0,
                fallback_used=True,
                fallback_reason="empty_scores",
            )

        top_role, top_score = max(scores.items(), key=lambda kv: kv[1])

        if top_score < self.floor:
            return RoutingDecision(
                role=self.fallback_role,
                scores=scores,
                confidence=top_score,
                fallback_used=True,
                fallback_reason="low_confidence",
            )

        return RoutingDecision(
            role=top_role,
            scores=scores,
            confidence=top_score,
        )

    def _format_input(self, prompt: str, context: str) -> str:
        """Delegate to `router_featurize.featurize` — single source of
        truth shared with training. See that module's docstring for
        the training/serving-skew rationale."""
        return featurize(
            prompt, context, context_char_cap=self.context_char_cap
        )
