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

The `LearnedRouter` implementation lands in Step 5; this module only
ships the types, the protocol, and a `KeywordRouter` fixture that wraps
P3's regex classifier so the policy (Step 2) and the ordered-synthesis
entry point (Step 4b) have something real to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from model_orchestration import OrchestratorResponse

from council_policies.models import TaskType
from council_policies.p3_policy import classify_task

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
    pre-sorted."""

    text: str
    order: int


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
