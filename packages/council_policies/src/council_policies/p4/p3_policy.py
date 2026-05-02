from __future__ import annotations

import logging
import re

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator

from council_policies.p4.types import TASK_TO_ROLE, CouncilResponse, TaskType

logger = logging.getLogger(__name__)

_MATH_RE = re.compile(
    r"\b(solve|calculat|comput|integrat|differentiat|derivativ|equation|inequalit|"
    r"theorem|proof|algebra|geometr|trigonometr|probabilit|statistic|matric|vector|"
    r"polynomial|logarithm|exponential|fraction|percent|modulo|combinat|permut)\w*\b",
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"\b(code|function|implement|algorithm|debug|program|class|method|variable|"
    r"python|javascript|typescript|java|c\+\+|golang|rust|sql|bash|script|api|"
    r"loop|recursion|sort|parse|regex|test|unittest|bug|fix|refactor|deploy)\w*\b",
    re.IGNORECASE,
)

_REASONING_RE = re.compile(
    r"\b(reason|analys|infer|deduc|logic|argument|cause|effect|explain why|"
    r"compare|contrast|evaluat|assess|critic|implicat|consequenc|hypothes|"
    r"evidence|conclus|step.by.step|think through)\w*\b",
    re.IGNORECASE,
)

_FEVER_RE = re.compile(
    r"\b(true or false|fact.check|verif|claim|supports|refutes|evidence|"
    r"fact|accurate|correct|wrong|misinform|debunk)\w*\b",
    re.IGNORECASE,
)


def classify_task(prompt: str) -> TaskType:
    """Rule-based task classifier using keyword pattern matching."""
    text = prompt.lower()

    scores: dict[TaskType, int] = {
        TaskType.MATH: len(_MATH_RE.findall(text)),
        TaskType.CODE: len(_CODE_RE.findall(text)),
        TaskType.REASONING: len(_REASONING_RE.findall(text)),
        TaskType.FEVER: len(_FEVER_RE.findall(text)),
    }

    best_type, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score > 0:
        return best_type

    return TaskType.QA


# ---------------------------------------------------------------------------
# Lazy synthesis import — kept lazy so an import error in synthesis.py
# degrades to "return specialist response directly" instead of breaking
# the whole policy.
# ---------------------------------------------------------------------------
try:
    from council_policies.p4.synthesis import synthesize as _synthesize  # type: ignore[import-not-found]
    _SYNTHESIS_AVAILABLE = True
    logger.debug("council_policies.p4.synthesis loaded — P3 will use synthesis aggregation.")
except ImportError:
    _synthesize = None  # type: ignore[assignment]
    _SYNTHESIS_AVAILABLE = False
    logger.debug(
        "council_policies.p4.synthesis not found — P3 will return the specialist "
        "response directly until synthesis.py is created."
    )


class RuleBasedRoutingPolicy:
    """
    P3: Route each request to the specialist model best suited for the task
    using keyword-based classification.

    Aggregation (per README §Two aggregation paradigms)
    ---------------------------------------------------
    P3 uses *synthesis*, not voting.  When ``synthesis.py`` is present, the
    routed specialist's response is passed to ``synthesize()`` which short-
    circuits for a single-specialist call (no extra LLM call) and returns the
    response verbatim.  This keeps P3 ready for the multi-skill upgrade
    described in the README without any further changes to this file.

    Parameters
    ----------
    orchestrator:
        A fully-configured ModelOrchestrator.
    fallback_role:
        Role to use when the classified role is not registered in the
        orchestrator.  Validated at construction time.  Defaults to
        ``"general"``.
    synthesizer_role:
        The role whose model fuses partials when ``synthesis.py`` is
        available.  For single-specialist P3 this role is never actually
        called (short-circuit), but it must be registered so the upgraded
        multi-skill path works without reconfiguring callers.  Defaults to
        the dedicated ``"synthesizer"`` role (excluded from TASK_TO_ROLE).
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        *,
        fallback_role: str = "general",
        synthesizer_role: str = "synthesizer",
    ) -> None:
        self.orchestrator = orchestrator
        self.fallback_role = fallback_role
        self.synthesizer_role = synthesizer_role
        # Validate both roles at construction time so misconfiguration surfaces
        # immediately rather than silently at inference time. synthesizer_role
        # is validated here too so the multi-skill path can't blow up on first
        # fan-out just because someone forgot to register the referee.
        self._validate_role("fallback_role", self.fallback_role)
        self._validate_role("synthesizer_role", self.synthesizer_role)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_role(self, field_name: str, role: str) -> None:
        try:
            self.orchestrator.get_client(role)
        except KeyError as exc:
            raise ValueError(
                f"RuleBasedRoutingPolicy: {field_name} {role!r} is not registered "
                f"in the orchestrator. Register a model with that role or alias, "
                f"or pass a different {field_name}."
            ) from exc

    def classify(self, prompt: str) -> TaskType:
        return classify_task(prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        request: PromptRequest,
        *,
        task_type: TaskType | None = None,
    ) -> CouncilResponse:
        if task_type is None:
            task_type = self.classify(request.user_prompt or "")

        role = TASK_TO_ROLE[task_type]

        # Dispatch to the specialist; fall back gracefully if not configured.
        try:
            response = await self.orchestrator.get_client(role).get_response(request)
            routed_role = role
        except KeyError:
            logger.warning(
                "Role %r not configured; falling back to %r", role, self.fallback_role
            )
            # FIX: guard the fallback call — a missing fallback_role now raises
            # a clear RuntimeError instead of a bare KeyError.
            try:
                response = await self.orchestrator.get_client(
                    self.fallback_role
                ).get_response(request)
                routed_role = self.fallback_role
            except KeyError as exc:
                raise RuntimeError(
                    f"Primary role {role!r} and fallback role {self.fallback_role!r} "
                    f"are both missing from the orchestrator. "
                    f"Register at least one of them before running P3."
                ) from exc

        # ------------------------------------------------------------------
        # Aggregation via synthesis (README §Two aggregation paradigms)
        # ------------------------------------------------------------------
        # Even for a single specialist, we route through synthesize() so the
        # caller gets a consistent CouncilResponse shape and the policy is
        # ready for the multi-skill upgrade with no further changes.
        #
        # synthesize({role: response}) short-circuits when len(partials) == 1
        # and returns the single response verbatim — no extra LLM call.
        # ------------------------------------------------------------------
        if _SYNTHESIS_AVAILABLE:
            try:
                result = await _synthesize(
                    question=request.user_prompt or "",
                    partials={routed_role: response},
                    orchestrator=self.orchestrator,
                    synthesizer_role=self.synthesizer_role,
                )
                return CouncilResponse(
                    winner=result.response,
                    policy="p3",
                    task_type=task_type,
                    candidates=(response,),
                    metadata={
                        "routed_role": routed_role,
                        "synthesizer_role": self.synthesizer_role,
                        "synthesis_short_circuit": result.metadata.get("short_circuit"),
                        "synthesis_used_fallback": result.used_fallback,
                    },
                )
            except Exception as exc:  # pragma: no cover
                # Synthesis call failed — log and fall through to direct return
                # so a synthesis bug never silently kills the whole policy run.
                logger.warning(
                    "synthesize() raised %s — returning specialist response directly: %s",
                    type(exc).__name__,
                    exc,
                )

        # synthesis.py not yet created (or synthesis call failed above).
        # Return the response directly.
        # TODO: remove the non-synthesis fallback once synthesis.py is merged.
        return CouncilResponse(
            winner=response,
            policy="p3",
            task_type=task_type,
            candidates=(response,),
            metadata={
                "routed_role": routed_role,
                "synthesis_available": _SYNTHESIS_AVAILABLE,
            },
        )