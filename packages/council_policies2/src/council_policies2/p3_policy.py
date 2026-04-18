from __future__ import annotations

import logging
import re

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator

from council_policies2.models import TASK_TO_ROLE, CouncilResponse, TaskType

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


class RuleBasedRoutingPolicy:
    """
    P3: Route each request to the specialist model best suited for the task
    using keyword-based classification.

    Parameters
    ----------
    orchestrator:
        A fully-configured ModelOrchestrator.
    fallback_role:
        Role to use when the classified role is missing from the orchestrator.
        Defaults to ``"general"``.

    Notes
    -----
    FIX (bug 4): The constructor validates that ``fallback_role`` is registered
    so misconfiguration is caught early, not silently at inference time.
    FIX (bug 4): The fallback call to ``get_client(fallback_role)`` is now also
    wrapped in a try/except so a missing fallback role raises a clear
    ``RuntimeError`` instead of a confusing bare ``KeyError``.
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        *,
        fallback_role: str = "general",
    ) -> None:
        self.orchestrator = orchestrator
        self.fallback_role = fallback_role
        # Validate at construction time so the error surfaces immediately.
        self._validate_fallback_role()

    def _validate_fallback_role(self) -> None:
        try:
            self.orchestrator.get_client(self.fallback_role)
        except KeyError as exc:
            raise ValueError(
                f"RuleBasedRoutingPolicy: fallback_role {self.fallback_role!r} is not "
                f"registered in the orchestrator. Register a model with that role or "
                f"alias, or pass a different fallback_role."
            ) from exc

    def classify(self, prompt: str) -> TaskType:
        return classify_task(prompt)

    async def run(
        self,
        request: PromptRequest,
        *,
        task_type: TaskType | None = None,
    ) -> CouncilResponse:
        if task_type is None:
            task_type = self.classify(request.user_prompt or "")

        role = TASK_TO_ROLE[task_type]
        try:
            response = await self.orchestrator.get_client(role).get_response(request)
        except KeyError:
            logger.warning(
                "Role %r not configured; falling back to %r", role, self.fallback_role
            )
            # FIX (bug 4): guard the fallback call with a clear RuntimeError.
            try:
                response = await self.orchestrator.get_client(self.fallback_role).get_response(request)
            except KeyError as exc:
                raise RuntimeError(
                    f"Primary role {role!r} and fallback role {self.fallback_role!r} "
                    f"are both missing from the orchestrator."
                ) from exc

        return CouncilResponse(
            winner=response,
            policy="p3",
            task_type=task_type,
            candidates=(response,),
            metadata={"routed_role": role},
        )
