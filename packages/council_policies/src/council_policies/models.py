from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from model_orchestration import OrchestratorResponse


class TaskType(str, Enum):
    QA = "qa"
    REASONING = "reasoning"
    MATH = "math"
    CODE = "code"
    GENERAL = "general"
    FEVER = "fever"


TASK_TO_ROLE: dict[TaskType, str] = {
    TaskType.QA: "qa",
    TaskType.REASONING: "reasoning",
    TaskType.MATH: "math",
    TaskType.CODE: "code",
    TaskType.GENERAL: "general",
    TaskType.FEVER: "fever",
}


@dataclass(slots=True, frozen=True)
class CouncilResponse:
    winner: OrchestratorResponse
    policy: str
    task_type: TaskType | None = None
    candidates: tuple[OrchestratorResponse, ...] = ()
    vote_tally: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.winner.text
