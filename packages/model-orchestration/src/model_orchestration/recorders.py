from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from model_orchestration.models import OrchestratorCallRecord


class BaseRecorder(ABC):
    output_path: Path | None = None

    def record_request(self, *, target: str, model: str, request: dict[str, Any]) -> str:
        del target, model, request
        return uuid4().hex

    @abstractmethod
    def record_response(self, record: OrchestratorCallRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_error(self, record: OrchestratorCallRecord) -> None:
        raise NotImplementedError


class NoOpRecorder(BaseRecorder):
    def record_response(self, record: OrchestratorCallRecord) -> None:
        del record

    def record_error(self, record: OrchestratorCallRecord) -> None:
        del record


class JSONLRecorder(BaseRecorder):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record_response(self, record: OrchestratorCallRecord) -> None:
        self._append(record)

    def record_error(self, record: OrchestratorCallRecord) -> None:
        self._append(record)

    def _append(self, record: OrchestratorCallRecord) -> None:
        payload = asdict(record)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


class InMemoryRecorder(BaseRecorder):
    def __init__(self) -> None:
        self.records: list[OrchestratorCallRecord] = []

    def record_response(self, record: OrchestratorCallRecord) -> None:
        self.records.append(record)

    def record_error(self, record: OrchestratorCallRecord) -> None:
        self.records.append(record)
