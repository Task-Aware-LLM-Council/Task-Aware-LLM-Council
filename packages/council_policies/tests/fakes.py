"""Shared test doubles for council_policies tests."""
from __future__ import annotations

from types import SimpleNamespace

from llm_gateway import PromptRequest, PromptResponse
from model_orchestration import OrchestratorResponse
from model_orchestration.models import OrchestratorCallRecord


def make_orch_response(role: str, text: str) -> OrchestratorResponse:
    """Build a minimal valid OrchestratorResponse for a fake client."""
    prompt_req = PromptRequest(user_prompt="__fake__")
    prompt_resp = PromptResponse(model=f"{role}-model", text=text)
    call_record = OrchestratorCallRecord(
        event_id=f"evt-{role}",
        started_at="2026-04-18T00:00:00",
        completed_at="2026-04-18T00:00:01",
        target=role,
        resolved_alias=role,
        model=f"{role}-model",
        provider="fake",
        provider_mode="test",
        request={},
        request_metadata={},
    )
    return OrchestratorResponse(
        target=role,
        resolved_alias=role,
        model=f"{role}-model",
        provider="fake",
        provider_mode="test",
        request=prompt_req,
        prompt_response=prompt_resp,
        started_at="2026-04-18T00:00:00",
        completed_at="2026-04-18T00:00:01",
        call_record=call_record,
    )


# keep old name as alias so existing test files don't break
fake_orch_response = make_orch_response


class FakeClient:
    """Records requests; returns canned text or raises when fail=True."""

    def __init__(self, role: str, *, text: str | None = None, fail: bool = False) -> None:
        self.role = role
        self._text = text if text is not None else f"response-from:{role}"
        self.fail = fail
        self.requests: list[PromptRequest] = []

    async def get_response(
        self, request: PromptRequest | None = None, **kwargs
    ) -> OrchestratorResponse:
        self.requests.append(request or PromptRequest())
        if self.fail:
            raise RuntimeError(f"boom:{self.role}")
        return make_orch_response(self.role, self._text)


class FakeOrchestrator:
    """Minimal stand-in for ModelOrchestrator — only get_client(role)."""

    def __init__(self, clients: dict[str, FakeClient]) -> None:
        self._clients = clients

    def get_client(self, role: str) -> FakeClient:
        if role not in self._clients:
            raise KeyError(role)
        return self._clients[role]


def make_orchestrator(*roles: str, texts: dict[str, str] | None = None) -> FakeOrchestrator:
    """Create a FakeOrchestrator with one FakeClient per role."""
    texts = texts or {}
    return FakeOrchestrator(
        {role: FakeClient(role, text=texts.get(role, "")) for role in roles}
    )


def make_case(
    question: str = "q",
    dataset: str = "test",
    eid: str = "ex-1",
    context: str | None = None,
    system_prompt: str | None = None,
    messages: tuple = (),
) -> SimpleNamespace:
    """Build a duck-typed EvaluationCase stand-in without importing task_eval.

    Policies access only .example.{example_id, dataset_name, question,
    context, system_prompt, messages} — SimpleNamespace satisfies that
    contract without triggering the HuggingFace `datasets` import chain.
    """
    example = SimpleNamespace(
        example_id=eid,
        dataset_name=dataset,
        question=question,
        context=context,
        system_prompt=system_prompt,
        messages=messages,
    )
    return SimpleNamespace(example=example)
