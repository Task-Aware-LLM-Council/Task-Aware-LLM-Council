"""
Tests for `LLMDecomposer` — covers the parser's defensive behavior
against real-world LLM output shapes.

The decomposer emits object-array JSON:
    [{"role": "...", "subtask": "..."}, ...]

Each entry becomes a `Subtask(text=<subtask>, order=i, suggested_role=<role>)`.
The `FakeClient` here implements `BaseLLMClient`'s surface just enough
to feed canned response text through the parser. No real LLM is called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from llm_gateway import PromptRequest, PromptResponse

from council_policies.p4.decomposer import LLMDecomposer
from council_policies.p4.router import Subtask


@dataclass
class FakeLLMClient:
    """Minimal stand-in for `BaseLLMClient`. Returns canned text (or
    raises). Records the last request so prompt-construction tests can
    inspect what the decomposer sent."""

    text: str = "[]"
    fail: bool = False
    last_request: PromptRequest | None = None

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.last_request = request
        if self.fail:
            raise RuntimeError("boom")
        return PromptResponse(model="fake", text=self.text)


def _json(items: list[dict]) -> str:
    return json.dumps(items)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_rejects_zero_max_subtasks():
    with pytest.raises(ValueError, match="max_subtasks"):
        LLMDecomposer(client=FakeLLMClient(), max_subtasks=0)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_returns_ordered_subtasks_with_role_hints():
    client = FakeLLMClient(
        text=_json(
            [
                {"role": "math", "subtask": "Compute 2+2"},
                {"role": "code", "subtask": "Write a haiku formatter in Python"},
            ]
        )
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("Compute 2+2 and code a haiku formatter")

    assert subtasks == [
        Subtask(text="Compute 2+2", order=0, suggested_role="math"),
        Subtask(
            text="Write a haiku formatter in Python",
            order=1,
            suggested_role="code",
        ),
    ]


@pytest.mark.asyncio
async def test_single_role_prompt_returns_single_subtask():
    """Per the decomposer prompt contract: 'If the prompt requires only
    one role, return a list with one element.' The policy's
    short-circuit path depends on this to skip the synthesis phase."""
    client = FakeLLMClient(
        text=_json([{"role": "math", "subtask": "Solve x^2 - 5x + 6 = 0"}])
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("Solve x^2 - 5x + 6 = 0")

    assert len(subtasks) == 1
    assert subtasks[0].suggested_role == "math"


@pytest.mark.asyncio
async def test_parses_array_embedded_in_prose():
    """LLMs sometimes prepend 'Here are the subtasks:'. The parser must
    locate the JSON array regardless."""
    client = FakeLLMClient(
        text=(
            "Here are the subtasks:\n"
            + _json(
                [
                    {"role": "fact_verify", "subtask": "Verify Paris is in France"},
                    {"role": "qa_multihop", "subtask": "Explain its founding history"},
                ]
            )
            + "\n"
        )
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("Tell me about Paris")

    assert [s.text for s in subtasks] == [
        "Verify Paris is in France",
        "Explain its founding history",
    ]
    assert [s.suggested_role for s in subtasks] == ["fact_verify", "qa_multihop"]


@pytest.mark.asyncio
async def test_parses_array_inside_markdown_fences():
    client = FakeLLMClient(
        text='```json\n'
        + _json(
            [
                {"role": "math", "subtask": "step one"},
                {"role": "code", "subtask": "step two"},
            ]
        )
        + '\n```'
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("whatever")

    assert [s.text for s in subtasks] == ["step one", "step two"]


@pytest.mark.asyncio
async def test_missing_role_field_kept_with_none_hint():
    """The prompt asks for role + subtask, but a missing role shouldn't
    cause the item to be dropped — the router can still classify from
    text. Only missing subtasks are skipped."""
    client = FakeLLMClient(
        text=_json(
            [
                {"subtask": "do the thing"},
                {"role": "math", "subtask": "compute"},
            ]
        )
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original")

    assert subtasks[0].suggested_role is None
    assert subtasks[1].suggested_role == "math"


# --------------------------------------------------------------------------- #
# max_subtasks enforcement
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_truncates_when_llm_returns_more_than_max_subtasks():
    client = FakeLLMClient(
        text=_json(
            [
                {"role": "math", "subtask": "a"},
                {"role": "math", "subtask": "b"},
                {"role": "code", "subtask": "c"},
                {"role": "code", "subtask": "d"},
                {"role": "qa_multihop", "subtask": "e"},
            ]
        )
    )
    d = LLMDecomposer(client=client, max_subtasks=3)

    subtasks = await d.decompose("whatever")

    assert [s.text for s in subtasks] == ["a", "b", "c"]
    assert [s.order for s in subtasks] == [0, 1, 2]


@pytest.mark.asyncio
async def test_prompt_mentions_max_subtasks_cap():
    client = FakeLLMClient(text=_json([{"role": "math", "subtask": "single"}]))
    d = LLMDecomposer(client=client, max_subtasks=2)

    await d.decompose("question")

    assert client.last_request is not None
    assert "AT MOST 2" in (client.last_request.user_prompt or "")


# --------------------------------------------------------------------------- #
# Parser fallback — every failure mode maps to single-subtask passthrough
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fallback_on_no_json_array_in_response():
    client = FakeLLMClient(text="I could not decompose this question.")
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original prompt text")

    assert subtasks == [Subtask(text="original prompt text", order=0)]


@pytest.mark.asyncio
async def test_fallback_on_malformed_json():
    client = FakeLLMClient(text='[{"role": "math", "subtask": "missing close quote]')
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original")

    assert subtasks == [Subtask(text="original", order=0)]


@pytest.mark.asyncio
async def test_non_object_items_are_skipped():
    """Mixed-shape arrays (objects + bare strings) keep the objects and
    drop the strings — the string form isn't the schema we asked for."""
    client = FakeLLMClient(
        text='[{"role": "math", "subtask": "compute"}, "stray string"]'
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original")

    assert len(subtasks) == 1
    assert subtasks[0].text == "compute"


@pytest.mark.asyncio
async def test_fallback_on_empty_list():
    client = FakeLLMClient(text="[]")
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original")

    assert subtasks == [Subtask(text="original", order=0)]


@pytest.mark.asyncio
async def test_fallback_on_all_empty_subtasks():
    client = FakeLLMClient(
        text=_json(
            [
                {"role": "math", "subtask": ""},
                {"role": "code", "subtask": "   "},
            ]
        )
    )
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original")

    assert subtasks == [Subtask(text="original", order=0)]


@pytest.mark.asyncio
async def test_fallback_on_client_exception():
    """LLM call raises → decomposer must not propagate. A transient
    provider failure shouldn't kill an entire P4 batch."""
    client = FakeLLMClient(fail=True)
    d = LLMDecomposer(client=client)

    subtasks = await d.decompose("original")

    assert subtasks == [Subtask(text="original", order=0)]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_context_is_included_in_user_prompt_when_provided():
    client = FakeLLMClient(text=_json([{"role": "math", "subtask": "do the thing"}]))
    d = LLMDecomposer(client=client)

    await d.decompose("question", context="relevant background facts")

    up = client.last_request.user_prompt or ""
    assert "relevant background facts" in up
    assert "question" in up


@pytest.mark.asyncio
async def test_context_is_omitted_when_blank():
    client = FakeLLMClient(text=_json([{"role": "math", "subtask": "do the thing"}]))
    d = LLMDecomposer(client=client)

    await d.decompose("question", context="   ")

    up = client.last_request.user_prompt or ""
    assert "Context:" not in up


@pytest.mark.asyncio
async def test_system_prompt_default_enumerates_roles():
    """Regression guard on the shipped default prompt — if someone
    rewrites it without including the role vocabulary the downstream
    router no longer has a documented target set."""
    from council_policies.p4.decomposer import DEFAULT_DECOMPOSER_SYSTEM_PROMPT

    for role in ("math", "code", "fact_verify", "qa_multihop", "qa_longctx"):
        assert role in DEFAULT_DECOMPOSER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_system_prompt_is_forwarded():
    client = FakeLLMClient(text=_json([{"role": "math", "subtask": "a"}]))
    d = LLMDecomposer(client=client, system_prompt="CUSTOM SYSTEM")

    await d.decompose("question")

    assert client.last_request.system_prompt == "CUSTOM SYSTEM"


@pytest.mark.asyncio
async def test_temperature_and_model_forwarded():
    client = FakeLLMClient(text=_json([{"role": "math", "subtask": "a"}]))
    d = LLMDecomposer(client=client, model="gpt-4", temperature=0.2)

    await d.decompose("question")

    assert client.last_request.model == "gpt-4"
    assert client.last_request.temperature == 0.2
