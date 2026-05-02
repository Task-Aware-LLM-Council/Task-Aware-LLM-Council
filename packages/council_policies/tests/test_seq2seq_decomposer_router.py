"""
Tests for `Seq2SeqDecomposerRouter` — the joint decomposer + router
serving class.

Unlike `test_decomposer_llm.py`, this file injects a synchronous
`GenerateFn` fake (text in, text out) rather than an async LLM client
— the joint model runs in-process via HF `transformers`, not as an
external LLM call. The parser ladder is shared with `LLMDecomposer`
via `_parse_subtask_array`, so we focus on:

  * the `GenerateFn` seam (happy path, raises → passthrough)
  * role-vocab pass-through (validation lives in the policy, NOT here)
  * featurize contract — byte-exact prompt construction, because
    train/serve skew here silently wrecks accuracy
  * parser ladder sanity-check (thin because the heavy ladder lives
    in test_decomposer_llm.py)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from council_policies.p4.router import Subtask
from council_policies.p4.router_featurize import DEFAULT_CONTEXT_CHAR_CAP, featurize
from council_policies.p4.seq2seq_decomposer_router import (
    INPUT_PREFIX,
    Seq2SeqDecomposerRouter,
)


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


@dataclass
class FakeGenerate:
    """Sync `GenerateFn` stand-in. Records the last input the serving
    class sent so featurize-contract tests can inspect it byte-exactly."""

    text: str = "[]"
    fail: bool = False
    last_input: str | None = field(default=None)

    def __call__(self, text: str) -> str:
        self.last_input = text
        if self.fail:
            raise RuntimeError("boom")
        return self.text


def _json(items: list[dict]) -> str:
    return json.dumps(items)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_rejects_zero_max_subtasks():
    with pytest.raises(ValueError, match="max_subtasks"):
        Seq2SeqDecomposerRouter(generate_fn=FakeGenerate(), max_subtasks=0)


def test_rejects_negative_context_char_cap():
    with pytest.raises(ValueError, match="context_char_cap"):
        Seq2SeqDecomposerRouter(generate_fn=FakeGenerate(), context_char_cap=-1)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_two_subtasks_with_valid_roles():
    fake = FakeGenerate(
        text=_json(
            [
                {"role": "math_code", "subtask": "compute 2+2"},
                {"role": "qa_reasoning", "subtask": "summarize the result"},
            ]
        )
    )
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("compute 2+2 then summarize")

    assert subtasks == [
        Subtask(text="compute 2+2", order=0, suggested_role="math_code"),
        Subtask(
            text="summarize the result", order=1, suggested_role="qa_reasoning"
        ),
    ]


@pytest.mark.asyncio
async def test_single_subtask_preserves_short_circuit_shape():
    """Single-element list must round-trip — the P4 policy's
    short-circuit path depends on `len(subtasks) == 1` to skip synth."""
    fake = FakeGenerate(
        text=_json([{"role": "fact_general", "subtask": "Paris is in France"}])
    )
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("verify a fact")

    assert len(subtasks) == 1
    assert subtasks[0].suggested_role == "fact_general"


# --------------------------------------------------------------------------- #
# Role-vocab pass-through (validation deferred to policy)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unknown_role_is_passed_through_verbatim():
    """Role validation lives in `LearnedRouterPolicy.plan()` (needs
    `PolicyRuntime.has_specialist_role`). The serving class must NOT
    rewrite unknown roles — it hands the raw string to the policy so
    the policy can stamp `fallback_reason="joint_model_unknown_role"`."""
    fake = FakeGenerate(
        text=_json([{"role": "math-code", "subtask": "do it"}])  # typo
    )
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("whatever")

    assert subtasks[0].suggested_role == "math-code"


@pytest.mark.asyncio
async def test_missing_role_yields_none_hint():
    """Missing `role` field → `suggested_role=None`. Policy interprets
    this as `joint_model_missing_role`."""
    fake = FakeGenerate(text=_json([{"subtask": "do it"}]))
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("whatever")

    assert subtasks[0].suggested_role is None


# --------------------------------------------------------------------------- #
# Fallback ladder — covered in depth by test_decomposer_llm.py; smoke-check
# here to prove the shared parser is wired correctly.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fallback_on_generate_exception():
    fake = FakeGenerate(fail=True)
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("original prompt")

    assert subtasks == [Subtask(text="original prompt", order=0)]


@pytest.mark.asyncio
async def test_fallback_on_malformed_json():
    fake = FakeGenerate(text='[{"role": "math_code", "subtask":]')
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("original")

    assert subtasks == [Subtask(text="original", order=0)]


@pytest.mark.asyncio
async def test_fallback_on_empty_list():
    fake = FakeGenerate(text="[]")
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    subtasks = await d.decompose("original")

    assert subtasks == [Subtask(text="original", order=0)]


# --------------------------------------------------------------------------- #
# max_subtasks truncation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_truncates_to_max_subtasks():
    fake = FakeGenerate(
        text=_json(
            [
                {"role": "math_code", "subtask": "a"},
                {"role": "math_code", "subtask": "b"},
                {"role": "qa_reasoning", "subtask": "c"},
                {"role": "fact_general", "subtask": "d"},
                {"role": "fact_general", "subtask": "e"},
            ]
        )
    )
    d = Seq2SeqDecomposerRouter(generate_fn=fake, max_subtasks=3)

    subtasks = await d.decompose("long prompt")

    assert [s.text for s in subtasks] == ["a", "b", "c"]
    assert [s.order for s in subtasks] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Featurize contract — byte-exact guard against train/serve skew
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generate_input_is_prefix_plus_featurize_no_context():
    fake = FakeGenerate(text=_json([{"role": "math_code", "subtask": "x"}]))
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    await d.decompose("question only, no context")

    expected = INPUT_PREFIX + featurize("question only, no context", "")
    assert fake.last_input == expected
    # Sanity: no-context featurize must NOT inject "Context:" — otherwise
    # training-time examples with context="" become structurally different
    # from serving-time examples with context="".
    assert "Context:" not in (fake.last_input or "")


@pytest.mark.asyncio
async def test_generate_input_is_prefix_plus_featurize_with_context():
    fake = FakeGenerate(text=_json([{"role": "math_code", "subtask": "x"}]))
    d = Seq2SeqDecomposerRouter(generate_fn=fake)

    await d.decompose("main question", context="supporting evidence here")

    expected = INPUT_PREFIX + featurize("main question", "supporting evidence here")
    assert fake.last_input == expected


@pytest.mark.asyncio
async def test_context_char_cap_is_forwarded_to_featurize():
    """If a caller overrides the cap, `featurize` in the serving class
    must use that same cap — not `DEFAULT_CONTEXT_CHAR_CAP`. Drift here
    silently truncates differently at serve vs. train time."""
    fake = FakeGenerate(text=_json([{"role": "math_code", "subtask": "x"}]))
    custom_cap = 10
    d = Seq2SeqDecomposerRouter(generate_fn=fake, context_char_cap=custom_cap)

    long_ctx = "A" * 200
    await d.decompose("q", context=long_ctx)

    expected = INPUT_PREFIX + featurize("q", long_ctx, context_char_cap=custom_cap)
    assert fake.last_input == expected
    # Positive check: no more than `custom_cap` context characters leak through.
    assert fake.last_input.count("A") == custom_cap


def test_default_context_char_cap_matches_featurize_default():
    """Regression guard. If the shared default changes, both sides must
    move together — otherwise training and serving will disagree on how
    much context they see."""
    d = Seq2SeqDecomposerRouter(generate_fn=FakeGenerate())
    assert d.context_char_cap == DEFAULT_CONTEXT_CHAR_CAP
