"""
Unit tests for `decomposer.py` — protocol + `PassthroughDecomposer`.

The LLM-backed `LLMDecomposer` (Step 3) has its own test file when it
lands. This file covers only the types shipped in Step 1.
"""

from __future__ import annotations

import pytest

from council_policies.p4.decomposer import PassthroughDecomposer
from council_policies.p4.router import Subtask


@pytest.mark.asyncio
async def test_passthrough_returns_single_subtask():
    d = PassthroughDecomposer()
    subtasks = await d.decompose("What is 2 + 2?")

    assert len(subtasks) == 1
    assert subtasks[0] == Subtask(text="What is 2 + 2?", order=0)


@pytest.mark.asyncio
async def test_passthrough_preserves_text_verbatim():
    """Whitespace, newlines, special chars — passthrough must not touch
    the prompt. Any normalization is the specialist's / decomposer's job."""
    d = PassthroughDecomposer()
    prompt = "  multi\nline\n\tprompt with\ttabs  "
    subtasks = await d.decompose(prompt)
    assert subtasks[0].text == prompt


@pytest.mark.asyncio
async def test_passthrough_ignores_context():
    """Context is accepted for protocol parity; passthrough must not fail
    or alter its output when context is provided."""
    d = PassthroughDecomposer()
    without = await d.decompose("prompt")
    with_ctx = await d.decompose("prompt", context="irrelevant background")
    assert without == with_ctx
