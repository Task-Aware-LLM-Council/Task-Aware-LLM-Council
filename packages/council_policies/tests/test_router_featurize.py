"""
Tests for `router_featurize.featurize`.

Trivial-looking, but load-bearing: inference and training call this
exact function. A silent format change here ≡ silent accuracy
regression in production.
"""

from __future__ import annotations

from council_policies.p4.router_featurize import (
    DEFAULT_CONTEXT_CHAR_CAP,
    featurize,
)


def test_no_context_returns_question_verbatim():
    assert featurize("Solve x^2") == "Solve x^2"


def test_blank_context_treated_as_no_context():
    """Whitespace-only context must not produce a trailing `Context:`
    block — the empty block would look like a real signal to the
    classifier."""
    assert featurize("Solve x^2", "   \n\t ") == "Solve x^2"


def test_context_appended_with_label():
    assert (
        featurize("Q?", "some background")
        == "Q?\n\nContext:\nsome background"
    )


def test_context_is_truncated_at_char_cap():
    big = "x" * (DEFAULT_CONTEXT_CHAR_CAP + 500)
    out = featurize("Q?", big)
    # Everything after `Context:\n` is the truncated context body.
    body = out.split("Context:\n", 1)[1]
    assert len(body) == DEFAULT_CONTEXT_CHAR_CAP
    assert body == "x" * DEFAULT_CONTEXT_CHAR_CAP


def test_custom_char_cap_honored():
    out = featurize("Q?", "y" * 200, context_char_cap=50)
    body = out.split("Context:\n", 1)[1]
    assert len(body) == 50
