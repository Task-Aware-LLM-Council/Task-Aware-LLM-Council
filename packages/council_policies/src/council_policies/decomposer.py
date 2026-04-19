"""
P4 decomposer types + passthrough fixture.

The decomposer splits a user prompt into an ordered list of `Subtask`s.
One LLM call in the full implementation; zero LLM calls for the
passthrough fixture shipped here.

The LLM-backed `LLMDecomposer` lands in Step 3 with its prompt and
parser. This module only ships the protocol and a `PassthroughDecomposer`
so the policy rewrite (Step 2) and the `len(runs) == 1` short-circuit
path have a working decomposer to test against.
"""

from __future__ import annotations

from typing import Protocol

from council_policies.router import Subtask


class Decomposer(Protocol):
    """Split a prompt into ordered subtasks. One LLM call in production;
    fixtures may be synchronous under the hood but the protocol is async
    so call sites don't branch on implementation."""

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]: ...


class PassthroughDecomposer:
    """No-op decomposer: returns the prompt as a single subtask.

    Used for:
      * The P4 single-skill fast path test — with this decomposer the
        policy must behave identically to P3 (one run → short-circuit
        synthesize → specialist response verbatim).
      * Any caller that wants P4's router + dispatch machinery without
        paying the decomposer LLM hop.
    """

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]:
        # Context is accepted for protocol parity and ignored — passthrough
        # by definition does not look at the prompt shape.
        del context
        return [Subtask(text=prompt, order=0)]
