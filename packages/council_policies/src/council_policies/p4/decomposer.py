"""
P4 decomposer types + fixtures.

The decomposer splits a user prompt into an ordered list of `Subtask`s.
One LLM call in the `LLMDecomposer` implementation; zero LLM calls for
the `PassthroughDecomposer` fixture.

Both implement the `Decomposer` protocol so the policy never branches on
which one is wired in.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from llm_gateway import BaseLLMClient, PromptRequest

from council_policies.p4.router import Subtask

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


class Decomposer(Protocol):
    """Split a prompt into ordered subtasks. One LLM call in production;
    fixtures may be synchronous under the hood but the protocol is async
    so call sites don't branch on implementation."""

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]: ...


# --------------------------------------------------------------------------- #
# Passthrough (no LLM call)
# --------------------------------------------------------------------------- #


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
        del context
        return [Subtask(text=prompt, order=0)]


# --------------------------------------------------------------------------- #
# LLM-backed decomposer
# --------------------------------------------------------------------------- #


DEFAULT_DECOMPOSER_SYSTEM_PROMPT = """\
You decompose a user prompt into the minimum number of independent subtasks \
needed to answer it fully, where each subtask maps to exactly one specialist role.

Available roles: math, code, fact_verify, qa_multihop, qa_longctx.

Rules:
- If the prompt requires only one role, return a list with one element. \
Do not split single-skill prompts.
- If the prompt genuinely requires two or more distinct roles (e.g. a math \
computation AND a code implementation), return one element per role.
- Each subtask must be self-contained: include all context the specialist \
needs to answer it without seeing the other subtasks.
- Never decompose along reasoning steps within a single role \
(e.g. do not split a multi-step math problem into sub-problems).
- Return a JSON array of objects: \
[{"role": "<role>", "subtask": "<self-contained prompt text>"}, ...]
- Return JSON only. No preamble, no explanation.\
"""


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_subtask_array(
    text: str, *, fallback_prompt: str, max_subtasks: int,
) -> list[Subtask]:
    """Extract a JSON array of `{"role", "subtask"}` objects from an LLM
    or seq2seq-model response into an ordered list of `Subtask`s.

    Shared between `LLMDecomposer` (external LLM decomposer) and
    `Seq2SeqDecomposerRouter` (the joint seq2seq model), so both honor
    the same fallback ladder:

      - no JSON array found             → single-passthrough Subtask
      - JSON decode failure             → single-passthrough Subtask
      - parsed value is not a list      → single-passthrough Subtask
      - list item not a dict            → skip item
      - item missing/empty `subtask`    → skip item
      - no usable items after filtering → single-passthrough Subtask

    `role` is passed through verbatim as `suggested_role` (after strip);
    role-vocab validation is the caller's responsibility — the parser
    doesn't know which label space is canonical.
    """
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        logger.warning(
            "subtask parse: no JSON array in response; "
            "falling back to passthrough. raw=%r",
            text[:200],
        )
        return [Subtask(text=fallback_prompt, order=0)]

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning(
            "subtask parse: JSON decode failed (%s); falling back. raw=%r",
            exc, match.group(0)[:200],
        )
        return [Subtask(text=fallback_prompt, order=0)]

    if not isinstance(parsed, list):
        logger.warning(
            "subtask parse: parsed JSON is %s, not list; falling back.",
            type(parsed).__name__,
        )
        return [Subtask(text=fallback_prompt, order=0)]

    subtasks: list[Subtask] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            logger.warning(
                "subtask parse: item %d is %s, not object; skipping.",
                index, type(item).__name__,
            )
            continue
        subtask_text = str(item.get("subtask", "")).strip()
        if not subtask_text:
            logger.warning(
                "subtask parse: item %d has no 'subtask' field; skipping.",
                index,
            )
            continue
        role_hint = item.get("role")
        role_value = str(role_hint).strip() if role_hint is not None else ""
        subtasks.append(
            Subtask(
                text=subtask_text,
                order=len(subtasks),
                suggested_role=role_value or None,
            )
        )
        if len(subtasks) >= max_subtasks:
            break

    if not subtasks:
        logger.warning(
            "subtask parse: parsed list contained no usable items; "
            "falling back to passthrough."
        )
        return [Subtask(text=fallback_prompt, order=0)]

    return subtasks


class LLMDecomposer:
    """
    LLM-backed decomposer. Calls a `BaseLLMClient` once per `decompose()`
    to produce a JSON array of subtask strings, then parses into
    `Subtask`s preserving array order.

    Parameters
    ----------
    client:
        Any `BaseLLMClient`. The decomposer calls `client.generate()`
        once; it does NOT share the specialist or synthesizer
        orchestrators from `PolicyRuntime` — decomposition is a
        pre-phase, not part of the two-phase dispatch.
    system_prompt:
        The instruction text given to the decomposer LLM. Defaults to
        `DEFAULT_DECOMPOSER_SYSTEM_PROMPT`; override per benchmark
        domain.
    max_subtasks:
        Hard ceiling enforced at parse time. Even if the LLM returns
        more, we truncate. The policy `LearnedRouterPolicy.max_subtasks`
        is redundant defense in depth.
    model:
        Optional model override passed through on the `PromptRequest`.
        `None` lets the client pick its default.
    temperature:
        Decomposition is deterministic-ish; default 0.0 keeps subtask
        lists reproducible across runs.

    Fallback behavior
    -----------------
    Any of { parse failure, empty result, non-list JSON, empty strings
    after filtering } falls back to a single passthrough subtask (the
    original prompt). A warning is logged so benchmark runs can surface
    decomposer failure rates without crashing.
    """

    def __init__(
        self,
        *,
        client: BaseLLMClient,
        system_prompt: str = DEFAULT_DECOMPOSER_SYSTEM_PROMPT,
        max_subtasks: int = 4,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if max_subtasks < 1:
            raise ValueError(f"max_subtasks must be >= 1, got {max_subtasks}")
        self.client = client
        self.system_prompt = system_prompt
        self.max_subtasks = max_subtasks
        self.model = model
        self.temperature = temperature

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]:
        request = PromptRequest(
            system_prompt=self.system_prompt,
            user_prompt=self._build_user_prompt(prompt, context),
            model=self.model,
            temperature=self.temperature,
        )
        try:
            response = await self.client.generate(request)
        except Exception as exc:
            logger.warning(
                "LLMDecomposer client call failed (%s); falling back to passthrough",
                type(exc).__name__,
            )
            return [Subtask(text=prompt, order=0)]

        return self._parse_response(response.text or "", fallback_prompt=prompt)

    def _build_user_prompt(self, prompt: str, context: str) -> str:
        parts = []
        if context.strip():
            parts.append(f"Context:\n{context.strip()}")
        parts.append(f"Prompt:\n{prompt.strip()}")
        parts.append(
            f"Decompose into AT MOST {self.max_subtasks} subtasks. "
            "Return the JSON array only."
        )
        return "\n\n".join(parts)

    def _parse_response(self, text: str, *, fallback_prompt: str) -> list[Subtask]:
        """Thin wrapper over the shared `_parse_subtask_array` helper so
        `LLMDecomposer` and `Seq2SeqDecomposerRouter` share one parser +
        fallback ladder."""
        return _parse_subtask_array(
            text,
            fallback_prompt=fallback_prompt,
            max_subtasks=self.max_subtasks,
        )
