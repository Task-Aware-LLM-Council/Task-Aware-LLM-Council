"""
Build the joint decomposer + router training dataset.

Pipeline:
    1. Load hand-annotated gold JSONL from `--gold`.
         Schema per row:
             {
               "question": str,
               "context":  str,
               "skill_tags": [str, ...],
               "targets": [{"role": <math_code|qa_reasoning|fact_general>,
                            "subtask": str}, ...]
             }
         Hold out `--gold-eval-frac` (default 20%) deterministically as the
         `gold_eval` split. Never used for training or early stopping.

    2. Load source pool `task-aware-llm-council/router-dataset` (dev split)
         and call a teacher LLM once per prompt with `TEACHER_SYSTEM_PROMPT`
         to produce a synthetic decomposition in the SAME JSON shape the
         gold rows use. Checkpoint every N prompts to `--out/_teacher_cache/`
         so failures don't restart the whole call.

    3. Filter synthetic rows with three gates:
         a. Teacher output parses via `_JSON_ARRAY_RE` + `json.loads`.
         b. Every emitted `role` is in `ROLE_LABELS`.
         c. `role_from_tags(row.skill_tags)` appears among emitted roles
             (consistency with source row's oracle label — guards against
             teacher hallucination).

    4. Teacher-agree-with-gold gate: run the teacher on the gold.train
         portion and compare role accuracy (per-slot, conditional on shape
         match). If disagreement > `--teacher-disagreement-ceiling`
         (default 0.25), halt — the teacher prompt needs revision before
         we bake thousands of synthetic rows on it.

    5. Publish HF dataset with splits {train, dev, mini_test, gold_eval}.
         train = gold.train ∪ filtered synthetic. dev = small slice of
         gold.train. mini_test = held-out slice of `router-dataset.mini_test`
         with synthetic decompositions (never touched during training).
         gold_eval = the gold held-out split from step 1.

Usage:
    uv run -m training.build_decomposer_router_dataset \\
        --gold data/decomposer_router_gold.jsonl \\
        --teacher-model claude-sonnet-4-5-20250929 \\
        --teacher-provider openrouter \\
        --source task-aware-llm-council/router-dataset \\
        --out artifacts/decomposer_router_dataset \\
        --push-to-hub task-aware-llm-council/decomposer-router-dataset

Heavy deps (`datasets`, `llm_gateway` clients, `huggingface_hub`) import
inside `main()` to keep this module grep-friendly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_SOURCE_DATASET = "task-aware-llm-council/router-dataset"
DEFAULT_TARGET_DATASET = "task-aware-llm-council/decomposer-router-dataset"
DEFAULT_TEACHER_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEACHER_PROVIDER = "openrouter"
DEFAULT_SEED = 42
DEFAULT_GOLD_EVAL_FRAC = 0.2
DEFAULT_TEACHER_DISAGREEMENT_CEILING = 0.25
DEFAULT_MAX_SUBTASKS = 4
DEFAULT_CHECKPOINT_EVERY = 50

# Reused from decomposer.py — same regex so train/serve share a parser.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


# --------------------------------------------------------------------------- #
# Teacher system prompt
# --------------------------------------------------------------------------- #
#
# >>> USER CONTRIBUTION POINT <<<
#
# This prompt is the single most consequential decision in the whole data
# pipeline — it shapes thousands of synthetic training examples that will
# become the joint decomposer+router's behavioral prior. The <1k hand-gold
# set only anchors it; the teacher's output is what fills most of training.
#
# What the prompt MUST specify:
#   1. The 3-role action vocab {math_code, qa_reasoning, fact_general}
#      exactly as-is. Any deviation (e.g. the old 5-role names) will be
#      filtered out by the quality gate and waste a teacher call.
#   2. When to decompose vs. not decompose. Our router's single-label
#      assumption means multi-skill handling comes from fan-out. A prompt
#      that genuinely needs only one role should yield a single-element
#      list. Splitting single-skill prompts (e.g. "split a multi-step
#      math problem into sub-steps") is an anti-pattern — it creates
#      subtasks all routed to the same specialist, which just adds
#      overhead.
#   3. Self-containment. Each subtask must be answerable without seeing
#      the other subtasks. Pronouns, implicit context, and "then"/"next"
#      references leak information across subtasks and break the grouping
#      invariant the policy relies on.
#   4. Output format: JSON array of `{"role": str, "subtask": str}`, no
#      preamble, no markdown fences, no trailing commentary.
#
# What the prompt SHOULD also do (quality wins):
#   - Include 1–2 short in-context examples (1-role and 2-role) so the
#     teacher has a concrete template, not just rules.
#   - State the max-subtask ceiling (`DEFAULT_MAX_SUBTASKS = 4`) explicitly.
#   - Tell the teacher to preserve the order in which subtasks must be
#     answered — downstream synthesis relies on that ordering.
#
# Trade-offs to consider:
#   - Longer prompt = more teacher cost per row × thousands of rows.
#     Keep it tight but not so tight it loses specificity.
#   - Too many role descriptions = teacher over-thinks and mis-routes.
#     Too few = ambiguous boundaries ("reasoning vs math_code" for word
#     problems, "qa_reasoning vs fact_general" for claim-check questions).
#   - Examples bias the teacher — pick ones that span your real
#     distribution (MuSiQue, QuALITY, FEVER, HardMath, HumanEvalPlus).
#
# Hints if you want them:
#   - The existing (5-role) prompt at
#     `packages/council_policies/src/council_policies/decomposer.py:65-83`
#     is a decent skeleton — lift the rule list, swap the vocab.
#   - The role→dataset map is documented at `router_labels.py:15-21`.

TEACHER_SYSTEM_PROMPT = f"""\
You decompose a user prompt into the minimum number of independent subtasks \
needed to answer it fully. Each subtask maps to exactly one of three specialist roles.

Roles:
- math_code     : numerical computation, algebra, proofs, code writing or debugging
- qa_reasoning  : multi-hop questions, reading comprehension, long-document QA, \
inference chains
- fact_general  : single-claim verification, fact-checking, yes/no or \
SUPPORTS/REFUTES/NOT_ENOUGH_INFO judgements

Rules:
1. If the prompt needs only one role, return a list with exactly one element. \
Never split a single-role prompt into sub-steps (e.g. do not decompose a \
multi-step math problem — it is still one math_code task).
2. If the prompt genuinely needs two or three distinct roles, return one element \
per role, maximum {DEFAULT_MAX_SUBTASKS}.
3. Each subtask must be fully self-contained: copy in all context, numbers, \
and entities the specialist needs. No pronouns or "it"/"this" references \
that depend on another subtask. No "then"/"next" sequencing language.
4. Preserve the order subtasks must be answered in.
5. Return a JSON array only. No preamble, no markdown fences, no commentary.

Output format:
[{{"role": "<role>", "subtask": "<self-contained prompt text>"}}, ...]

Examples:

Prompt: "If f(x) = 3x² + 2x − 1, find f(4) and write a Python function \
that evaluates f for any input x."
Output:
[
  {{"role": "math_code", "subtask": "Given f(x) = 3x² + 2x − 1, compute f(4)."}},
  {{"role": "math_code", "subtask": "Write a Python function f(x) that returns \
3*x**2 + 2*x - 1 for any numeric input x."}}
]

Prompt: "The 2004 Indian Ocean tsunami was caused by a megathrust earthquake. \
Does scientific consensus support this claim?"
Output:
[
  {{"role": "fact_general", "subtask": "Claim: The 2004 Indian Ocean tsunami was \
caused by a megathrust earthquake. Is this claim SUPPORTS, REFUTES, or \
NOT_ENOUGH_INFO according to scientific consensus?"}}
]

Prompt: "Based on the following article: [article text] — Who founded the \
organisation mentioned in paragraph 3, and verify whether that person was \
born before 1900."
Output:
[
  {{"role": "qa_reasoning", "subtask": "Based on the following article: \
[article text] — Who founded the organisation mentioned in paragraph 3?"}},
  {{"role": "fact_general", "subtask": "Claim: the founder of the organisation \
mentioned in paragraph 3 of the following article was born before 1900. \
Article: [article text]. Is this SUPPORTS, REFUTES, or NOT_ENOUGH_INFO?"}}
]\
"""


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BuildArgs:
    out: Path
    gold: Path
    source: str
    source_revision: str
    teacher_model: str
    teacher_provider: str
    teacher_api_base: str | None
    teacher_api_key_env: str | None
    target_dataset: str
    push_to_hub: bool
    gold_eval_frac: float
    teacher_disagreement_ceiling: float
    max_subtasks: int
    checkpoint_every: int
    limit: int | None
    seed: int
    request_delay_seconds: float


def parse_args(argv: list[str] | None = None) -> BuildArgs:
    p = argparse.ArgumentParser(
        description="Build the joint decomposer + router training dataset."
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True,
                   help="JSONL of hand-annotated gold rows.")
    p.add_argument("--source", default=DEFAULT_SOURCE_DATASET)
    p.add_argument("--source-revision", default="",
                   help="HF dataset revision/SHA to pin. Empty = latest.")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--teacher-provider", default=DEFAULT_TEACHER_PROVIDER)
    p.add_argument("--teacher-api-base", default=None,
                   help="Override OpenAI-compatible base URL (e.g. NVIDIA NIM, "
                        "DeepSeek, Together). Required when --teacher-provider "
                        "is openai_compatible.")
    p.add_argument("--teacher-api-key-env", default=None,
                   help="Name of env var holding the teacher API key "
                        "(e.g. NVIDIA_API_KEY). Required when --teacher-provider "
                        "is openai_compatible.")
    p.add_argument("--target-dataset", default=DEFAULT_TARGET_DATASET)
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push the built dataset to --target-dataset on HF Hub.")
    p.add_argument("--gold-eval-frac", type=float, default=DEFAULT_GOLD_EVAL_FRAC)
    p.add_argument("--teacher-disagreement-ceiling", type=float,
                   default=DEFAULT_TEACHER_DISAGREEMENT_CEILING,
                   help="Halt if teacher-on-gold disagreement exceeds this.")
    p.add_argument("--max-subtasks", type=int, default=DEFAULT_MAX_SUBTASKS)
    p.add_argument("--checkpoint-every", type=int,
                   default=DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap source prompts for dry runs; None = use all.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--request-delay-seconds", type=float, default=0.0,
                   help="Sleep between teacher calls to avoid 429s. "
                        "Stacks with gateway retry backoff; set ~0.5-1.5s "
                        "for NVIDIA NIM on the shared free tier.")
    ns = p.parse_args(argv)
    return BuildArgs(
        out=ns.out,
        gold=ns.gold,
        source=ns.source,
        source_revision=ns.source_revision,
        teacher_model=ns.teacher_model,
        teacher_provider=ns.teacher_provider,
        teacher_api_base=ns.teacher_api_base,
        teacher_api_key_env=ns.teacher_api_key_env,
        target_dataset=ns.target_dataset,
        push_to_hub=ns.push_to_hub,
        gold_eval_frac=ns.gold_eval_frac,
        teacher_disagreement_ceiling=ns.teacher_disagreement_ceiling,
        max_subtasks=ns.max_subtasks,
        checkpoint_every=ns.checkpoint_every,
        limit=ns.limit,
        seed=ns.seed,
        request_delay_seconds=ns.request_delay_seconds,
    )


# --------------------------------------------------------------------------- #
# Gold IO + split
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GoldRow:
    """Validated hand-annotated row. Raises on malformed input so bad
    gold doesn't quietly pollute training."""
    question: str
    context: str
    skill_tags: list[str]
    targets: list[dict[str, str]]  # [{"role": str, "subtask": str}, ...]
    source_id: str | None = None


def load_gold_jsonl(path: Path, *, role_labels: tuple[str, ...]) -> list[GoldRow]:
    """Load + validate gold rows. Raises with row index on the first
    malformed record — we want loud failure on bad gold data, not silent
    skipping."""
    if not path.is_file():
        raise FileNotFoundError(f"gold file not found: {path}")

    rows: list[GoldRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"gold row {index}: JSON decode error: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(f"gold row {index}: not an object")
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"gold row {index}: missing/empty 'question'")
            context = obj.get("context", "") or ""
            if not isinstance(context, str):
                raise ValueError(f"gold row {index}: 'context' not str")
            tags = obj.get("skill_tags", []) or []
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                raise ValueError(f"gold row {index}: 'skill_tags' not list[str]")
            targets = obj.get("targets")
            if not isinstance(targets, list) or not targets:
                raise ValueError(
                    f"gold row {index}: 'targets' must be a non-empty list"
                )
            validated_targets: list[dict[str, str]] = []
            for t_idx, target in enumerate(targets):
                if not isinstance(target, dict):
                    raise ValueError(
                        f"gold row {index} target {t_idx}: not an object"
                    )
                role = target.get("role")
                subtask = target.get("subtask")
                if role not in role_labels:
                    raise ValueError(
                        f"gold row {index} target {t_idx}: role {role!r} "
                        f"not in {role_labels}"
                    )
                if not isinstance(subtask, str) or not subtask.strip():
                    raise ValueError(
                        f"gold row {index} target {t_idx}: missing/empty 'subtask'"
                    )
                validated_targets.append({"role": role, "subtask": subtask.strip()})
            rows.append(
                GoldRow(
                    question=question.strip(),
                    context=context,
                    skill_tags=list(tags),
                    targets=validated_targets,
                    source_id=obj.get("source_id"),
                )
            )
    if not rows:
        raise ValueError(f"gold file {path} contained no rows")
    return rows


def split_gold(
    rows: list[GoldRow], *, eval_frac: float, seed: int,
) -> tuple[list[GoldRow], list[GoldRow]]:
    """Deterministic (train, eval) split on gold rows."""
    if not 0.0 < eval_frac < 1.0:
        raise ValueError(f"eval_frac must be in (0, 1), got {eval_frac}")
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    eval_size = max(1, int(round(len(shuffled) * eval_frac)))
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    if not train_rows:
        raise ValueError(
            f"eval_frac={eval_frac} leaves no rows for gold.train "
            f"({len(rows)} total)"
        )
    return train_rows, eval_rows


# --------------------------------------------------------------------------- #
# Teacher call + checkpoint
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SyntheticRow:
    """A row produced by the teacher. `raw_response` kept for audit;
    `parsed_targets` populated only after the filter gates pass."""
    question: str
    context: str
    skill_tags: list[str]
    raw_response: str
    parsed_targets: list[dict[str, str]] | None = None
    filter_drop_reason: str | None = None
    source_id: str | None = None


def _build_teacher_user_prompt(
    *, question: str, context: str, max_subtasks: int,
) -> str:
    """Mirror of LLMDecomposer._build_user_prompt — same train-time
    framing so teacher inputs look like serve-time decomposer inputs."""
    parts: list[str] = []
    if context.strip():
        parts.append(f"Context:\n{context.strip()}")
    parts.append(f"Prompt:\n{question.strip()}")
    parts.append(
        f"Decompose into AT MOST {max_subtasks} subtasks. "
        "Return the JSON array only."
    )
    return "\n\n".join(parts)


async def call_teacher_one(
    client: Any,  # llm_gateway.BaseLLMClient
    *,
    question: str,
    context: str,
    system_prompt: str,
    max_subtasks: int,
    model: str,
) -> str:
    """One teacher call. Returns the raw response text. Retries are the
    gateway's responsibility (configured via RetryPolicy in factory)."""
    from llm_gateway import PromptRequest  # local import keeps module grep-able
    request = PromptRequest(
        model=model,
        system_prompt=system_prompt,
        user_prompt=_build_teacher_user_prompt(
            question=question, context=context, max_subtasks=max_subtasks,
        ),
        temperature=0.0,
    )
    response = await client.generate(request)
    return response.text or ""


def _checkpoint_key(row: dict[str, Any], index: int) -> str:
    """Stable id per source row. Prefer explicit `id` / `source_id`;
    fall back to index so the checkpoint still works for sources without
    an id column."""
    for key in ("id", "source_id", "qid"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return f"_idx_{index}"


def _load_checkpoint(path: Path) -> dict[str, SyntheticRow]:
    """Load already-completed teacher calls from a JSONL checkpoint."""
    if not path.is_file():
        return {}
    cached: dict[str, SyntheticRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("checkpoint: skipping malformed line")
                continue
            key = obj.get("_key")
            if not isinstance(key, str):
                continue
            cached[key] = SyntheticRow(
                question=obj["question"],
                context=obj.get("context", "") or "",
                skill_tags=list(obj.get("skill_tags") or []),
                raw_response=obj.get("raw_response", "") or "",
                source_id=obj.get("source_id"),
            )
    logger.info("checkpoint: resuming with %d cached rows from %s", len(cached), path)
    return cached


def _flush_checkpoint(
    path: Path, *, batch: list[tuple[str, SyntheticRow]],
) -> None:
    """Append new rows to the checkpoint. Atomic-ish: one open → write all
    → close, so a crash mid-loop won't leave a half-written line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for key, row in batch:
            handle.write(json.dumps({
                "_key": key,
                "question": row.question,
                "context": row.context,
                "skill_tags": row.skill_tags,
                "raw_response": row.raw_response,
                "source_id": row.source_id,
            }, ensure_ascii=False) + "\n")


async def call_teacher_batch(
    client: Any,
    *,
    rows: list[dict[str, Any]],   # raw source-dataset rows
    system_prompt: str,
    max_subtasks: int,
    model: str,
    checkpoint_path: Path,
    checkpoint_every: int,
    request_delay_seconds: float = 0.0,
) -> list[SyntheticRow]:
    """Drive teacher calls sequentially with a JSONL checkpoint so a
    network blip doesn't wipe the run. Returns rows in source order."""
    cached = _load_checkpoint(checkpoint_path)
    results: list[SyntheticRow] = []
    pending: list[tuple[str, SyntheticRow]] = []

    for index, row in enumerate(rows):
        key = _checkpoint_key(row, index)
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "")
        skill_tags = list(row.get("skill_tags") or [])
        source_id = row.get("source_id") or row.get("id")

        if key in cached:
            results.append(cached[key])
            continue

        if not question:
            logger.warning("row %d (%s): empty question, skipping", index, key)
            continue

        try:
            raw = await call_teacher_one(
                client,
                question=question,
                context=context,
                system_prompt=system_prompt,
                max_subtasks=max_subtasks,
                model=model,
            )
        except Exception as exc:
            logger.warning(
                "row %d (%s): teacher call failed (%s); recording empty response",
                index, key, type(exc).__name__,
            )
            raw = ""

        synthetic = SyntheticRow(
            question=question,
            context=context,
            skill_tags=skill_tags,
            raw_response=raw,
            source_id=source_id if isinstance(source_id, str) else None,
        )
        results.append(synthetic)
        pending.append((key, synthetic))

        if len(pending) >= checkpoint_every:
            _flush_checkpoint(checkpoint_path, batch=pending)
            pending.clear()

        if request_delay_seconds > 0 and index < len(rows) - 1:
            await asyncio.sleep(request_delay_seconds)

    if pending:
        _flush_checkpoint(checkpoint_path, batch=pending)
    return results


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def parse_teacher_response(
    raw: str, *, max_subtasks: int,
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Extract the JSON array. Returns (parsed, None) on success, or
    (None, reason) for filter telemetry. Mirrors the ladder in
    `decomposer._parse_response` but returns a reason string instead of
    logging, so we can tally drop reasons in the data card."""
    if not raw:
        return None, "empty_response"
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return None, "no_json_array"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "json_decode_error"
    if not isinstance(parsed, list):
        return None, "not_a_list"
    cleaned: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        subtask = item.get("subtask")
        if not isinstance(role, str) or not role.strip():
            continue
        if not isinstance(subtask, str) or not subtask.strip():
            continue
        cleaned.append({"role": role.strip(), "subtask": subtask.strip()})
        if len(cleaned) >= max_subtasks:
            break
    if not cleaned:
        return None, "no_usable_items"
    return cleaned, None


def passes_role_vocab_gate(
    targets: list[dict[str, str]], *, role_labels: tuple[str, ...],
) -> bool:
    """Every emitted role must be in the canonical 3-role vocab."""
    return all(t.get("role") in role_labels for t in targets)


def passes_oracle_consistency_gate(
    targets: list[dict[str, str]], *, oracle_role: str,
) -> bool:
    """Emitted roles must include the source row's oracle role
    (`role_from_tags(skill_tags)`). Guards against the teacher ignoring
    the skill signal and hallucinating a different route."""
    return any(t.get("role") == oracle_role for t in targets)


def apply_filters(
    rows: list[SyntheticRow],
    *,
    role_labels: tuple[str, ...],
    max_subtasks: int,
) -> tuple[list[SyntheticRow], dict[str, int]]:
    """Run all three gates. Returns (kept, drop_counts_by_reason).

    The oracle-consistency gate requires `role_from_tags` — imported
    here (not at module top) to keep the module import-light for
    non-ML tooling.
    """
    from council_policies.router_labels import role_from_tags

    kept: list[SyntheticRow] = []
    drops: dict[str, int] = {}
    for row in rows:
        parsed, reason = parse_teacher_response(
            row.raw_response, max_subtasks=max_subtasks,
        )
        if parsed is None:
            row.filter_drop_reason = reason
            drops[reason or "unknown"] = drops.get(reason or "unknown", 0) + 1
            continue
        if not passes_role_vocab_gate(parsed, role_labels=role_labels):
            row.filter_drop_reason = "bad_role_vocab"
            drops["bad_role_vocab"] = drops.get("bad_role_vocab", 0) + 1
            continue
        oracle_role = role_from_tags(row.skill_tags)
        if not passes_oracle_consistency_gate(parsed, oracle_role=oracle_role):
            row.filter_drop_reason = "oracle_disagreement"
            drops["oracle_disagreement"] = drops.get("oracle_disagreement", 0) + 1
            continue
        row.parsed_targets = parsed
        kept.append(row)
    return kept, drops


# --------------------------------------------------------------------------- #
# Teacher-on-gold sanity gate
# --------------------------------------------------------------------------- #


async def teacher_agreement_on_gold(
    client: Any,
    gold_train: list[GoldRow],
    *,
    system_prompt: str,
    max_subtasks: int,
    model: str,
) -> float:
    """Run the teacher on gold.train prompts and compare emitted roles
    to gold targets. Returns disagreement rate in [0, 1]. Higher is
    worse. If > ceiling, the caller halts before publishing.

    Metric: row-level disagreement on the **set of emitted roles**.
    Shape-match is NOT required — we care about routing fidelity, not
    exact subtask count match (teacher may phrase things differently).
    """
    if not gold_train:
        raise ValueError("teacher_agreement_on_gold: gold_train is empty")

    disagreements = 0
    for row in gold_train:
        try:
            raw = await call_teacher_one(
                client,
                question=row.question,
                context=row.context,
                system_prompt=system_prompt,
                max_subtasks=max_subtasks,
                model=model,
            )
        except Exception as exc:
            logger.warning(
                "teacher_agreement_on_gold: call failed (%s); counting as disagreement",
                type(exc).__name__,
            )
            disagreements += 1
            continue
        parsed, _ = parse_teacher_response(raw, max_subtasks=max_subtasks)
        if parsed is None:
            disagreements += 1
            continue
        emitted_roles = {t["role"] for t in parsed}
        gold_roles = {t["role"] for t in row.targets}
        if emitted_roles != gold_roles:
            disagreements += 1
    return disagreements / len(gold_train)


# --------------------------------------------------------------------------- #
# Publish
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DataCard:
    """Records provenance for the published dataset. Written as
    `data_card.json` in the output dir and, if pushed, in the HF dataset
    card."""
    target_dataset: str
    source_dataset: str
    source_revision: str
    teacher_model: str
    teacher_provider: str
    role_labels: tuple[str, ...]
    splits: dict[str, int]
    filter_drop_counts: dict[str, int]
    teacher_gold_disagreement: float
    created_at: str
    git_sha: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dataset": self.target_dataset,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "teacher_model": self.teacher_model,
            "teacher_provider": self.teacher_provider,
            "role_labels": list(self.role_labels),
            "splits": dict(self.splits),
            "filter_drop_counts": dict(self.filter_drop_counts),
            "teacher_gold_disagreement": self.teacher_gold_disagreement,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "seed": self.seed,
        }


def publish(
    *,
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    mini_test_rows: list[dict[str, Any]],
    gold_eval_rows: list[dict[str, Any]],
    card: DataCard,
    push_to_hub: bool,
    target_dataset: str,
) -> None:
    """Write parquet splits + card.json locally. Optional HF push.

    Schema per row (uniform across splits):
        {question, context, skill_tags, targets, source_id, split_origin}

    `split_origin` tracks which upstream pool a row came from
    (gold.train / gold.eval / synthetic / mini_test) so downstream
    analysis can slice quality metrics by source.
    """
    from datasets import Dataset, DatasetDict

    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": Dataset.from_list(train_rows),
        "dev": Dataset.from_list(dev_rows),
        "mini_test": Dataset.from_list(mini_test_rows),
        "gold_eval": Dataset.from_list(gold_eval_rows),
    }
    dataset = DatasetDict(splits)

    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ds in dataset.items():
        ds.to_parquet(str(parquet_dir / f"{split_name}.parquet"))

    card_path = out_dir / "data_card.json"
    card_path.write_text(
        json.dumps(card.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("wrote dataset to %s (splits: %s)", out_dir, list(splits.keys()))

    if push_to_hub:
        from huggingface_hub import login
        login()
        dataset.push_to_hub(target_dataset)
        logger.info("pushed dataset to %s", target_dataset)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _seed_everything(seed: int) -> None:
    random.seed(seed)


def _assert_teacher_prompt_ready() -> None:
    """Loud guard so nobody runs the pipeline with the placeholder prompt."""
    if "<REPLACE ME" in TEACHER_SYSTEM_PROMPT:
        raise RuntimeError(
            "TEACHER_SYSTEM_PROMPT is still the placeholder. "
            "Edit build_decomposer_router_dataset.py and replace it "
            "before running the pipeline."
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _gold_row_to_record(row: GoldRow, split_origin: str) -> dict[str, Any]:
    return {
        "question": row.question,
        "context": row.context,
        "skill_tags": list(row.skill_tags),
        "targets": list(row.targets),
        "source_id": row.source_id,
        "split_origin": split_origin,
    }


def _synthetic_row_to_record(row: SyntheticRow, split_origin: str) -> dict[str, Any]:
    return {
        "question": row.question,
        "context": row.context,
        "skill_tags": list(row.skill_tags),
        "targets": list(row.parsed_targets or []),
        "source_id": row.source_id,
        "split_origin": split_origin,
    }


def _build_teacher_client(args: BuildArgs) -> Any:
    """Construct the teacher `BaseLLMClient` via `llm_gateway.create_client`.

    Provider-specific config (api_base, api_key_env) uses factory
    defaults. The caller sets credentials via environment variables
    (e.g. `OPENROUTER_API_KEY`) — we don't read them here.
    """
    from llm_gateway.factory import create_client
    from llm_gateway.models import ProviderConfig, RetryPolicy
    if args.teacher_provider == "openai-compatible" and not (
        args.teacher_api_base and args.teacher_api_key_env
    ):
        raise ValueError(
            "--teacher-provider openai-compatible requires "
            "--teacher-api-base and --teacher-api-key-env."
        )
    config = ProviderConfig(
        provider=args.teacher_provider,
        default_model=args.teacher_model,
        api_base=args.teacher_api_base,
        api_key_env=args.teacher_api_key_env,
    )
    return create_client(config, retry_policy=RetryPolicy(max_retries=5))


async def _run_pipeline(args: BuildArgs) -> int:
    from council_policies.router_labels import ROLE_LABELS

    logger.info("loading gold from %s", args.gold)
    gold_all = load_gold_jsonl(args.gold, role_labels=ROLE_LABELS)
    gold_train, gold_eval = split_gold(
        gold_all, eval_frac=args.gold_eval_frac, seed=args.seed,
    )
    logger.info(
        "gold: %d total → %d train / %d eval",
        len(gold_all), len(gold_train), len(gold_eval),
    )

    client = _build_teacher_client(args)

    # Gate 0: teacher must agree with gold on at least (1 - ceiling) of rows.
    logger.info("running teacher-on-gold agreement gate (%d rows)", len(gold_train))
    disagreement = await teacher_agreement_on_gold(
        client,
        gold_train,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        max_subtasks=args.max_subtasks,
        model=args.teacher_model,
    )
    logger.info("teacher-on-gold disagreement: %.3f", disagreement)
    if disagreement > args.teacher_disagreement_ceiling:
        raise RuntimeError(
            f"teacher disagrees with gold on {disagreement:.1%} of rows "
            f"(> ceiling {args.teacher_disagreement_ceiling:.1%}). "
            f"Revise TEACHER_SYSTEM_PROMPT before publishing."
        )

    # Load source pool (router-dataset) for synthetic augmentation.
    from datasets import load_dataset
    logger.info("loading source dataset %s", args.source)
    source_kwargs: dict[str, Any] = {}
    if args.source_revision:
        source_kwargs["revision"] = args.source_revision
    source = load_dataset(args.source, **source_kwargs)

    dev_rows_raw = list(source["validation"])
    mini_test_rows_raw = list(source["test"])
    if args.limit is not None:
        dev_rows_raw = dev_rows_raw[: args.limit]
        mini_test_rows_raw = mini_test_rows_raw[: max(1, args.limit // 4)]

    logger.info(
        "source: %d dev rows, %d mini_test rows (after --limit=%s)",
        len(dev_rows_raw), len(mini_test_rows_raw), args.limit,
    )

    # Teacher pass 1: dev → synthetic training augmentation.
    synthetic_raw = await call_teacher_batch(
        client,
        rows=dev_rows_raw,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        max_subtasks=args.max_subtasks,
        model=args.teacher_model,
        checkpoint_path=args.out / "_teacher_cache" / "dev.jsonl",
        checkpoint_every=args.checkpoint_every,
        request_delay_seconds=args.request_delay_seconds,
    )
    synthetic_kept, drops = apply_filters(
        synthetic_raw,
        role_labels=ROLE_LABELS,
        max_subtasks=args.max_subtasks,
    )
    retention = len(synthetic_kept) / max(1, len(synthetic_raw))
    logger.info(
        "synthetic filter: kept %d/%d (%.1f%%); drops=%s",
        len(synthetic_kept), len(synthetic_raw), 100 * retention, drops,
    )

    # Teacher pass 2: mini_test → held-out eval set with synthetic labels.
    mini_test_raw = await call_teacher_batch(
        client,
        rows=mini_test_rows_raw,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        max_subtasks=args.max_subtasks,
        model=args.teacher_model,
        checkpoint_path=args.out / "_teacher_cache" / "mini_test.jsonl",
        checkpoint_every=args.checkpoint_every,
        request_delay_seconds=args.request_delay_seconds,
    )
    mini_test_kept, mini_drops = apply_filters(
        mini_test_raw,
        role_labels=ROLE_LABELS,
        max_subtasks=args.max_subtasks,
    )
    for reason, count in mini_drops.items():
        drops[f"mini_test:{reason}"] = count
    logger.info(
        "mini_test filter: kept %d/%d", len(mini_test_kept), len(mini_test_raw),
    )

    # Split gold.train into (train-gold, dev-gold) so dev is NEVER synthetic.
    dev_gold_size = max(1, int(round(len(gold_train) * 0.1)))
    dev_gold = gold_train[:dev_gold_size]
    train_gold = gold_train[dev_gold_size:]

    train_records = (
        [_gold_row_to_record(r, "gold_train") for r in train_gold]
        + [_synthetic_row_to_record(r, "synthetic") for r in synthetic_kept]
    )
    dev_records = [_gold_row_to_record(r, "gold_dev") for r in dev_gold]
    mini_test_records = [
        _synthetic_row_to_record(r, "mini_test_synthetic") for r in mini_test_kept
    ]
    gold_eval_records = [_gold_row_to_record(r, "gold_eval") for r in gold_eval]

    card = DataCard(
        target_dataset=args.target_dataset,
        source_dataset=args.source,
        source_revision=args.source_revision,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        role_labels=ROLE_LABELS,
        splits={
            "train": len(train_records),
            "dev": len(dev_records),
            "mini_test": len(mini_test_records),
            "gold_eval": len(gold_eval_records),
        },
        filter_drop_counts=drops,
        teacher_gold_disagreement=disagreement,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        seed=args.seed,
    )

    publish(
        out_dir=args.out,
        train_rows=train_records,
        dev_rows=dev_records,
        mini_test_rows=mini_test_records,
        gold_eval_rows=gold_eval_records,
        card=card,
        push_to_hub=args.push_to_hub,
        target_dataset=args.target_dataset,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _assert_teacher_prompt_ready()
    _seed_everything(args.seed)

    import asyncio
    return asyncio.run(_run_pipeline(args))


if __name__ == "__main__":
    raise SystemExit(main())
