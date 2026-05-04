"""
P3 Policy evaluation CLI — profile-driven variant (cli_handles.py).

Fork of cli.py that stops reading from the pre-flattened
`task-aware-llm-council/router_dataset` and instead builds each case via the
matching DatasetProfile in `task_eval.profiles`. This restores two things
the router_dataset throws away:

  • Oracle Context (musique) — MusiqueProfile.row_to_case loads the raw
    `bdsaglam/musique` rows and keeps ONLY paragraphs flagged
    is_supporting=True (the 2–4 paragraphs needed for the hops), instead
    of the ~20-paragraph distractor soup stored in router_dataset.context.

  • Profile-native constrained prompts — scratchpad CoT for musique, the
    Claim/SUPPORTS-REFUTES-NOT_ENOUGH_INFO shape for fever, \\boxed{} for
    hardmath, etc. router_dataset only carries the bare question.

Scoring is delegated back to each profile's `.score()` method, so each
dataset uses its own extractor (extract_qa_answer_musique,
extract_fever_label, extract_math_answer, extract_code_answer,
extract_qa_answer for quality). The output jsonl keeps the same record
shape as cli.py — the `metrics` dict still exposes the unified
{token_f1, exact_match, accuracy, pass_at_1} keys so downstream tools
keep working, and `metric_metadata` carries the profile's native metric
values verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarking_pipeline import BenchmarkExample
from model_orchestration import ModelOrchestrator, build_default_local_vllm_orchestrator_config

from task_eval.interfaces import DatasetProfile
from task_eval.models import EvaluationCase, MetricResult
from task_eval.profiles import (
    FeverProfile,
    HardMathProfile,
    HumanEvalPlusProfile,
    MusiqueProfile,
    QualityProfile,
)

from council_policies.p3_policy import RuleBasedRoutingPolicy


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_FILE = "results_p3_eval_local.jsonl"
N_PER_DATASET = 16            # matches the row count in the current results file
MAX_CONCURRENT_QUESTIONS = 4

# Profile.name → the short name used by p3_policy._DATASET_TO_TASK for routing
# and by downstream aggregators that bucket by cli.py's normalised names.
# Only humaneval_plus needs aliasing; the others already match.
_PROFILE_TO_ROUTING_NAME: dict[str, str] = {
    "humaneval_plus": "humaneval",
}

_VALID_TASK_TYPES = {"qa", "math", "code", "fever"}


# ── Per-dataset reinforcement system prompts ──────────────────────────────────
#
# The profile-built `question` already contains format constraints (scratchpad
# for musique, "Final Answer: <LABEL>" for fever, \\boxed{} for hardmath). We
# only *prepend* a system prompt where the profile's template is thin:
#
#   - HumanEvalPlusProfile passes the raw HumanEval prompt with no fencing
#     instruction, which costs pass@1 when the model wraps prose around code.
#   - QualityProfile only appends a one-line "answer concisely" suffix, so
#     we reinforce with the full QA prompt.
#
# For musique / fever / hardmath we deliberately do NOT stack an extra prompt
# on top of the profile's own format rules — the profiles are the source of
# truth for those datasets.

_SYS_QUALITY = (
    "You answer reading-comprehension questions using ONLY the provided context.\n"
    "Respond with the shortest possible span that answers the question — typically "
    "a few words or a single phrase copied from the context. Do not add commentary, "
    "restate the question, or explain your reasoning.\n"
    "End your response with a line in this exact format:\n"
    "Final Answer: <answer>"
)

_SYS_CODE = (
    "You are a Python code generator. Given a function signature and docstring, "
    "return ONLY the completed function inside a single fenced code block:\n"
    "```python\n"
    "def func_name(...):\n"
    "    ...\n"
    "```\n"
    "Do NOT include prose, test cases, usage examples, or explanations. "
    "The fenced block must be valid, self-contained Python that defines the "
    "function exactly as named in the prompt."
)

_REINFORCEMENT_PROMPTS: dict[str, str] = {
    "quality":        _SYS_QUALITY,
    "humaneval_plus": _SYS_CODE,
    "humaneval":      _SYS_CODE,
}


# ── Orchestrator setup ────────────────────────────────────────────────────────

vllm_specialist_config = build_default_local_vllm_orchestrator_config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Profile instantiation ─────────────────────────────────────────────────────
#
# We do NOT use p3_policy.load_all_profiles() here because it inherits the
# BaseDatasetProfile.split="validation" default for every profile, but the
# MATH-500 and openai_humaneval datasets only publish a "test" split on HF.
# Overriding per profile here avoids touching shared code.

def _profiles_with_correct_splits() -> list[DatasetProfile]:
    return [
        MusiqueProfile(),
        QualityProfile(),
        FeverProfile(),
        HardMathProfile(split="test"),
        HumanEvalPlusProfile(split="test"),
    ]


# ── Case preparation ──────────────────────────────────────────────────────────

def _reinforce_question(dataset_name: str, question: str) -> str:
    sys_prompt = _REINFORCEMENT_PROMPTS.get(dataset_name)
    if not sys_prompt:
        return question
    return f"{sys_prompt}\n\n{question}"


def _rewrap_case(case: EvaluationCase, routing_name: str) -> EvaluationCase:
    """Reclone an EvaluationCase with a routing-friendly dataset_name and an
    optionally reinforced question. BenchmarkExample and EvaluationCase are
    both frozen dataclasses, so we rebuild rather than mutate.
    """
    example = case.example
    new_question = _reinforce_question(routing_name, example.question or "")
    new_example = BenchmarkExample(
        example_id=example.example_id,
        dataset_name=routing_name,
        question=new_question,
        context=example.context,
        system_prompt=example.system_prompt,
        messages=example.messages,
        metadata=dict(example.metadata),
    )
    return EvaluationCase(
        example=new_example,
        reference=dict(case.reference),
        metadata=dict(case.metadata),
    )


def build_cases(
    profiles: list[DatasetProfile],
    n_per_dataset: int,
) -> tuple[list[EvaluationCase], dict[tuple[str, str], DatasetProfile]]:
    """Iterate each profile, sample up to n_per_dataset cases, and return:
      - the list of cases (with rewrapped dataset_name for routing)
      - a (example_id, dataset_name) → profile lookup for scoring later
    """
    all_cases: list[EvaluationCase] = []
    case_profile: dict[tuple[str, str], DatasetProfile] = {}

    for profile in profiles:
        routing_name = _PROFILE_TO_ROUTING_NAME.get(profile.name, profile.name)
        sampled = 0
        try:
            for raw_case in profile.iter_cases():
                case = _rewrap_case(raw_case, routing_name)
                all_cases.append(case)
                case_profile[(case.example.example_id, routing_name)] = profile
                sampled += 1
                if sampled >= n_per_dataset:
                    break
        except Exception as exc:
            logger.warning("Skipping dataset %r — iter_cases failed: %s", profile.name, exc)
            continue
        logger.info("Sampled %d / %d from %s", sampled, n_per_dataset, profile.name)

    return all_cases, case_profile


# ── Metric adaptation ─────────────────────────────────────────────────────────
#
# Profile.score returns a MetricResult whose keys are dataset-specific. cli.py
# downstream tools expect a uniform four-key shape. _unify_metrics translates
# the profile's keys into that shape; _PROFILE_METRIC_KEY_MAP below covers the
# one-to-one translations.

_EMPTY_METRICS: dict[str, Any] = {
    "token_f1":    None,
    "exact_match": None,
    "accuracy":    None,
    "pass_at_1":   None,
}


def _unify_metrics(routing_name: str, metric_values: dict[str, Any]) -> dict[str, Any]:
    """Project a profile's native metric dict onto cli.py's unified schema."""
    unified = dict(_EMPTY_METRICS)

    if routing_name in ("musique", "quality"):
        if "token_f1" in metric_values:
            unified["token_f1"] = metric_values["token_f1"]
        if "exact_match" in metric_values:
            unified["exact_match"] = metric_values["exact_match"]

    elif routing_name == "fever":
        if "label_accuracy" in metric_values:
            unified["accuracy"] = metric_values["label_accuracy"]

    elif routing_name == "hardmath":
        if "math_exact_match" in metric_values:
            unified["accuracy"] = metric_values["math_exact_match"]

    elif routing_name == "humaneval":
        if "pass_at_1" in metric_values:
            unified["pass_at_1"] = metric_values["pass_at_1"]

    return unified


def _score_case(
    profile: DatasetProfile,
    case: EvaluationCase,
    response_text: str | None,
    failed: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run profile.score and return (unified_metrics, native_metric_values, metadata)."""
    routing_name = case.example.dataset_name

    if failed or not response_text:
        return dict(_EMPTY_METRICS), {}, {"extracted_answer": ""}

    try:
        result: MetricResult = profile.score(
            case=case,
            prediction={"response_text": response_text},
        )
    except Exception as exc:
        logger.warning("Scoring failed for %s/%s: %s", routing_name, case.example.example_id, exc)
        return dict(_EMPTY_METRICS), {}, {"extracted_answer": "", "score_error": str(exc)}

    unified = _unify_metrics(routing_name, result.values)
    return unified, dict(result.values), dict(result.metadata)


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def async_main() -> None:
    suite_id = f"suite_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"Suite: {suite_id}")

    print("Loading profiles and building cases from raw HF datasets...")
    profiles = _profiles_with_correct_splits()
    cases, case_profile = build_cases(profiles, n_per_dataset=N_PER_DATASET)
    print(f"Prepared {len(cases)} cases across {len(profiles)} profiles.")

    if not cases:
        print("No cases built — aborting.")
        return

    async with ModelOrchestrator(vllm_specialist_config) as orchestrator:
        policy = RuleBasedRoutingPolicy(
            orchestrator,
            n_per_dataset=N_PER_DATASET,
            max_concurrent_questions=MAX_CONCURRENT_QUESTIONS,
        )
        print("Running P3 routing policy...")
        result = await policy.run(cases)

    print("Building metric-ready output records...")
    output_records: list[dict[str, Any]] = []

    for qr in result.results:
        case = qr.case
        example = case.example
        resp = qr.response
        routing_name = example.dataset_name

        task_type_str = qr.task_type.value
        if task_type_str not in _VALID_TASK_TYPES:
            task_type_str = "qa"

        model_id = resp.model if resp is not None else None

        usage_obj = resp.usage if resp is not None else None
        usage_dict: dict | None = None
        if usage_obj is not None and any(
            v is not None
            for v in (usage_obj.input_tokens, usage_obj.output_tokens, usage_obj.total_tokens)
        ):
            usage_dict = {
                "prompt_tokens": usage_obj.input_tokens,
                "completion_tokens": usage_obj.output_tokens,
                "total_tokens": usage_obj.total_tokens,
            }

        response_text = resp.text if resp is not None else None

        profile = case_profile.get((example.example_id, routing_name))
        if profile is None:
            unified_metrics = dict(_EMPTY_METRICS)
            native_metrics: dict[str, Any] = {}
            score_meta: dict[str, Any] = {"extracted_answer": "", "score_error": "profile_missing"}
        else:
            unified_metrics, native_metrics, score_meta = _score_case(
                profile, case, response_text, qr.failed
            )

        record: dict[str, Any] = {
            "example_id":   example.example_id,
            "dataset_name": routing_name,
            "suite_id":     suite_id,
            "question":     example.question,
            "system_prompt": _REINFORCEMENT_PROMPTS.get(routing_name),
            "context":      example.context,
            "response_text": response_text,
            "model":         model_id,
            "latency_ms":    qr.latency_ms,
            "failed":        qr.failed,
            "error":         qr.error,
            "task_type":     task_type_str,
            "routed_role":   qr.routed_role,
            "routed_models": [model_id] if model_id else [],
            "reference":     case.reference,
        }

        if usage_dict is not None:
            record["usage"] = usage_dict

        answerable = case.metadata.get("answerable")
        if answerable is not None:
            record["answerable"] = answerable

        record["metrics"] = unified_metrics
        metric_metadata = dict(score_meta)
        if native_metrics:
            metric_metadata["profile_metrics"] = native_metrics
        record["metric_metadata"] = metric_metadata

        output_records.append(record)

    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")
    print(f"\nSaved {len(output_records)} records to {output_path.resolve()}")

    print("\n=== Routing Summary ===")
    for summary in result.routing_summaries:
        print(f"\nDataset : {summary.dataset_name}")
        print(f"  Questions   : {summary.question_count}")
        print(f"  Roles       : {summary.role_counts}")
        print(f"  Task types  : {summary.task_type_counts}")
        if summary.avg_latency_ms is not None:
            print(f"  Avg latency : {summary.avg_latency_ms:.0f} ms")

    skipped = len(result.skipped_question_ids)
    succeeded = sum(1 for r in result.results if not r.failed)
    print(f"\nTotal: {succeeded} succeeded, {skipped} skipped/failed.")


def main() -> None:
    print("Starting P3 policy evaluation CLI (profile-driven, cli_handles variant)...")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
