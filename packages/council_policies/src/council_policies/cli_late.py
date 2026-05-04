"""
P3 Policy evaluation CLI — strict local dataset variant.

Fork of cli.py that enforces the "single provided dataset" contract:

  * NO call to `datasets.load_dataset(...)` with a remote repo id.
  * NO implicit Hugging Face Hub resolution.
  * The evaluation dataset MUST be passed in via `--dataset-path` and is read
    from the local filesystem (JSONL, JSON array, or a `load_from_disk`
    directory). If the path is missing or empty, the CLI fails fast.
  * `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `HF_HUB_OFFLINE` are
    set at import time so that any stray HF call elsewhere in the process
    raises instead of silently reaching the network.

The scoring / routing / orchestrator wiring is unchanged from cli.py —
only the dataset source and its entry-point plumbing are different.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ── Offline guard ─────────────────────────────────────────────────────────────
# Must run BEFORE importing anything that transitively imports `datasets`
# or `transformers`, otherwise those libs latch onto the online defaults.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from llm_gateway import LOCAL_LAUNCH_QUANTIZATION  # noqa: F401
from model_orchestration import ModelOrchestrator, build_default_local_vllm_orchestrator_config
from model_orchestration.models import LocalVLLMPresetConfig  # noqa: F401

from task_eval.models import EvaluationCase
from benchmarking_pipeline import BenchmarkExample
from council_policies.p3_policy import RuleBasedRoutingPolicy

from common import get_current_user  # noqa: F401

try:
    from task_eval.scoring import pass_at_1 as _sandboxed_pass_at_1
except Exception:
    _sandboxed_pass_at_1 = None


# ── Dataset name normalisation ─────────────────────────────────────────────────
_DATASET_NORM: dict[str, str] = {
    "musique": "musique",
    "quality": "quality",
    "fever": "fever",
    "hardmath": "hardmath",
    "humaneval_plus": "humaneval",
    "humaneval+": "humaneval",
    "humanevalplus": "humaneval",
    "humaneval": "humaneval",
}

_VALID_TASK_TYPES = {"qa", "math", "code", "fever"}


def _normalize_dataset(source: str) -> str:
    key = source.lower().replace(" ", "_").replace("-", "_").replace("+", "")
    return _DATASET_NORM.get(key, key)


# ── Per-dataset system prompts ────────────────────────────────────────────────

_SYS_QA = (
    "You answer reading-comprehension questions using ONLY the provided context.\n"
    "Respond with the shortest possible span that answers the question — typically "
    "a few words or a single phrase copied from the context. Do not add commentary, "
    "restate the question, or explain your reasoning.\n"
    "If the context does not contain the answer, respond exactly: NOT_FOUND\n"
    "End your response with a line in this exact format:\n"
    "Final Answer: <answer>"
)

_SYS_FEVER = (
    "You are a fact-verification classifier. Given a claim, output exactly ONE "
    "of these three labels and nothing else:\n"
    "  SUPPORTS\n"
    "  REFUTES\n"
    "  NOT ENOUGH INFO\n"
    "Do not add explanation, punctuation, or any other text. End with:\n"
    "Final Answer: <LABEL>"
)

_SYS_MATH = (
    "You are a math problem solver. Reason step by step, then put ONLY the final "
    "answer inside \\boxed{}. Do not include units or words inside the box.\n"
    "Your response MUST end with \\boxed{<answer>} on the last line."
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

_SYSTEM_PROMPTS: dict[str, str] = {
    "musique":   _SYS_QA,
    "quality":   _SYS_QA,
    "fever":     _SYS_FEVER,
    "hardmath":  _SYS_MATH,
    "humaneval": _SYS_CODE,
}


def _build_system_prompt(dataset_name: str) -> str | None:
    return _SYSTEM_PROMPTS.get(dataset_name)


def _build_reference(row: dict, dataset_name: str) -> dict:
    if dataset_name == "musique":
        answer = str(row.get("gold_answer") or row.get("answer") or "")
        aliases = row.get("answer_aliases") or row.get("answers") or []
        if isinstance(aliases, list):
            answers = [answer] + [str(a) for a in aliases if str(a) != answer]
        else:
            answers = [answer]
        answers = [a for a in answers if a]
        return {"answers": answers} if answers else {}

    if dataset_name == "quality":
        gold = row.get("gold_answer") or row.get("answer") or row.get("answers")
        if isinstance(gold, list):
            return {"answers": [str(a) for a in gold if a]}
        return {"answers": [str(gold)]} if gold else {}

    if dataset_name == "fever":
        label = str(row.get("gold_label") or row.get("label") or "")
        return {"answers": [label]} if label else {}

    if dataset_name == "hardmath":
        answer = str(row.get("gold_answer") or row.get("answer") or row.get("solution") or "")
        return {"answers": [answer]} if answer else {}

    if dataset_name == "humaneval":
        solution = str(
            row.get("canonical_solution") or row.get("solution") or row.get("gold_answer") or ""
        )
        ref: dict = {"answers": [solution]} if solution else {"answers": []}
        test_code = str(row.get("test") or row.get("test_code") or "")
        entry_point = str(row.get("entry_point") or "")
        if test_code:
            ref["test_code"] = test_code
        if entry_point:
            ref["entry_point"] = entry_point
        return ref

    return {}


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Local vLLM specialist config ───────────────────────────────────────────────
_QA_MODEL        = "task-aware-llm-council/gemma-2-9b-it-GPTQ"
_REASONING_MODEL = "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
_GENERAL_MODEL   = "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"

vllm_specialist_config = build_default_local_vllm_orchestrator_config()
DEFAULT_OUTPUT_FILE = "results_p3_eval_local.jsonl"


# ── Local dataset loader ──────────────────────────────────────────────────────

def _iter_rows_from_path(path: Path) -> Iterator[dict]:
    """Yield dataset rows from a local file or directory.

    Supported inputs:
      * `.jsonl` / `.ndjson` — one JSON object per line.
      * `.json` — a single JSON array of objects.
      * A directory — treated as an Arrow dataset via `datasets.load_from_disk`.
        This still uses the `datasets` library but reads exclusively from disk
        and does not resolve any remote identifier.

    Anything else raises `ValueError`. We never fall through to a remote fetch.
    """
    if not path.exists():
        raise FileNotFoundError(f"--dataset-path does not exist: {path}")

    if path.is_dir():
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError(
                "Reading a saved-to-disk dataset directory requires the "
                "`datasets` package to be installed locally."
            ) from exc
        ds = load_from_disk(str(path))
        if hasattr(ds, "keys"):
            raise ValueError(
                f"{path} is a DatasetDict with splits {list(ds.keys())!r}. "
                "Point --dataset-path at a specific split subdirectory."
            )
        for row in ds:
            yield dict(row)
        return

    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON at {path}:{line_no}: {exc}"
                    ) from exc
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(
                f"{path} must contain a JSON array of row objects; got "
                f"{type(payload).__name__}."
            )
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path} contains a non-object row: {type(row).__name__}"
                )
            yield row
        return

    raise ValueError(
        f"Unsupported dataset file extension {suffix!r} at {path}. "
        "Use .jsonl, .ndjson, .json, or a directory from dataset.save_to_disk()."
    )


# ── Scoring / metrics ──────────────────────────────────────────────────────────

def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    common = sum((pred_counter & gt_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def _best_token_f1(prediction: str, answers: list[str]) -> float:
    return max((_token_f1(prediction, a) for a in answers), default=0.0)


def _best_exact_match(prediction: str, answers: list[str]) -> float:
    return max((_exact_match(prediction, a) for a in answers), default=0.0)


def _extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else ""


def _extract_code_block(text: str) -> str:
    if not text:
        return ""
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip("\n")

    m = re.search(r"^(?:def |class |import |from )", text, re.MULTILINE)
    if m:
        return text[m.start():].strip()
    return text.strip()


def _extract_answer(prediction: str, dataset: str = "") -> str:
    if not prediction:
        return ""

    match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", prediction, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().rstrip(".").strip()
    elif dataset in ("musique", "quality", "fever"):
        lines = [ln.strip() for ln in prediction.splitlines() if ln.strip()]
        extracted = lines[-1] if lines else prediction.strip()
    else:
        extracted = prediction.strip()

    if extracted.upper().strip() == "NOT_FOUND":
        return "NOT_FOUND"
    return extracted


_FEVER_LABEL_PATTERNS: dict[str, re.Pattern] = {
    "SUPPORTS":         re.compile(r"\bSUPPORT(?:S|ED|ING)?\b", re.IGNORECASE),
    "REFUTES":          re.compile(r"\bREFUTE(?:S|D|ING)?\b", re.IGNORECASE),
    "NOT ENOUGH INFO":  re.compile(r"NOT[\s_]ENOUGH[\s_]INFO", re.IGNORECASE),
}


def _fever_label_accuracy(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    gold = answers[0].strip().upper()
    gold_key = "NOT ENOUGH INFO" if gold.startswith("NOT") else gold
    pattern = _FEVER_LABEL_PATTERNS.get(gold_key)
    if pattern is None:
        return 1.0 if gold and gold in prediction.upper() else 0.0
    return 1.0 if pattern.search(prediction) else 0.0


def _pass_at_1(prediction: str, reference: dict) -> float:
    code = _extract_code_block(prediction)
    if not code:
        return 0.0

    test_code  = str(reference.get("test_code") or "")
    entry_point = str(reference.get("entry_point") or "")

    if _sandboxed_pass_at_1 is not None and test_code and entry_point:
        try:
            return float(_sandboxed_pass_at_1(
                code, test_code=test_code, entry_point=entry_point, timeout_seconds=10
            ))
        except Exception as exc:
            logger.warning("Sandboxed pass_at_1 failed, falling back: %s", exc)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


_EMPTY_METRICS: dict = {
    "token_f1":    None,
    "exact_match": None,
    "accuracy":    None,
    "pass_at_1":   None,
}


def compute_metrics(record: dict) -> tuple[dict, dict]:
    dataset    = record.get("dataset_name", "")
    prediction = record.get("response_text") or ""
    reference  = record.get("reference") or {}
    answers    = reference.get("answers") or []
    failed     = record.get("failed", False)
    answerable = record.get("answerable", True)

    extracted = _extract_answer(prediction, dataset) if (prediction and not failed) else ""

    if dataset in ("musique", "quality"):
        if failed or not prediction:
            em, f1 = 0.0, 0.0
        elif dataset == "musique" and not answerable:
            if extracted == "NOT_FOUND":
                em, f1 = 1.0, 1.0
            else:
                em, f1 = 0.0, 0.0
        else:
            em = _best_exact_match(extracted, answers)
            f1 = _best_token_f1(extracted, answers)

        metrics = {**_EMPTY_METRICS, "token_f1": f1, "exact_match": em}

        meta: dict = {"extracted_answer": extracted}
        if dataset == "musique":
            meta["false_negative"] = bool(
                answerable and extracted == "NOT_FOUND"
            ) or bool(
                answerable and f1 == 0.0 and em == 0.0 and extracted != "NOT_FOUND"
            )
        return metrics, meta

    if dataset == "fever":
        acc = 0.0 if (failed or not prediction) else _fever_label_accuracy(prediction, answers)
        return {**_EMPTY_METRICS, "accuracy": acc}, {"extracted_answer": extracted}

    if dataset == "hardmath":
        if failed or not prediction:
            acc = 0.0
        else:
            gold_text  = answers[0] if answers else ""
            gold_boxed = _extract_boxed(gold_text)
            pred_boxed = _extract_boxed(prediction)
            if gold_boxed and pred_boxed:
                acc = 1.0 if gold_boxed.lower() == pred_boxed.lower() else 0.0
            elif gold_boxed and not pred_boxed:
                acc = 0.0
            else:
                gold_ans = gold_boxed or gold_text.strip()
                pred_ans = pred_boxed or extracted
                acc = 1.0 if gold_ans.lower() == pred_ans.lower() else 0.0
        return {**_EMPTY_METRICS, "accuracy": acc}, {"extracted_answer": extracted}

    if dataset == "humaneval":
        p1 = 0.0 if (failed or not prediction) else _pass_at_1(prediction, reference)
        return {**_EMPTY_METRICS, "pass_at_1": p1}, {"extracted_answer": extracted}

    return dict(_EMPTY_METRICS), {"extracted_answer": extracted}


# ── CLI parsing ───────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P3 rule-based routing policy against a LOCAL dataset. "
            "No Hugging Face Hub calls are made."
        )
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help=(
            "Path to the evaluation dataset. Accepts a .jsonl/.ndjson file "
            "(one row per line), a .json file containing an array of rows, "
            "or a directory produced by `Dataset.save_to_disk(...)`."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f"Where to write the metric-ready JSONL. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: cap the number of cases evaluated (for smoke tests).",
    )
    return parser.parse_args(argv)


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    suite_id = f"suite_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"Suite: {suite_id}")
    print(f"Loading local dataset from: {args.dataset_path}")

    cases: list[EvaluationCase] = []
    for i, row in enumerate(_iter_rows_from_path(args.dataset_path)):
        if args.limit is not None and i >= args.limit:
            break
        dataset_name = _normalize_dataset(row.get("source_dataset", ""))
        reference = _build_reference(row, dataset_name)
        system_prompt = _build_system_prompt(dataset_name)

        question = row["question"]
        if system_prompt:
            question = f"{system_prompt}\n\n{question}"

        metadata: dict = {}
        if dataset_name == "musique":
            metadata["answerable"] = bool(row.get("answerable", True))

        case = EvaluationCase(
            example=BenchmarkExample(
                question=question,
                context=row.get("context") or None,
                system_prompt=None,
                example_id=row.get("example_id", str(i)),
                dataset_name=dataset_name,
            ),
            reference=reference,
            metadata=metadata,
        )
        cases.append(case)

    print(f"Dataset loaded. {len(cases)} cases prepared.")

    if not cases:
        print("No cases loaded — aborting. Check --dataset-path.")
        return

    async with ModelOrchestrator(vllm_specialist_config) as orchestrator:
        policy = RuleBasedRoutingPolicy(orchestrator)

        print("Running P3 routing policy...")
        result = await policy.run(cases)

    print("Building metric-ready output records...")
    output_records: list[dict] = []

    for qr in result.results:
        case = qr.case
        example = case.example
        resp = qr.response

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

        record: dict = {
            "example_id":   example.example_id,
            "dataset_name": example.dataset_name,
            "suite_id":     suite_id,
            "question":     example.question,
            "system_prompt": _build_system_prompt(example.dataset_name),
            "context":      example.context,
            "response_text": resp.text if resp is not None else None,
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

        record["metrics"], record["metric_metadata"] = compute_metrics(record)

        output_records.append(record)

    output_path = Path(args.output_file)
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
    args = _parse_args()
    print("Starting P3 policy evaluation CLI (local-only, cli_late variant)...")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
