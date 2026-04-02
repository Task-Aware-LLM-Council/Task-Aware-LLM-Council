"""
evaluation.py — Evaluation Suite for Task-Aware LLM Council Benchmarking
=========================================================================

Metrics per dataset:
    MuSiQue         → Exact Match (EM), Token F1 (SQuAD-style)
    QuALITY         → Exact Match (EM), Token F1 (SQuAD-style)
    FEVER           → Label Accuracy (3-way: Supported / Refuted / NEI)
    HARDMath        → Numeric Accuracy (within relative tolerance)
    HumanEval+      → Pass@1 (sandboxed code execution)

Cross-cutting:
    Hallucination Rate  → fraction of outputs containing unsupported claims
    Avg Tokens / Q      → cost proxy (prompt + completion tokens)
    Routing Accuracy    → classification accuracy of skill-tag prediction (P3/P4)

Usage:
    from evaluation import evaluate_sample, evaluate_batch, generate_report
"""

from __future__ import annotations

import json
import math
import re
import string
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1: TEXT NORMALIZATION (SQuAD-standard)
# ═════════════════════════════════════════════════════════════════════════

def normalize_answer(text: str) -> str:
    """
    SQuAD-style normalization:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Collapse whitespace
    """
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return normalize_answer(text).split()


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2: ANSWER EXTRACTION
# ═════════════════════════════════════════════════════════════════════════
# Models don't return clean answers — they return paragraphs with
# reasoning, chain-of-thought, markdown, etc. These extractors pull
# out the actual answer from raw model output.

def extract_answer(response: str, dataset: str) -> str:
    """
    Route to the appropriate answer extractor based on dataset.

    Args:
        response: Raw model output string.
        dataset:  One of "musique", "quality", "fever", "hardmath", "humaneval_plus".

    Returns:
        Extracted answer string.
    """
    if not response or not response.strip():
        return ""

    extractors = {
        "musique": extract_qa_answer,
        "quality": extract_mcq_answer,
        "fever": extract_fever_label,
        "hardmath": extract_math_answer,
        "humaneval_plus": extract_code_answer,
    }

    extractor = extractors.get(dataset)
    if extractor is None:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of {list(extractors.keys())}")

    return extractor(response)


def extract_qa_answer(response: str) -> str:
    """
    Extract short answer from QA response (MuSiQue).

    Strategy:
    1. If the model put the answer after "Answer:" or "answer is", take that.
    2. Otherwise, take the first non-preamble line.
    3. Fallback: first line.
    """
    response = response.strip()

    # Pattern: "The answer is ..."
    match = re.search(
        r"(?:the answer is|answer:|final answer:)\s*(.+?)(?:\.|$)",
        response, re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Take first non-preamble line
    preamble_starters = (
        "let me", "based on", "according to", "looking at", "from the",
        "to answer", "the passage", "in the", "first,", "step",
    )
    for line in response.split("\n"):
        line = line.strip()
        if line and not line.lower().startswith(preamble_starters):
            # Remove trailing period if it looks like a short answer
            if len(line.split()) <= 10:
                return line.rstrip(".")
            return line

    return response.split("\n")[0].strip()


def extract_mcq_answer(response: str) -> str:
    """
    Extract letter answer (A/B/C/D) from MCQ response (QuALITY).

    Strategy:
    1. Direct letter at start: "B" or "(B)"
    2. Pattern: "answer is (B)" or "answer: B"
    3. Fallback: first standalone A-D letter found
    """
    response = response.strip()

    # Direct letter answer at start
    match = re.match(r"^\(?([A-Da-d])\)?[\s\.\,\:]", response)
    if match:
        return match.group(1).upper()

    # Just a single letter
    if re.match(r"^[A-Da-d]$", response):
        return response.upper()

    # "The answer is (B)" or "Answer: B"
    match = re.search(
        r"(?:answer is|answer:)\s*\(?([A-Da-d])\)?",
        response, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Fallback: find any standalone letter A-D
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1)

    # Last resort
    return response.strip()[0].upper() if response.strip() else ""


def extract_fever_label(response: str) -> str:
    """
    Extract FEVER label: Supported / Refuted / NEI.

    Handles common model output variations:
    - "Supported" / "SUPPORTS" / "True" → "SUPPORTED"
    - "Refuted" / "REFUTES" / "False"   → "REFUTED"
    - "NEI" / "Not Enough Information"   → "NEI"
    """
    text = response.strip().lower()

    # Check for each label in order of specificity
    if any(kw in text for kw in ["not enough information", "nei", "cannot determine",
                                   "insufficient", "not enough evidence", "cannot be determined"]):
        return "NEI"
    if any(kw in text for kw in ["refuted", "refutes", "refute", "false", "contradicted",
                                   "incorrect", "not supported"]):
        return "REFUTED"
    if any(kw in text for kw in ["supported", "supports", "support", "true", "confirmed",
                                   "correct", "verified"]):
        return "SUPPORTED"

    # Fallback: return the first word
    first_word = text.split()[0] if text.split() else ""
    return first_word.upper()


def extract_math_answer(response: str) -> str:
    """
    Extract final numerical answer from math response (HARDMath).

    Strategy:
    1. Look for explicit "ANSWER: X" or "Final Answer: X"
    2. Look for boxed answer: \\boxed{X}
    3. Look for "the answer is X"
    4. Fallback: last number in the response
    """
    response = response.strip()

    # Explicit ANSWER: pattern
    match = re.search(r"(?:ANSWER|Final Answer)\s*[:=]\s*([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)", response, re.IGNORECASE)
    if match:
        return match.group(1)

    # LaTeX boxed answer: \boxed{42}
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1).strip()

    # "the answer is X"
    match = re.search(
        r"(?:the answer is|equals|result is|approximately)\s*([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)",
        response, re.IGNORECASE
    )
    if match:
        return match.group(1)

    # Fallback: last number in the response
    numbers = re.findall(r"[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?", response)
    if numbers:
        return numbers[-1]

    return ""


def extract_code_answer(response: str) -> str:
    """
    Extract Python code from response (HumanEval+).

    Strategy:
    1. Extract content inside ```python ... ``` fences
    2. If no fences, take the raw response
    3. Strip any leading explanation before the first `def `
    """
    response = response.strip()

    # Extract from markdown code fences
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If response starts with explanation before code, try to find first `def `
    def_match = re.search(r"((?:from .+\n|import .+\n)*def .+)", response, re.DOTALL)
    if def_match:
        return def_match.group(1).strip()

    # Remove markdown fences if present (without language tag)
    response = re.sub(r"^```\s*\n?", "", response)
    response = re.sub(r"\n?```\s*$", "", response)

    return response.strip()


# ═════════════════════════════════════════════════════════════════════════
# SECTION 3: SCORING METRICS
# ═════════════════════════════════════════════════════════════════════════

# ── 3.1 Exact Match ──────────────────────────────────────────────────

def exact_match(prediction: str, reference: str) -> float:
    """
    SQuAD-style Exact Match.
    Returns 1.0 if normalized prediction equals normalized reference, else 0.0.

    For datasets with multiple valid answers, call this for each and take max.
    """
    return float(normalize_answer(prediction) == normalize_answer(reference))


def exact_match_multi(prediction: str, references: list[str]) -> float:
    """EM with multiple valid reference answers (take best match)."""
    if not references:
        return 0.0
    return max(exact_match(prediction, ref) for ref in references)


# ── 3.2 Token F1 (SQuAD-style) ──────────────────────────────────────

def token_f1(prediction: str, reference: str) -> float:
    """
    SQuAD-style Token F1 Score.

    Computes precision and recall over the token overlap between the
    normalized prediction and normalized reference, then returns their
    harmonic mean (F1).

    Returns:
        float: F1 score in [0.0, 1.0].
    """
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)

    # Edge cases
    if not ref_tokens and not pred_tokens:
        return 1.0  # both empty → perfect match
    if not ref_tokens or not pred_tokens:
        return 0.0

    # Count common tokens (order-independent, with multiplicity)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def token_f1_multi(prediction: str, references: list[str]) -> float:
    """Token F1 with multiple valid reference answers (take best match)."""
    if not references:
        return 0.0
    return max(token_f1(prediction, ref) for ref in references)


# ── 3.3 Label Accuracy (FEVER) ──────────────────────────────────────

def label_accuracy(prediction: str, reference: str) -> float:
    """
    Exact match on 3-way FEVER labels: SUPPORTED / REFUTED / NEI.

    Both prediction and reference should be label strings.
    Normalization handles common aliases (True→SUPPORTED, False→REFUTED, etc.)
    """
    label_map = {
        "supported": "SUPPORTED", "supports": "SUPPORTED", "support": "SUPPORTED",
        "true": "SUPPORTED", "confirmed": "SUPPORTED", "verified": "SUPPORTED",
        "correct": "SUPPORTED",
        "refuted": "REFUTED", "refutes": "REFUTED", "refute": "REFUTED",
        "false": "REFUTED", "contradicted": "REFUTED", "incorrect": "REFUTED",
        "not supported": "REFUTED",
        "nei": "NEI", "not enough information": "NEI", "unknown": "NEI",
        "cannot determine": "NEI", "insufficient": "NEI",
        "cannot be determined": "NEI", "not enough evidence": "NEI",
    }

    pred_normalized = label_map.get(prediction.strip().lower(), prediction.strip().upper())
    ref_normalized = label_map.get(reference.strip().lower(), reference.strip().upper())

    return float(pred_normalized == ref_normalized)


# ── 3.4 Numeric Accuracy (HARDMath) ─────────────────────────────────

def numeric_accuracy(prediction: str, reference: str, rel_tol: float = 0.01, abs_tol: float = 1e-6) -> float:
    """
    Compare two numeric strings within tolerance.

    Uses both relative tolerance (default 1%) and absolute tolerance.
    Returns 1.0 if match, 0.0 otherwise.

    Args:
        prediction: Predicted answer string (may contain non-numeric text).
        reference:  Gold answer string.
        rel_tol:    Relative tolerance (default 0.01 = 1%).
        abs_tol:    Absolute tolerance for near-zero answers.
    """
    def parse_number(text: str) -> float | None:
        """Try to parse a number from a string."""
        text = text.strip()
        # Remove commas in numbers like "1,234.56"
        text = text.replace(",", "")
        # Handle fractions like "3/4"
        frac_match = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", text)
        if frac_match:
            num, den = float(frac_match.group(1)), float(frac_match.group(2))
            return num / den if den != 0 else None
        # Handle scientific notation and plain numbers
        try:
            return float(text)
        except ValueError:
            # Try to find a number in the string
            nums = re.findall(r"[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?", text)
            if nums:
                return float(nums[-1])
            return None

    pred_num = parse_number(prediction)
    ref_num = parse_number(reference)

    if pred_num is None or ref_num is None:
        return 0.0

    return float(math.isclose(pred_num, ref_num, rel_tol=rel_tol, abs_tol=abs_tol))


# ── 3.5 Pass@1 (HumanEval+) ─────────────────────────────────────────

def pass_at_1(
    prediction: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> float:
    """
    Execute predicted code against test cases in a sandboxed subprocess.

    Returns 1.0 if all tests pass, 0.0 otherwise.

    SECURITY NOTE:
        This executes untrusted LLM-generated code. On CARC, run this on
        compute nodes (not login nodes). The subprocess has:
        - Timeout (default 10s)
        - No network access (inherited from sandbox)
        - Captured stdout/stderr
    """
    if not prediction.strip():
        return 0.0

    # Combine function definition + test harness
    full_code = f"{prediction}\n\n{test_code}\n\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=True, dir="/tmp"
        ) as f:
            f.write(full_code)
            f.flush()

            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return float(result.returncode == 0)

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k from the Codex paper.

    Given n total samples and c correct samples, estimates the probability
    that at least one of k random samples is correct.

    Args:
        n: Total number of code samples generated.
        c: Number of samples that pass all tests.
        k: Number of samples to draw.

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ═════════════════════════════════════════════════════════════════════════
# SECTION 4: DATASET-AWARE EVALUATION DISPATCHER
# ═════════════════════════════════════════════════════════════════════════

# Dataset → metric configuration
DATASET_METRICS = {
    "musique": {
        "metrics": ["exact_match", "token_f1"],
        "primary": "token_f1",
    },
    "quality": {
        "metrics": ["exact_match", "token_f1"],
        "primary": "token_f1",
    },
    "fever": {
        "metrics": ["label_accuracy"],
        "primary": "label_accuracy",
    },
    "hardmath": {
        "metrics": ["numeric_accuracy"],
        "primary": "numeric_accuracy",
    },
    "humaneval_plus": {
        "metrics": ["pass_at_1"],
        "primary": "pass_at_1",
    },
}


@dataclass
class SampleResult:
    """Result of evaluating one sample."""
    sample_id: str
    dataset: str
    model: str
    raw_response: str
    extracted_answer: str
    reference: str
    scores: dict[str, float]
    tokens_used: int = 0
    latency_seconds: float = 0.0


def evaluate_sample(
    raw_response: str,
    dataset: str,
    reference: str,
    metadata: Optional[dict] = None,
) -> dict[str, float]:
    """
    Evaluate a single model response against its gold reference.

    This is the main entry point for scoring. It:
    1. Extracts the answer from the raw model response.
    2. Computes all metrics configured for the dataset.
    3. Returns a dict of {metric_name: score}.

    Args:
        raw_response: Raw model output string.
        dataset:      Dataset name (determines which metrics + extractor to use).
        reference:    Gold answer string.
        metadata:     Optional dict with extra info (e.g., test_code for HumanEval+).

    Returns:
        dict: {metric_name: float_score, "extracted_answer": str}
    """
    if metadata is None:
        metadata = {}

    config = DATASET_METRICS.get(dataset)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Step 1: Extract answer
    extracted = extract_answer(raw_response, dataset)

    # Step 2: Compute each configured metric
    scores = {}

    for metric_name in config["metrics"]:
        if metric_name == "exact_match":
            scores["exact_match"] = exact_match(extracted, reference)

        elif metric_name == "token_f1":
            scores["token_f1"] = token_f1(extracted, reference)

        elif metric_name == "label_accuracy":
            scores["label_accuracy"] = label_accuracy(extracted, reference)

        elif metric_name == "numeric_accuracy":
            scores["numeric_accuracy"] = numeric_accuracy(extracted, reference)

        elif metric_name == "pass_at_1":
            scores["pass_at_1"] = pass_at_1(
                extracted,
                metadata.get("test", ""),
                metadata.get("entry_point", ""),
            )

    scores["extracted_answer"] = extracted
    return scores


# ═════════════════════════════════════════════════════════════════════════
# SECTION 5: BATCH EVALUATION & AGGREGATION
# ═════════════════════════════════════════════════════════════════════════

def evaluate_batch(results: list[SampleResult]) -> dict:
    """
    Aggregate scores across a batch of SampleResults.

    Returns per-metric averages and the count of samples.
    """
    if not results:
        return {"count": 0}

    # Collect all metric names from first result
    metric_names = [k for k in results[0].scores.keys() if k != "extracted_answer"]

    aggregated = {"count": len(results)}
    for metric in metric_names:
        values = [r.scores.get(metric, 0.0) for r in results]
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    # Cost metrics
    if any(r.tokens_used > 0 for r in results):
        token_values = [r.tokens_used for r in results]
        aggregated["avg_tokens_per_question"] = float(np.mean(token_values))

    if any(r.latency_seconds > 0 for r in results):
        latency_values = [r.latency_seconds for r in results]
        aggregated["avg_latency_seconds"] = float(np.mean(latency_values))

    return aggregated


def generate_report(
    all_results: dict[str, dict[str, list[SampleResult]]],
    output_path: Optional[Path] = None,
) -> dict:
    """
    Generate a full benchmarking report from nested results.

    Args:
        all_results: Nested dict of {model_name: {dataset_name: [SampleResult, ...]}}.
        output_path: Optional path to save the JSON report.

    Returns:
        Report dict with per-model, per-dataset, and aggregate scores.
    """
    report = {
        "models": {},
        "datasets": {},
        "leaderboard": [],
    }

    model_avg_ranks = {}

    # ── Per-model, per-dataset scores ────────────────
    for model_name, datasets in all_results.items():
        report["models"][model_name] = {}

        for dataset_name, sample_results in datasets.items():
            agg = evaluate_batch(sample_results)
            primary_metric = DATASET_METRICS[dataset_name]["primary"]
            agg["primary_metric"] = primary_metric
            agg["primary_score"] = agg.get(primary_metric, {}).get("mean", 0.0)
            report["models"][model_name][dataset_name] = agg

    # ── Per-dataset rankings ─────────────────────────
    for dataset_name in DATASET_METRICS.keys():
        primary_metric = DATASET_METRICS[dataset_name]["primary"]
        model_scores = []

        for model_name in report["models"]:
            if dataset_name in report["models"][model_name]:
                score = report["models"][model_name][dataset_name]["primary_score"]
                model_scores.append((model_name, score))

        # Sort by score descending → assign ranks
        model_scores.sort(key=lambda x: x[1], reverse=True)
        rankings = []
        for rank, (model, score) in enumerate(model_scores, 1):
            rankings.append({"rank": rank, "model": model, "score": round(score, 4)})
            model_avg_ranks.setdefault(model, []).append(rank)

        report["datasets"][dataset_name] = {
            "primary_metric": primary_metric,
            "rankings": rankings,
            "best_model": model_scores[0][0] if model_scores else None,
            "best_score": round(model_scores[0][1], 4) if model_scores else 0.0,
        }

    # ── Overall leaderboard (by average rank) ────────
    leaderboard = []
    for model, ranks in model_avg_ranks.items():
        avg_rank = sum(ranks) / len(ranks)
        leaderboard.append({
            "model": model,
            "avg_rank": round(avg_rank, 2),
            "ranks": {
                ds: next(
                    (r["rank"] for r in report["datasets"][ds]["rankings"] if r["model"] == model),
                    None
                )
                for ds in DATASET_METRICS.keys()
            },
        })
    leaderboard.sort(key=lambda x: x["avg_rank"])
    report["leaderboard"] = leaderboard

    # ── Specialist map (best model per dataset) ──────
    report["specialist_map"] = {
        ds: info["best_model"]
        for ds, info in report["datasets"].items()
    }
    report["best_overall_model"] = leaderboard[0]["model"] if leaderboard else None

    # ── Save ─────────────────────────────────────────
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))

    return report


# ═════════════════════════════════════════════════════════════════════════
# SECTION 6: UNIT TESTS (run with: python evaluation.py --test)
# ═════════════════════════════════════════════════════════════════════════

def run_tests():
    """Comprehensive test suite for all metrics and extractors."""
    passed = 0
    failed = 0

    def assert_close(name, actual, expected, tol=1e-4):
        nonlocal passed, failed
        if abs(actual - expected) <= tol:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — expected {expected}, got {actual}")

    def assert_eq(name, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — expected {expected!r}, got {actual!r}")

    # ── Normalization ────────────────────────────────
    print("Testing normalization...")
    assert_eq("normalize_basic", normalize_answer("The Eiffel Tower"), "eiffel tower")
    assert_eq("normalize_punctuation", normalize_answer("Hello, world!"), "hello world")
    assert_eq("normalize_articles", normalize_answer("A big red dog"), "big red dog")
    assert_eq("normalize_whitespace", normalize_answer("  lots   of   space  "), "lots of space")

    # ── Exact Match ──────────────────────────────────
    print("Testing exact_match...")
    assert_close("em_identical", exact_match("Paris", "Paris"), 1.0)
    assert_close("em_case_insensitive", exact_match("PARIS", "paris"), 1.0)
    assert_close("em_with_article", exact_match("The Eiffel Tower", "Eiffel Tower"), 1.0)
    assert_close("em_wrong", exact_match("London", "Paris"), 0.0)
    assert_close("em_empty", exact_match("", ""), 1.0)

    # ── Token F1 ─────────────────────────────────────
    print("Testing token_f1...")
    assert_close("f1_perfect", token_f1("big red dog", "big red dog"), 1.0)
    assert_close("f1_partial", token_f1("red dog", "big red dog"), 0.8)  # P=1.0, R=2/3, F1=0.8
    assert_close("f1_no_overlap", token_f1("cat", "dog"), 0.0)
    assert_close("f1_both_empty", token_f1("", ""), 1.0)
    assert_close("f1_pred_empty", token_f1("", "dog"), 0.0)

    # verify the P=1.0, R=2/3 case manually
    # pred = ["red", "dog"], ref = ["big", "red", "dog"]
    # common = 2, P = 2/2 = 1.0, R = 2/3 = 0.667, F1 = 2*(1.0*0.667)/(1.0+0.667) = 0.8
    assert_close("f1_precision_recall", token_f1("red dog", "big red dog"), 0.8)

    # token f1 with articles (articles get removed in normalization)
    assert_close("f1_articles", token_f1("The big red dog", "big red dog"), 1.0)

    # real-world QA example
    assert_close(
        "f1_real_example",
        token_f1("William Shakespeare", "Shakespeare"),
        # pred=["william", "shakespeare"], ref=["shakespeare"]
        # common=1, P=1/2=0.5, R=1/1=1.0, F1=2*0.5*1.0/1.5 = 0.6667
        0.6667,
        tol=0.01,
    )

    # ── Label Accuracy (FEVER) ───────────────────────
    print("Testing label_accuracy...")
    assert_close("fever_exact", label_accuracy("SUPPORTED", "SUPPORTED"), 1.0)
    assert_close("fever_case", label_accuracy("supported", "SUPPORTED"), 1.0)
    assert_close("fever_alias_true", label_accuracy("True", "SUPPORTED"), 1.0)
    assert_close("fever_alias_false", label_accuracy("False", "REFUTED"), 1.0)
    assert_close("fever_nei", label_accuracy("Not Enough Information", "NEI"), 1.0)
    assert_close("fever_wrong", label_accuracy("SUPPORTED", "REFUTED"), 0.0)

    # ── Numeric Accuracy (HARDMath) ──────────────────
    print("Testing numeric_accuracy...")
    assert_close("num_exact", numeric_accuracy("42", "42"), 1.0)
    assert_close("num_close", numeric_accuracy("42.1", "42"), 1.0)  # within 1% tolerance
    assert_close("num_far", numeric_accuracy("50", "42"), 0.0)  # >1% off
    assert_close("num_scientific", numeric_accuracy("3.14e2", "314"), 1.0)
    assert_close("num_fraction", numeric_accuracy("3/4", "0.75"), 1.0)
    assert_close("num_negative", numeric_accuracy("-5.0", "-5"), 1.0)
    assert_close("num_with_text", numeric_accuracy("The answer is 42.0", "42"), 1.0)
    assert_close("num_no_number", numeric_accuracy("no number here", "42"), 0.0)

    # ── Answer Extraction ────────────────────────────
    print("Testing answer extraction...")

    # QA extraction
    assert_eq("extract_qa_direct", extract_qa_answer("Paris"), "Paris")
    assert_eq(
        "extract_qa_with_reasoning",
        extract_qa_answer("Based on the passages, the answer is Paris."),
        "Paris",
    )

    # MCQ extraction
    assert_eq("extract_mcq_direct", extract_mcq_answer("B"), "B")
    assert_eq("extract_mcq_paren", extract_mcq_answer("(C) Some explanation"), "C")
    assert_eq("extract_mcq_sentence", extract_mcq_answer("The answer is (D)."), "D")

    # FEVER extraction
    assert_eq("extract_fever_supported", extract_fever_label("The claim is supported by evidence"), "SUPPORTED")
    assert_eq("extract_fever_refuted", extract_fever_label("This is refuted"), "REFUTED")
    assert_eq("extract_fever_nei", extract_fever_label("There is not enough information"), "NEI")

    # Math extraction
    assert_eq("extract_math_answer", extract_math_answer("Let me compute... ANSWER: 42.5"), "42.5")
    assert_eq("extract_math_boxed", extract_math_answer("Therefore \\boxed{3.14}"), "3.14")
    assert_eq("extract_math_last_num", extract_math_answer("Step 1: 10, Step 2: 20, Final: 30"), "30")

    # Code extraction
    code_with_fences = '```python\ndef add(a, b):\n    return a + b\n```'
    assert_eq("extract_code_fences", extract_code_answer(code_with_fences), "def add(a, b):\n    return a + b")

    # ── Full pipeline (evaluate_sample) ──────────────
    print("Testing evaluate_sample pipeline...")

    # MuSiQue
    result = evaluate_sample("The answer is Paris.", "musique", "Paris")
    assert_close("pipeline_musique_em", result["exact_match"], 1.0)
    assert_close("pipeline_musique_f1", result["token_f1"], 1.0)

    # FEVER
    result = evaluate_sample("This claim is supported by the evidence.", "fever", "SUPPORTED")
    assert_close("pipeline_fever", result["label_accuracy"], 1.0)

    # HARDMath
    result = evaluate_sample("After calculation, ANSWER: 3.14", "hardmath", "3.14159", )
    assert_close("pipeline_hardmath", result["numeric_accuracy"], 1.0)  # within 1% of 3.14159

    # ── Summary ──────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Tests: {total} total, {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"⚠ {failed} TESTS FAILED — fix before running benchmarks")
    print(f"{'='*50}")

    return failed == 0


# ═════════════════════════════════════════════════════════════════════════
# SECTION 7: CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation suite for Task-Aware LLM Council")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--evaluate", type=str, metavar="RESULTS_JSON",
        help="Evaluate a results JSON file (list of {raw_response, dataset, reference, metadata})"
    )
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        exit(0 if success else 1)

    if args.evaluate:
        data = json.loads(Path(args.evaluate).read_text())
        for item in data:
            scores = evaluate_sample(
                item["raw_response"],
                item["dataset"],
                item["reference"],
                item.get("metadata", {}),
            )
            print(f"  {item.get('id', '?'):20s} | {item['dataset']:15s} | {scores}")