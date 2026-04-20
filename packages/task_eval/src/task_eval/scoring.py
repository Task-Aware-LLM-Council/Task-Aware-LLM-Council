from __future__ import annotations

import logging
import math
import re
import subprocess
import tempfile
import os
import shutil
import traceback
from collections import Counter

from task_eval.normalization import normalize_answer, normalize_fever_label, tokenize_normalized

logger = logging.getLogger(__name__)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def exact_match_multi(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(exact_match(prediction, reference) for reference in references)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_normalized(prediction)
    ref_tokens = tokenize_normalized(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1_multi(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(token_f1(prediction, reference) for reference in references)


def label_accuracy(prediction: str, reference: str) -> float:
    return float(
        normalize_fever_label(prediction) == normalize_fever_label(reference)
    )


def _parse_number(text: str) -> float | None:
    candidate = (text or "").strip().replace(",", "")
    frac_match = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", candidate)
    if frac_match:
        numerator = float(frac_match.group(1))
        denominator = float(frac_match.group(2))
        return None if denominator == 0 else numerator / denominator
    try:
        return float(candidate)
    except ValueError:
        numbers = re.findall(r"[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?", candidate)
        return float(numbers[-1]) if numbers else None


def numeric_accuracy(
    prediction: str,
    reference: str,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-6,
) -> float:
    logger.debug("prediction:%s - reference:%s", prediction, reference)
    predicted = _parse_number(prediction)
    gold = _parse_number(reference)
    if predicted is None or gold is None:
        return 0.0
    return float(math.isclose(predicted, gold, rel_tol=rel_tol, abs_tol=abs_tol))


def normalize_latex(text: str) -> str:
    """Cleans LaTeX strings to ensure fair string comparison."""
    if not text:
        return ""
    # Remove whitespace
    text = re.sub(r"\s+", "", text)
    # Remove common formatting wrappers
    text = text.replace("\\left", "").replace("\\right", "")
    # Remove optional outermost braces
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return text


def math_exact_match(prediction: str, reference: str) -> float:
    """Scores symbolic math datasets by comparing normalized strings."""
    norm_pred = normalize_latex(prediction)
    norm_ref = normalize_latex(reference)
    return 1.0 if (norm_pred and norm_pred == norm_ref) else 0.0

def _docker_available() -> bool:
    """Check if Docker binary exists and daemon is running."""
    if shutil.which("docker") is None:
        return False

    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=True,
        )
        return True
    except Exception:
        return False


def _apptainer_available() -> bool:
    """Check if Apptainer binary exists."""
    if shutil.which("apptainer") is None:
        return False

    try:
        subprocess.run(
            ["apptainer", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=True,
        )
        return True
    except Exception:
        return False

def pass_at_1(
    prediction: str,
    *,
    test_code: str,
    entry_point: str,
    timeout_seconds: int = 10,
) -> float:
    if not prediction.strip():
        return 0.0

    if not _apptainer_available():
        logger.warning("Apptainer not available")
        return 0.0
    # if not _docker_available():
    #     print("Docker not available or not running")
    #     return 0.0

    standard_imports = (
        "import math\n"
        "from typing import *\n"
        "import collections\n"
        "import itertools\n"
        "import re\n"
    )

    program = f"""
{standard_imports}

{prediction}

{test_code}

if __name__ == "__main__":
    check({entry_point})
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "script.py")

        with open(file_path, "w") as f:
            f.write(program)

        os.chmod(temp_dir, 0o755)
        os.chmod(file_path, 0o644)

        # docker_cmd = [
        #     "docker", "run",
        #     "--rm",
        #     "--network", "none",
        #     "--memory", "256m",
        #     "--cpus", "1.0",
        #     "--pids-limit", "64",
        #     "--read-only",
        #     "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        #     "--security-opt", "no-new-privileges",
        #     "--cap-drop", "ALL",
        #     "-v", f"{temp_dir}:/code:ro",
        #     "-w", "/code",
        #     "python:3.10-slim",
        #     "python3", "/code/script.py",
        # ]

        # Build Apptainer sandboxed command
        apptainer_cmd = [
            "apptainer", "exec",
            "--containall",          # Isolates PID, IPC, and creates a clean, ephemeral /tmp
            "--net",                 # Enables network namespace isolation
            "--network=none",        # Explicitly disables networking
            "--cleanenv",            # Prevents host environment variables from leaking
            "--bind", f"{temp_dir}:/code:ro",
            "--pwd", "/code",
            "python_3.10-slim.sif",  # Apptainer automatically pulls and caches from Docker Hub
            "python3", "script.py",
        ]

        try:
            # result = subprocess.run(
            #     docker_cmd,
            #     capture_output=True,
            #     text=True,
            #     timeout=timeout_seconds + 2,  # buffer for Docker startup
            # )
            # # Success = exit code 0
            # return float(result.returncode == 0)

            result = subprocess.run(
                apptainer_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 15,  # Slightly higher buffer for Apptainer startup
            )
            # Success = exit code 0
            return float(result.returncode == 0)

        except subprocess.TimeoutExpired:
            print("Execution timed out (host-level)")
            return 0.0

        except Exception as e:
            print("Unexpected error:", e)
            traceback.print_exc()
            return 0.0


def aggregate_numeric_metrics(records: list[dict[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for record in records:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, int | float):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

    return {
        key: totals[key] / counts[key]
        for key in totals
        if counts.get(key, 0) > 0
    }
