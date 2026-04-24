"""Score HumanEvalPlus predictions from a P4 JSONL using pass@1.

Joins each prediction by index against task-aware-llm-council/router_dataset
to get unit_tests, extracts the entry-point name from the prompt, and runs
the candidate code in a subprocess with a timeout.

Note: runs code unsandboxed. Intended only for trusted public test suites
(HumanEval). Do not point at untrusted unit_tests.

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/score_humaneval_pass1.py \\
        --results p4_gemma_lora_v2.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset
from task_eval.extraction import extract_code_answer


_ENTRY_RE = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)


def parse_entry_point(prompt: str) -> str:
    m = _ENTRY_RE.search(prompt or "")
    return m.group(1) if m else ""


def run_pass_at_1(prediction: str, unit_tests: str, entry_point: str,
                  timeout_seconds: int = 10) -> int:
    """Returns 1 on test success, 0 otherwise."""
    if not prediction.strip() or not entry_point:
        return 0

    imports = (
        "import math\n"
        "from typing import *\n"
        "import collections\n"
        "import itertools\n"
        "import functools\n"
        "import heapq\n"
        "import re\n"
        "import string\n"
        "from decimal import *\n"
        "from fractions import *\n"
        "from statistics import *\n"
    )
    program = (
        f"{imports}\n{prediction}\n\n{unit_tests}\n\n"
        f'if __name__ == "__main__":\n    check({entry_point})\n'
    )

    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "script.py")
        with open(script, "w") as f:
            f.write(program)
        try:
            r = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
            return int(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0
        except Exception:
            return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--dataset", default="task-aware-llm-council/router_dataset")
    p.add_argument("--split", default="validation")
    p.add_argument("--timeout", type=int, default=10)
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    by_index = {i: row for i, row in enumerate(ds)}

    n = passes = 0
    fails: list[tuple[int, str]] = []

    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source_dataset") != "HumanEvalPlus":
                continue
            if row.get("error"):
                n += 1
                continue
            idx = row["index"]
            gold_row = by_index.get(idx)
            if gold_row is None:
                continue

            unit_tests = gold_row.get("unit_tests") or ""
            entry_point = parse_entry_point(gold_row.get("question") or "")
            pred_code = extract_code_answer(row.get("p4_answer") or "")

            result = run_pass_at_1(
                pred_code, unit_tests, entry_point, args.timeout,
            )
            n += 1
            passes += result
            if not result:
                fails.append((idx, entry_point))

    print(f"HumanEvalPlus pass@1: {passes}/{n} = "
          f"{(passes / n) if n else 0.0:.3f}")
    if fails:
        print(f"Failed indices ({len(fails)}): "
              f"{', '.join(f'{i}:{e}' for i, e in fails[:15])}"
              + (" ..." if len(fails) > 15 else ""))


if __name__ == "__main__":
    main()
