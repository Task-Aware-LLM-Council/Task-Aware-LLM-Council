"""Bucket HumanEvalPlus failures by failure mode.

The aggregate p@1=0.100 is low; this script splits failures into:

- EXTRACTION_FAILED: extract_code_answer returned empty — the P4 response
  doesn't contain a parseable code block. Usually means the specialist
  wrote prose/pseudo-code instead of real Python.
- NO_ENTRY_POINT: couldn't locate the expected function name in the
  extracted code — function missing or renamed.
- SYNTAX_ERROR: extracted code doesn't parse.
- RUNTIME_ERROR: code parses but raises during execution (NameError,
  TypeError, ImportError, etc) — usually missing imports or wrong API.
- ASSERT_FAILED: code runs and returns, but unit test assertions fail —
  logic is wrong.
- TIMEOUT: code hangs past the subprocess timeout.
- PASS: works.

Also reports:
- router role each row was dispatched to (look for misroutes to qa/general)
- synthesis_used distribution (long CoT reasoning preamble often survives
  synth and breaks extractors)

Usage:
    uv run --package task_eval python scratch/diagnose_humaneval_failures.py \\
        --results p4_gemma_lora_v2_rd2_val.jsonl
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from task_eval.extraction import extract_code_answer


_ENTRY_RE = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)


def parse_entry_point(prompt: str) -> str:
    m = _ENTRY_RE.search(prompt or "")
    return m.group(1) if m else ""


def classify_failure(
    prediction_raw: str, unit_tests: str, entry_point: str,
    timeout_seconds: int = 10,
) -> tuple[str, str]:
    """Return (bucket, detail). detail is a short string with evidence."""
    pred_code = extract_code_answer(prediction_raw)
    if not pred_code.strip():
        return "EXTRACTION_FAILED", f"len(raw)={len(prediction_raw)}"

    if not entry_point:
        return "NO_ENTRY_POINT", "could not parse entry_point"
    if entry_point not in pred_code:
        return "NO_ENTRY_POINT", f"{entry_point} missing from extracted code"

    # Compile-test
    try:
        ast.parse(pred_code)
    except SyntaxError as e:
        return "SYNTAX_ERROR", f"{type(e).__name__}: {e.msg} (line {e.lineno})"

    # Run under subprocess with timeout
    imports = (
        "import math\n"
        "from typing import *\n"
        "import collections\n"
        "import itertools\n"
        "import re\n"
    )
    program = (
        f"{imports}\n{pred_code}\n\n{unit_tests}\n\n"
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
        except subprocess.TimeoutExpired:
            return "TIMEOUT", f">{timeout_seconds}s"
        except Exception as e:
            return "RUNTIME_ERROR", f"subprocess_error: {e}"

    if r.returncode == 0:
        return "PASS", ""

    stderr = (r.stderr or "").strip()
    last_line = stderr.splitlines()[-1] if stderr else ""
    if "AssertionError" in stderr:
        return "ASSERT_FAILED", last_line[:200]
    return "RUNTIME_ERROR", last_line[:200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--dataset", default="task-aware-llm-council/router_dataset-2")
    p.add_argument("--split", default="validation")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--sample", type=int, default=3,
                   help="sample rows per bucket to print")
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    by_index = {i: row for i, row in enumerate(ds)}

    rows = []
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("source_dataset") == "HumanEvalPlus":
                rows.append(r)

    print(f"=== HumanEvalPlus diagnostic ({len(rows)} rows) ===\n")

    buckets: Counter[str] = Counter()
    role_by_bucket: dict[str, Counter[str]] = {}
    synth_by_bucket: dict[str, Counter[object]] = {}
    examples: dict[str, list[tuple[int, str, str]]] = {}

    for r in rows:
        idx = r.get("index")
        gold_row = by_index.get(idx) or {}
        unit_tests = gold_row.get("unit_tests") or ""
        entry_point = parse_entry_point(gold_row.get("question") or "")
        pred_raw = r.get("p4_answer") or ""

        if r.get("error"):
            bucket, detail = "P4_ERROR", str(r.get("error"))[:200]
        else:
            bucket, detail = classify_failure(
                pred_raw, unit_tests, entry_point, args.timeout,
            )
        buckets[bucket] += 1

        # predicted_route is stored at top-level in newer JSONLs, in metadata for older
        route = (r.get("predicted_route")
                 or (r.get("metadata") or {}).get("predicted_route")
                 or "UNKNOWN")
        if isinstance(route, list):
            route_key = "+".join(route) if route else "(none)"
        else:
            route_key = str(route)
        role_by_bucket.setdefault(bucket, Counter())[route_key] += 1

        synth = (r.get("synthesis_used")
                 if r.get("synthesis_used") is not None
                 else (r.get("metadata") or {}).get("synthesis_used"))
        synth_by_bucket.setdefault(bucket, Counter())[synth] += 1

        examples.setdefault(bucket, []).append((idx, entry_point, detail))

    ordered = [
        "PASS", "EXTRACTION_FAILED", "NO_ENTRY_POINT", "SYNTAX_ERROR",
        "ASSERT_FAILED", "RUNTIME_ERROR", "TIMEOUT", "P4_ERROR",
    ]
    print(f"{'bucket':<22} {'n':>4}  top_routes")
    print("-" * 60)
    for b in ordered:
        if b not in buckets:
            continue
        n = buckets[b]
        routes = role_by_bucket.get(b, Counter()).most_common(3)
        route_str = ", ".join(f"{k}:{v}" for k, v in routes)
        print(f"{b:<22} {n:>4}  {route_str}")

    # synthesis distribution for the two biggest non-pass buckets
    failure_buckets = [b for b in ordered if b != "PASS" and b in buckets]
    for b in failure_buckets[:2]:
        synth_dist = dict(synth_by_bucket.get(b, Counter()))
        print(f"\n{b}: synthesis_used = {synth_dist}")

    # samples
    for b in failure_buckets:
        ex = examples.get(b) or []
        if not ex:
            continue
        print(f"\n[{b}] sample rows (up to {args.sample}):")
        for idx, ep, detail in ex[: args.sample]:
            print(f"  idx={idx}  entry={ep}  detail: {detail}")

    print("\n=== interpretation ===")
    pred_dominant = max(
        (b for b in buckets if b != "PASS"),
        key=lambda k: buckets[k],
        default=None,
    )
    if pred_dominant == "EXTRACTION_FAILED":
        print("Dominant failure: EXTRACTION_FAILED. Specialist is writing "
              "text/pseudo-code instead of real Python. Look at a few "
              "p4_answer samples — fix by tightening the HumanEval prompt "
              "template or using the reasoning specialist exclusively.")
    elif pred_dominant == "ASSERT_FAILED":
        print("Dominant failure: ASSERT_FAILED. Code compiles and runs but "
              "fails unit tests — specialist is getting the logic wrong. "
              "This is a model-quality issue; scorer tweaks won't help.")
    elif pred_dominant == "RUNTIME_ERROR":
        print("Dominant failure: RUNTIME_ERROR. Often missing imports or "
              "wrong API usage. Check if additional imports would help.")
    elif pred_dominant == "NO_ENTRY_POINT":
        print("Dominant failure: NO_ENTRY_POINT. Extractor finds code but "
              "the expected function name is missing — specialist renamed "
              "the function or answered with a different signature.")


if __name__ == "__main__":
    main()
