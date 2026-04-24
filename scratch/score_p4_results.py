"""Score a P4 benchmark JSONL using P3's compute_metrics logic.

Mirrors the P3 `council_policies.cli:compute_metrics` scoring path as
described in the session handoff:
  - _extract_answer priority: (1) `Final Answer:` regex, (2) last non-empty
    line for musique|quality|fever, (3) full stripped response.
  - FEVER       : label_accuracy (whole-word SUPPORT/REFUTE/NOT ENOUGH INFO).
  - HARDMATH    : `\\boxed{}` extraction on both pred AND gold, then EM.
  - MuSiQue     : exact_match_multi + token_f1_multi (best-of-golds), plus
    abstention credit: if pred == 'NOT_FOUND' and the row is unanswerable,
    score EM=1, F1=1 (mirrors P3's bonus). The answerable flag is recovered
    by joining router_dataset[index].original_id → bdsaglam/musique.id.
  - QuALITY     : exact_match_multi + token_f1_multi.

HumanEvalPlus pass@1 is scored separately via score_humaneval_pass1.py.

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/score_p4_results.py \\
        --results p4_gemma_lora_v2.jsonl
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from task_eval.extraction import extract_math_answer
from task_eval.scoring import (
    exact_match_multi,
    label_accuracy,
    math_exact_match,
    token_f1_multi,
)


_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LAST_LINE_SOURCES = {"MuSiQue", "QuALITY", "FEVER"}

_ORIGINAL_ID_KEYS = ("original_id", "id", "musique_id")
_ANSWERABLE_KEYS = ("answerable", "is_answerable")

# Abstention sentinels the MuSiQue-style template may produce. The P4
# `_MUSIQUE_CONSTRAINED_PROMPT` instructs the model to say
# "NOT PRESENT IN CONTEXT" rather than "NOT_FOUND", so matching only the
# latter would never credit abstention even on a perfectly-behaved model.
_ABSTAIN_SENTINELS = frozenset({
    "NOT_FOUND",
    "NOT FOUND",
    "NOT PRESENT IN CONTEXT",
    "NOT PRESENT",
    "UNANSWERABLE",
    "CANNOT BE ANSWERED",
    "CANNOT ANSWER",
    "NO ANSWER",
    "I DON'T KNOW",
})


def _is_abstention(text: str) -> bool:
    """Case-insensitive abstention sentinel match on stripped text."""
    return (text or "").strip().rstrip(".").upper() in _ABSTAIN_SENTINELS


def _extract_answer(response: str, source: str) -> str:
    """P3-style priority extraction. Preserves NOT_FOUND sentinel."""
    response = (response or "").strip()
    if not response:
        return ""
    if response == "NOT_FOUND":
        return "NOT_FOUND"

    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        return matches[-1].strip().rstrip(".")

    if source in _LAST_LINE_SOURCES:
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped.rstrip(".")

    return response


def _gold_as_list(gold):
    """Normalize a gold_answer field to list[str].

    Supports three shapes seen across the benchmark datasets:
      - str: wrap in a list.
      - list[str]: pass through.
      - list[dict]: QuALITY on router_dataset-2 stores each reference as
        {"text": "...", "tokens": [...]}. Pull out the text field; without
        this, scoring compares predictions to the dict's Python repr and
        reports EM=0 even on perfect matches.
    """
    if gold is None:
        return []
    if isinstance(gold, list):
        items: list[str] = []
        for g in gold:
            if not g:
                continue
            if isinstance(g, dict):
                text = g.get("text")
                if text:
                    items.append(str(text))
            else:
                items.append(str(g))
        return items
    if isinstance(gold, dict):
        text = gold.get("text")
        return [str(text)] if text else []
    return [str(gold)]


def _first_present(mapping, keys):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def build_musique_answerable_map(
    router_source: str, router_split: str,
    musique_source: str, musique_split: str,
) -> dict[int, bool]:
    """Map router_dataset row index -> answerable bool.

    P3's MuSiQue abstention credit requires the upstream `answerable` flag
    that router_dataset strips. We rejoin: router_dataset[i].original_id
    matches the id on bdsaglam/musique, whose rows carry `answerable`.
    Returns {} and prints a warning if the join cannot be built (lets
    scoring proceed without the bonus instead of failing).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[abstention] datasets package unavailable; skipping.")
        return {}
    try:
        router = load_dataset(router_source, split=router_split)
        musique = load_dataset(musique_source, split=musique_split)
    except Exception as exc:
        print(f"[abstention] dataset load failed ({exc}); skipping.")
        return {}

    by_mid: dict[str, bool] = {}
    for row in musique:
        mid = _first_present(row, _ORIGINAL_ID_KEYS)
        ans = _first_present(row, _ANSWERABLE_KEYS)
        if mid is None or ans is None:
            continue
        by_mid[str(mid)] = bool(ans)

    if not by_mid:
        print(
            f"[abstention] {musique_source}:{musique_split} has no "
            f"answerable flags on the expected keys {_ANSWERABLE_KEYS}; "
            f"skipping."
        )
        return {}

    out: dict[int, bool] = {}
    for i, row in enumerate(router):
        if (row.get("source_dataset") or "") != "MuSiQue":
            continue
        rid = _first_present(row, _ORIGINAL_ID_KEYS)
        if rid is None:
            continue
        rid_s = str(rid)
        if rid_s in by_mid:
            out[i] = by_mid[rid_s]

    print(
        f"[abstention] built answerable map for {len(out)} MuSiQue rows "
        f"(unanswerable: {sum(1 for v in out.values() if not v)})."
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--router-source", default="task-aware-llm-council/router_dataset",
    )
    p.add_argument("--router-split", default="validation")
    p.add_argument("--musique-source", default="bdsaglam/musique")
    p.add_argument("--musique-split", default="validation")
    p.add_argument(
        "--no-abstention", action="store_true",
        help="Skip MuSiQue abstention credit (default: enabled).",
    )
    args = p.parse_args()

    per_source = defaultdict(
        lambda: {"em": [], "f1": [], "acc": [], "n": 0, "errors": 0,
                 "abstain_credited": 0}
    )

    answerable_map: dict[int, bool] = {}
    if not args.no_abstention:
        answerable_map = build_musique_answerable_map(
            args.router_source, args.router_split,
            args.musique_source, args.musique_split,
        )

    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = row.get("source_dataset") or "UNKNOWN"
            if source == "HumanEvalPlus":
                # Scored separately via score_humaneval_pass1.py (execution).
                continue
            per_source[source]["n"] += 1
            if row.get("error"):
                per_source[source]["errors"] += 1
                continue

            pred_raw = row.get("p4_answer") or ""

            if source == "HARDMATH":
                pred = extract_math_answer(pred_raw)
            else:
                pred = _extract_answer(pred_raw, source)

            gold_field = "gold_label" if source == "FEVER" else "gold_answer"
            gold = _gold_as_list(row.get(gold_field))
            if not gold:
                continue

            if source == "FEVER":
                per_source[source]["acc"].append(label_accuracy(pred, gold[0]))
            elif source == "HARDMATH":
                gold_extracted = extract_math_answer(gold[0])
                per_source[source]["acc"].append(
                    math_exact_match(pred, gold_extracted)
                )
            else:
                # P3 MuSiQue abstention bonus: correct NOT_FOUND on an
                # unanswerable row scores EM=1, F1=1.
                if (
                    source == "MuSiQue"
                    and _is_abstention(pred)
                    and answerable_map.get(row.get("index")) is False
                ):
                    per_source[source]["em"].append(1.0)
                    per_source[source]["f1"].append(1.0)
                    per_source[source]["abstain_credited"] += 1
                else:
                    per_source[source]["em"].append(exact_match_multi(pred, gold))
                    per_source[source]["f1"].append(token_f1_multi(pred, gold))

    print(f"{'source':<20} {'n':>4} {'err':>4} {'EM':>8} {'F1':>8} {'Acc':>8} {'abst':>5}")
    print("-" * 66)
    for source, stats in sorted(per_source.items()):
        n = stats["n"]
        em = sum(stats["em"]) / len(stats["em"]) if stats["em"] else None
        f1 = sum(stats["f1"]) / len(stats["f1"]) if stats["f1"] else None
        acc = sum(stats["acc"]) / len(stats["acc"]) if stats["acc"] else None
        em_s = f"{em:>8.3f}" if em is not None else f"{'-':>8}"
        f1_s = f"{f1:>8.3f}" if f1 is not None else f"{'-':>8}"
        acc_s = f"{acc:>8.3f}" if acc is not None else f"{'-':>8}"
        abst = stats["abstain_credited"]
        print(f"{source:<20} {n:>4} {stats['errors']:>4} {em_s} {f1_s} {acc_s} {abst:>5}")


if __name__ == "__main__":
    main()
