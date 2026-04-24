"""Dissect the P4 vs P3 gap on FEVER and MuSiQue.

For each source, buckets failing rows by failure mode:

FEVER failure modes:
  - MODEL_WRONG: prediction is a valid label (SUPPORTS/REFUTES/NOT ENOUGH INFO)
    but doesn't match gold — specialist chose the wrong label.
  - EXTRACTION_MISS: prediction contains the correct label substring but
    `label_accuracy` (whole-word match) doesn't catch it — fixable in
    scorer by widening the matcher or reprompting the specialist.
  - MODEL_VAGUE: prediction lacks any of the three label tokens —
    specialist didn't commit. Template fix or specialist quality issue.

MuSiQue failure modes:
  - MISSED_ABSTAIN: row is unanswerable (bdsaglam/musique.answerable==False)
    but specialist tried to answer. Would be +1 EM if model had abstained.
    Biggest source of gap to P3 on this dataset.
  - WRONG_ABSTAIN: row IS answerable but specialist abstained — costs a
    real answer.
  - ANSWER_WRONG: answerable row, specialist answered, but extraction
    picked wrong entity / answer is semantically wrong.

Usage:
    uv run --package task_eval python scratch/diagnose_p3_gap.py \\
        --results p4_gemma_lora_v2_rd2_val.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Mirror scoring logic — keep in sync with score_p4_results.py.
_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
_ABSTAIN_SENTINELS = {
    "NOT_FOUND", "NOT FOUND", "NOT PRESENT IN CONTEXT", "NOT PRESENT",
    "UNANSWERABLE", "CANNOT BE ANSWERED", "CANNOT ANSWER", "NO ANSWER",
    "I DON'T KNOW",
}


def _extract_answer(response: str, source: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""
    if response == "NOT_FOUND":
        return "NOT_FOUND"
    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        return matches[-1].strip().rstrip(".")
    if source in {"MuSiQue", "QuALITY", "FEVER"}:
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped.rstrip(".")
    return response


def _is_abstention(text: str) -> bool:
    return (text or "").strip().rstrip(".").upper() in _ABSTAIN_SENTINELS


def _fever_label_tokens_in(text: str) -> list[str]:
    upper = (text or "").upper()
    return [lbl for lbl in _FEVER_LABELS if re.search(rf"\b{lbl}\b", upper)]


def diagnose_fever(rows: list[dict]) -> None:
    print(f"=== FEVER ({len(rows)} rows) ===")
    bucket: Counter[str] = Counter()
    examples: dict[str, list[tuple[int, str, str]]] = {}
    for r in rows:
        pred_raw = r.get("p4_answer") or ""
        gold = (r.get("gold_label") or r.get("gold_answer") or "").strip().upper()
        labels_found = _fever_label_tokens_in(pred_raw)
        pred_extracted = _extract_answer(pred_raw, "FEVER").strip().upper()

        if pred_extracted == gold:
            bucket["CORRECT"] += 1
            continue
        if not labels_found:
            bucket["MODEL_VAGUE"] += 1
            cls = "MODEL_VAGUE"
        elif gold in labels_found and pred_extracted != gold:
            # The gold label appears in the response but extraction picked
            # a different one or no label at all.
            bucket["EXTRACTION_MISS"] += 1
            cls = "EXTRACTION_MISS"
        else:
            bucket["MODEL_WRONG"] += 1
            cls = "MODEL_WRONG"
        examples.setdefault(cls, []).append((r.get("index"), pred_raw, gold))

    for k, n in bucket.most_common():
        print(f"  {k}: {n}")
    for cls in ("EXTRACTION_MISS", "MODEL_VAGUE", "MODEL_WRONG"):
        for idx, pred, gold in (examples.get(cls) or [])[:2]:
            print(f"  [{cls}] idx={idx} gold={gold}")
            print(f"    pred tail: ...{pred[-240:].replace(chr(10), ' / ')}")
    print()


def diagnose_musique(rows: list[dict], answerable_map: dict[int, bool]) -> None:
    print(f"=== MuSiQue ({len(rows)} rows) ===")
    bucket: Counter[str] = Counter()
    examples: dict[str, list[tuple[int, str, object]]] = {}
    for r in rows:
        pred_raw = r.get("p4_answer") or ""
        pred = _extract_answer(pred_raw, "MuSiQue")
        gold = r.get("gold_answer")
        idx = r.get("index")
        is_answerable = answerable_map.get(idx)

        if is_answerable is False:
            if _is_abstention(pred):
                bucket["CORRECT_ABSTAIN"] += 1
            else:
                bucket["MISSED_ABSTAIN"] += 1
                examples.setdefault("MISSED_ABSTAIN", []).append((idx, pred_raw, gold))
        elif is_answerable is True:
            if _is_abstention(pred):
                bucket["WRONG_ABSTAIN"] += 1
                examples.setdefault("WRONG_ABSTAIN", []).append((idx, pred_raw, gold))
            elif pred.strip().lower() == (gold or "").strip().lower():
                bucket["CORRECT_ANSWER"] += 1
            else:
                bucket["ANSWER_WRONG"] += 1
                examples.setdefault("ANSWER_WRONG", []).append((idx, pred_raw, gold))
        else:
            bucket["UNKNOWN_ANSWERABLE"] += 1

    for k, n in bucket.most_common():
        print(f"  {k}: {n}")
    for cls in ("MISSED_ABSTAIN", "ANSWER_WRONG", "WRONG_ABSTAIN"):
        for idx, pred, gold in (examples.get(cls) or [])[:2]:
            print(f"  [{cls}] idx={idx} gold={gold}")
            print(f"    pred tail: ...{pred[-240:].replace(chr(10), ' / ')}")
    print()


def build_answerable_map(router_source: str, router_split: str,
                         musique_source: str, musique_split: str) -> dict[int, bool]:
    try:
        from datasets import load_dataset
    except ImportError:
        return {}
    try:
        router = load_dataset(router_source, split=router_split)
        musique = load_dataset(musique_source, split=musique_split)
    except Exception as exc:
        print(f"[warn] dataset load failed: {exc}")
        return {}
    _ID_KEYS = ("original_id", "id", "musique_id")
    _ANS_KEYS = ("answerable", "is_answerable")

    def _first(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    by_mid: dict[str, bool] = {}
    for row in musique:
        mid = _first(row, _ID_KEYS)
        ans = _first(row, _ANS_KEYS)
        if mid is not None and ans is not None:
            by_mid[str(mid)] = bool(ans)

    out: dict[int, bool] = {}
    for i, row in enumerate(router):
        if (row.get("source_dataset") or "") != "MuSiQue":
            continue
        rid = _first(row, _ID_KEYS)
        if rid is not None and str(rid) in by_mid:
            out[i] = by_mid[str(rid)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--router-source", default="task-aware-llm-council/router_dataset-2")
    p.add_argument("--router-split", default="validation")
    p.add_argument("--musique-source", default="bdsaglam/musique")
    p.add_argument("--musique-split", default="validation")
    args = p.parse_args()

    fever, musique = [], []
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source_dataset")
            if src == "FEVER":
                fever.append(row)
            elif src == "MuSiQue":
                musique.append(row)

    if fever:
        diagnose_fever(fever)

    if musique:
        amap = build_answerable_map(
            args.router_source, args.router_split,
            args.musique_source, args.musique_split,
        )
        diagnose_musique(musique, amap)


if __name__ == "__main__":
    main()
