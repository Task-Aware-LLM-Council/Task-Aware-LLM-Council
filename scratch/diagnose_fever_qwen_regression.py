"""Diagnose why force-routing FEVER to fact_general (Qwen-14B) crashed
FEVER accuracy from 0.533 to 0.017.

Three candidate hypotheses:

A. FORMAT:  Qwen-14B ignores the template's 'Answer strictly with one of
            these three labels' constraint and buries the verdict in prose.
            `label_accuracy` needs a whole-word SUPPORTS/REFUTES/NEI and
            misses it.

B. TRUNCATION: max_tokens budget cuts the response before the final label.
               Signal: tails that end mid-sentence or in a reasoning step.

C. ROLE ALIAS: 'fact_general' doesn't actually map to Qwen-14B — maybe
               resolves to a different model. Checked via the orchestrator
               config.

Usage:
    uv run --package task_eval python scratch/diagnose_fever_qwen_regression.py \\
        --results p4_gemma_lora_v2_rd2_val_oracle.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


_FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def _fever_tokens_in(text: str) -> list[str]:
    upper = (text or "").upper()
    return [lbl for lbl in _FEVER_LABELS if re.search(rf"\b{lbl}\b", upper)]


def _ends_mid_sentence(text: str) -> bool:
    """Heuristic for truncation: last non-whitespace char isn't terminal
    punctuation and the tail doesn't look like a label."""
    t = (text or "").rstrip()
    if not t:
        return True
    last = t[-1]
    return last not in ".!?)]>}\"'" and not t.rstrip(".").upper().endswith(
        tuple(_FEVER_LABELS)
    )


def hypothesis_c_config_check() -> None:
    print("=== [H-C] role-alias resolution ===")
    try:
        from model_orchestration import build_default_local_vllm_orchestrator_config
    except ImportError as exc:
        print(f"  skipped: couldn't import orchestrator ({exc})")
        return
    cfg = build_default_local_vllm_orchestrator_config()
    for m in cfg.models:
        aliases = tuple(m.aliases)
        note = "  <-- expected to host fact_general" if "fact_general" in aliases else ""
        print(f"  role={m.role:<12} model={m.model}  aliases={aliases}{note}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--sample", type=int, default=5,
                   help="rows per failure bucket to print")
    p.add_argument("--tail-chars", type=int, default=300,
                   help="trailing chars of pred to print per sample")
    args = p.parse_args()

    rows = []
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("source_dataset") == "FEVER":
                rows.append(r)

    if not rows:
        print(f"No FEVER rows in {args.results}")
        return

    print(f"=== FEVER diagnostic ({len(rows)} rows) ===\n")

    # --- Hypothesis C: alias config ---
    hypothesis_c_config_check()

    # --- Hypothesis A: format — does pred contain a label at all? ---
    labels_counter: Counter = Counter()
    no_label_rows, mid_sentence_rows, labeled_rows = [], [], []
    correct_rows: list[dict] = []
    routes_seen: Counter[str] = Counter()

    for r in rows:
        pred = r.get("p4_answer") or ""
        gold = ((r.get("gold_label") or r.get("gold_answer") or "")
                .strip().upper())
        tokens = _fever_tokens_in(pred)
        labels_counter[len(tokens)] += 1

        route = r.get("predicted_route") or []
        if isinstance(route, list):
            key = "+".join(route) if route else "(none)"
        else:
            key = str(route)
        routes_seen[key] += 1

        if not tokens:
            no_label_rows.append(r)
            if _ends_mid_sentence(pred):
                mid_sentence_rows.append(r)
        else:
            labeled_rows.append((r, tokens, gold))
            if gold in tokens:
                correct_rows.append(r)

    print("[H-A] format — label tokens per pred response:")
    for k in sorted(labels_counter):
        print(f"  {k} label token(s) in pred: {labels_counter[k]} rows")
    print(f"  no label at all: {len(no_label_rows)} rows")
    print(f"  gold label appears somewhere in pred: {len(correct_rows)} rows\n")

    print("[H-B] truncation proxy — tails that look cut mid-sentence:")
    print(f"  {len(mid_sentence_rows)}/{len(no_label_rows)} "
          f"no-label rows end mid-sentence (suggests truncation)\n")

    print("routes seen on FEVER:")
    for k, v in routes_seen.most_common():
        print(f"  {k}: {v}")
    print()

    def _show(bucket_name: str, bucket, n: int) -> None:
        if not bucket:
            return
        print(f"[sample: {bucket_name}] up to {n} rows")
        for r in bucket[:n]:
            pred = r.get("p4_answer") or ""
            gold = r.get("gold_label") or r.get("gold_answer")
            print(f"  idx={r.get('index')}  gold={gold}  route={r.get('predicted_route')}")
            tail = pred[-args.tail_chars:].replace("\n", " / ")
            print(f"    tail: ...{tail}")
            print()

    _show("no_label_mid_sentence", mid_sentence_rows, args.sample)
    _show("no_label_not_truncated",
          [r for r in no_label_rows if r not in mid_sentence_rows],
          args.sample)
    # Wrong labels chosen despite containing label token(s)
    wrong_labeled = [
        r for r, tokens, gold in labeled_rows if gold not in tokens
    ]
    _show("wrong_label_picked", wrong_labeled, args.sample)

    # --- Verdict ---
    print("=== verdict ===")
    if len(no_label_rows) > len(rows) * 0.5:
        if len(mid_sentence_rows) > len(no_label_rows) * 0.5:
            print("H-B likely: most no-label rows end mid-sentence → "
                  "response is being truncated. Bump max_tokens, or add a "
                  "stronger stop sequence + tighter prompt.")
        else:
            print("H-A likely: Qwen-14B is ignoring the three-label "
                  "constraint and writing prose that doesn't contain a "
                  "clean label token. Tighten the FEVER template to forbid "
                  "prose, e.g. 'Output ONLY one of SUPPORTS | REFUTES | "
                  "NOT ENOUGH INFO on a single line.'")
    else:
        print("labels are present but extraction or pred-vs-gold match "
              "fails. Re-inspect `wrong_label_picked` samples above — "
              "could be model choosing the wrong label, or the scorer "
              "extractor dropping it.")


if __name__ == "__main__":
    main()
