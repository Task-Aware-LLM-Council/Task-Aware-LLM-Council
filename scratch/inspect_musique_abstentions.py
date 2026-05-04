"""For MuSiQue rows where the model abstained ("NOT PRESENT IN CONTEXT"),
report the routing path and synthesis state so we can tell whether the
hedging is concentrated in multi-specialist+synth rows (synthesizer
problem) or in single-specialist rows (specialist problem).

For answerable abstentions, also dumps the tail of `p4_answer` so we
can see the model's reasoning right before its abstention sentinel —
sometimes the answer is in the reasoning text and the model abstains
on commitment.

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/inspect_musique_abstentions.py \\
        --results p4_smoke.jsonl
"""
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


_ABSTAIN_SENTINELS = frozenset({
    "NOT_FOUND", "NOT FOUND", "NOT PRESENT IN CONTEXT", "NOT PRESENT",
    "UNANSWERABLE", "CANNOT BE ANSWERED", "CANNOT ANSWER", "NO ANSWER",
})


def _is_abstention(pred: str) -> bool:
    cleaned = (pred or "").strip().rstrip(".").upper()
    return cleaned in _ABSTAIN_SENTINELS


def _extract_final_answer(response: str) -> str:
    """Mirror score_p4_results._extract_answer's last-line behavior."""
    response = (response or "").strip()
    if not response:
        return ""
    for line in reversed(response.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped.rstrip(".")
    return response


def _gold_as_list(gold):
    if gold is None:
        return []
    if isinstance(gold, list):
        return [str(g.get("text", g)) if isinstance(g, dict) else str(g)
                for g in gold if g]
    s = str(gold).strip()
    if s.startswith("[{") or s.startswith('["'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return _gold_as_list(parsed)
        except (ValueError, SyntaxError):
            pass
    return [s]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--show-rows", type=int, default=10,
        help="Print full detail for the first N answerable abstentions.",
    )
    p.add_argument(
        "--gold-in-reasoning", action="store_true",
        help="Additionally check whether each gold token appears in the "
             "model's reasoning text — flags rows where synth had the "
             "answer but hedged.",
    )
    args = p.parse_args()

    # Bucket: (route_tuple, synth_used) -> [(idx, golds, full_pred), ...]
    by_path: dict[tuple, list] = {}
    n_abstain = 0
    n_total = 0
    samples: list[dict] = []

    with args.results.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("source_dataset") != "MuSiQue":
                continue
            if r.get("error"):
                continue
            n_total += 1

            pred_raw = r.get("p4_answer") or ""
            final = _extract_final_answer(pred_raw)
            if not _is_abstention(final):
                continue

            n_abstain += 1
            golds = _gold_as_list(r.get("gold_answer"))
            route = tuple(r.get("predicted_route") or [])
            synth = bool(r.get("synthesis_used"))
            key = (route, synth)
            by_path.setdefault(key, []).append({
                "index": r["index"],
                "golds": golds,
                "pred_raw": pred_raw,
                "final": final,
            })
            samples.append({
                "index": r["index"],
                "route": route,
                "synth": synth,
                "golds": golds,
                "pred_raw": pred_raw,
            })

    print(f"MuSiQue rows total (excluding errors): {n_total}")
    print(f"Abstentions (final answer matches sentinel): {n_abstain}")

    print("\n=== route × synthesis breakdown for abstentions ===")
    for (route, synth), rows in sorted(by_path.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(rows):4}  route={route}  synth_used={synth}")

    print(f"\n=== first {min(args.show_rows, len(samples))} abstention samples ===")
    for s in samples[:args.show_rows]:
        print(f"\n[index {s['index']}]  route={s['route']}  synth={s['synth']}")
        print(f"  golds: {[g[:80] for g in s['golds']]}")

        if args.gold_in_reasoning and s["golds"]:
            # Quick check: does the reasoning text mention any gold token?
            reasoning_lower = s["pred_raw"].lower()
            for gold in s["golds"]:
                gold_norm = gold.lower().strip().rstrip(".")
                if gold_norm and gold_norm in reasoning_lower:
                    print(f"  ⚠ reasoning text contains gold {gold!r} verbatim — "
                          f"synth had the answer but hedged.")
                    break

        # Show the last 500 chars of the response — where the model
        # transitions from reasoning to abstention.
        tail = s["pred_raw"][-600:].replace("\n", " ")
        print(f"  tail: ...{tail}")


if __name__ == "__main__":
    main()
