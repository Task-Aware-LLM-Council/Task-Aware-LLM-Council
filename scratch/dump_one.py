"""Dump one MuSiQue row from a P4 results JSONL by index.

Prints route, gold, the context the specialist saw, and the model's
answer — so you can tell oracle-injection failures (wrong paragraphs in
context) apart from model hallucinations (right context, wrong answer).
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: dump_one.py <jsonl> <index>")
        sys.exit(1)

    out, target = sys.argv[1], int(sys.argv[2])
    for line in open(out):
        r = json.loads(line)
        if r.get("source_dataset") != "MuSiQue" or r.get("index") != target:
            continue
        print("=== INDEX", target, "===")
        print("ROUTE:", r.get("predicted_route"), " synth:", r.get("synthesis_used"))
        print("GOLD :", r.get("gold_answer"))
        ctx = r.get("context") or ""
        print(f"\n---- context ({len(ctx)} chars) ----")
        print(ctx)
        print("\n---- p4_answer ----")
        print(r.get("p4_answer"))
        return

    print(f"index {target} not found among MuSiQue rows in {out}")


if __name__ == "__main__":
    main()
