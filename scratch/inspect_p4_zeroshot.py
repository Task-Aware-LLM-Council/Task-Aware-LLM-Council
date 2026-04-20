"""Scan a p4_*.jsonl — aggregate stats + 5 sample rows.

Usage:
    python scratch/inspect_p4_zeroshot.py                      # default: p4_gemma_zeroshot.jsonl
    python scratch/inspect_p4_zeroshot.py p4_gemma_lora.jsonl  # compare the LoRA run
"""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "p4_gemma_zeroshot.jsonl"
print(f"=== {path} ===")
rows = [json.loads(l) for l in open(path)]
print("total:", len(rows))

roles = Counter()
n_subtasks = Counter()
errors = 0
synth_used = 0
latencies = []
for r in rows:
    pr = r.get("predicted_route") or []
    n_subtasks[len(pr) if isinstance(pr, list) else 0] += 1
    if isinstance(pr, list):
        for s in pr:
            if isinstance(s, dict):
                roles[s.get("role", "?")] += 1
            else:
                roles[str(s)] += 1
    if r.get("error"):
        errors += 1
    if r.get("synthesis_used"):
        synth_used += 1
    if r.get("latency_ms"):
        latencies.append(r["latency_ms"])

print("role distribution:", dict(roles))
print("subtasks-per-prompt:", dict(n_subtasks))
print("errors:", errors)
print("synthesis_used:", synth_used)
if latencies:
    print(f"latency ms: mean={sum(latencies) / len(latencies):.0f} "
          f"max={max(latencies):.0f}")

print()
for i in [0, 3, 9, 16, 19]:
    if i >= len(rows):
        continue
    r = rows[i]
    print(f"--- row {i} ---")
    print("Q:", r["question"][:140])
    print("ROUTE:", r.get("predicted_route"))
    ans = (r.get("p4_answer") or "")[:240].replace("\n", " ")
    print("ANS:", ans)
    print("GOLD:", r.get("gold_answer"))
    print()
