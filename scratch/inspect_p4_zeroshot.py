"""Scan p4_gemma_zeroshot.jsonl — decide whether zero-shot Gemma is good enough.

Prints aggregate stats (role distribution, subtask count, errors, synth rate,
latency) plus 5 sample rows with question/route/answer/gold truncated for
readability. Run from repo root on CARC:

    python scratch/inspect_p4_zeroshot.py
"""
import json
from collections import Counter

rows = [json.loads(l) for l in open("p4_gemma_zeroshot.jsonl")]
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
            roles[s.get("role", "?")] += 1
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
