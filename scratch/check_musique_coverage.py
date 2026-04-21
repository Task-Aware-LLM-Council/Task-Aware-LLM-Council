"""Verify that every MuSiQue row in our router_dataset validation split has
an oracle-context entry in the cached map. If hits < needed, the P4 client
will silently fall back to the full-context concatenation for missing rows."""
import json

from datasets import load_dataset

m = json.load(open("artifacts/musique_oracle_map.json"))
ds = load_dataset("task-aware-llm-council/router_dataset", split="validation")
needed = [str(r["original_id"]) for r in ds if r["source_dataset"] == "MuSiQue"]
hits = sum(1 for k in needed if k in m)
print(f"MuSiQue val rows: {len(needed)}, oracle hits: {hits}")
if hits < len(needed):
    missing = [k for k in needed if k not in m]
    print("missing sample:", missing[:5])
