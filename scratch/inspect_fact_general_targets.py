"""Show a sample of `fact_general` teacher-labeled subtasks from training.

If the subtasks are wrapped in FEVER template ("Claim: X. Is this
SUPPORTS/REFUTES/NOT_ENOUGH_INFO...?"), the teacher baked in the leakage
and rebalancing alone won't help — we'd need to sanitize targets too.

If the subtasks look clean and question-shaped, the LoRA learned the
FEVER template from prompt *inputs* and rebalancing will fix it.
"""
import pyarrow.parquet as pq

BASE = "artifacts/decomposer_router_dataset/parquet"
t = pq.read_table(f"{BASE}/train.parquet").to_pylist()

fact_general_rows = 0
fever_wrapped = 0
examples = []
for row in t:
    for tgt in (row.get("targets") or []):
        if tgt["role"] == "fact_general":
            fact_general_rows += 1
            subtask = tgt["subtask"]
            is_fever = "SUPPORTS" in subtask or "REFUTES" in subtask
            if is_fever:
                fever_wrapped += 1
            if len(examples) < 15:
                examples.append((is_fever, subtask[:180]))
            break

print(f"fact_general subtasks: {fact_general_rows} ({fever_wrapped} FEVER-wrapped)")
print(f"wrapped fraction: {fever_wrapped / max(fact_general_rows, 1):.1%}")
print()
print("--- samples (mix of wrapped/unwrapped) ---")
for is_fever, s in examples:
    tag = "[FEVER]" if is_fever else "[CLEAN]"
    print(f"{tag} {s}")
