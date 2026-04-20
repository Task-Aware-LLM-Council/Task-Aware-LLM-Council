"""Quick check that FEVER rows are wrapped as questions, not bare claims.

Prints the first FEVER question from dev.parquet. Expected:
    'Is it true that <claim>?'
If it prints a bare claim, the `load_fever.py` edit didn't save.
"""
import pyarrow.parquet as pq

t = pq.read_table("data/router_dataset/dev.parquet").to_pylist()
for r in t:
    if r["source_dataset"] == "FEVER":
        print("FEVER Q:", r["question"][:200])
        break
else:
    print("no FEVER rows in dev split")
