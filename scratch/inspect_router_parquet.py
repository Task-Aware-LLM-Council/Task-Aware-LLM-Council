"""Row counts + first-row peek for the teacher-labeled router dataset."""
import pyarrow.parquet as pq

BASE = "artifacts/decomposer_router_dataset/parquet"

for s in ["train", "dev", "mini_test", "gold_eval"]:
    t = pq.read_table(f"{BASE}/{s}.parquet")
    print(f"{s}: {t.num_rows} rows, cols={t.column_names}")

print("\n--- train[0] ---")
t = pq.read_table(f"{BASE}/train.parquet")
row = t.slice(0, 1).to_pydict()
for k, v in row.items():
    print(f"{k}: {str(v[0])[:200]}")
