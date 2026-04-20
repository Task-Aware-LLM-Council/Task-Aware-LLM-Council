"""Probe trained joint decomposer+router on mini_test split.

Prints 3 example predictions + pred/gold role distributions across 20 rows.
Tells us whether the model learned the 3-role vocabulary at all.

Usage (from repo root, on CARC):
    uv run --package training python scratch/probe_decomposer_router.py
"""
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from council_policies.router_featurize import featurize
from council_policies.seq2seq_decomposer_router import (
    INPUT_PREFIX,
    parse_targets,
    serialize_targets,
)

MODEL_DIR = "artifacts/decomposer_router_model"
PARQUET = "artifacts/decomposer_router_dataset/parquet/mini_test.parquet"
N_ROWS = 20
N_SHOW = 3

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).eval()
ds = load_dataset("parquet", data_files={"x": PARQUET})["x"]

pred_roles: Counter = Counter()
gold_roles: Counter = Counter()

for i in range(min(N_ROWS, len(ds))):
    row = ds[i]
    prompt = INPUT_PREFIX + featurize(
        row["question"] or "",
        row["context"] or "",
    )
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=256,
            num_beams=2,
            decoder_start_token_id=0,
        )
    pred_text = tok.decode(out[0], skip_special_tokens=True)
    gold_text = serialize_targets(row["targets"])

    for r in parse_targets(pred_text):
        pred_roles[r["role"]] += 1
    for r in parse_targets(gold_text):
        gold_roles[r["role"]] += 1

    if i < N_SHOW:
        print(f"--- row {i} ---")
        print("PRED:", repr(pred_text))
        print("GOLD:", repr(gold_text))

print("pred role dist:", pred_roles)
print("gold role dist:", gold_roles)
