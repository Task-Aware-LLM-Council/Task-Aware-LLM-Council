"""Bypass build_router_dataset and call load_fever directly — tells us
whether the wrap-as-question edit is actually reaching the loader code.

Expected: every question starts with 'Is it true that'.
"""
from data_prep.load_fever import load_fever

recs = load_fever(3)
for r in recs:
    print("Q:", r["question"][:200])
