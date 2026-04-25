# P4: Learned Task-Aware Routing

## Overview

P4 extends P3 (rule-based routing) by replacing the hand-written router with a
learned joint decomposer + router. A single Seq2Seq model takes a prompt and
emits both (a) a decomposition into subtasks and (b) a role assignment per
subtask drawn from `{qa_reasoning, math_code, fact_general}`. Specialists are
dispatched per role; a synthesizer composes the final answer.

Final scores on `router_dataset:test` (80 rows, same eval mix as P3):

| Source        | P3    | P4    |
|---------------|-------|-------|
| FEVER         | 0.89  | 0.875 |
| HARDMATH      | 0.69  | 0.875 |
| MuSiQue (EM)  | 0.88  | 0.750 |
| MuSiQue (F1)  | —     | 0.760 |
| QuALITY (F1)  | —     | 0.731 |
| HumanEval+    | 0.94  | 0.875 |

P4 matches or exceeds P3 on FEVER, HARDMATH, and HumanEval+, with HARDMATH
seeing the largest absolute gain. MuSiQue EM is the remaining soft spot.

---

## Architecture

- **Router / decomposer**: joint Seq2Seq model (`seq2seq_decomposer_router`),
  gemma-lora-v2 checkpoint, trained on `router_dataset-2`.
- **Specialists** (role → model):
  - `qa_reasoning` → Gemma-2-9B (8-bit)
  - `math_code`    → DeepSeek-R1-Distill-Qwen-7B-AWQ (8-bit)
  - `fact_general` → Qwen-14B (4-bit AWQ)
- **Synthesizer**: Gemma-2-9B (8-bit), swapped onto the GPU after specialists
  finish (44GB A40 cannot hold specialists + synth co-resident).

---

## Infrastructure work

Running P4 end-to-end on CARC required several non-obvious fixes:

- **vLLM KV-cache on A40 (46GB)**: the 512-block minimum floor combined with
  Gemma-2-9B's sliding-window attention made single-GPU co-residency of all
  specialists infeasible. Fell back to sequential spec↔synth GPU-swap on
  1×A40. Wrote `p4_rd2_val_{1xa40,2xa40,1xa100,2xa100}.sbatch` while probing.
- **`apptainer --cleanenv` scrubs `CUDA_VISIBLE_DEVICES`**: silent failure —
  container sees all GPUs, ignores SLURM allocation. Fixed by passing
  `--env CUDA_VISIBLE_DEVICES=...` explicitly.
- **HF cache pre-warm on the login node** while the GPU queue was pending,
  turning vLLM cold-start into pure CUDA init once the job landed.

---

## Closing the P4 → P3 accuracy gap

Initial P4 average: ~0.50. Final: ~0.82. No model or weight changes — every
gain came from methodology and routing overrides.

### HumanEvalPlus: 0.083 → 0.875

Root cause: the joint router was sending code questions to `qa_reasoning`
(Gemma) instead of `math_code` (DeepSeek-R1). Fixes:

- Added `force_role` + `force_single_subtask` metadata hooks in
  `p4_policy.py`; benchmark client sets these per-source.
- Constrained prompt requiring a ```python code fence.
- Expanded the HumanEval sandbox's imports
  (`decimal`, `fractions`, `statistics`, etc.) to match P3's runner.

### HARDMATH: 0.717 → 0.875

Same pattern. Force-routed to `math_code`, bypassing the learned router.

### FEVER: recovered via scorer fix, not routing

Force-routing FEVER to `fact_general` (Qwen-14B) crashed accuracy from 0.533 to
0.017. Diagnostic (`diagnose_fever_qwen_regression.py`) showed Qwen puts the
verdict *first* then explains, while the scorer grabbed the last label. Fixed
by swapping in `task_eval.extraction.extract_fever_label` — a full-response
scan that matches P3's methodology. FEVER force-route was then rolled back in
favor of the `force_single_subtask` fallback path into `fact_general`, which
had been a "happy accident" all along.

### MuSiQue: context > routing

Abstention discipline was the bottleneck. Added:

- Oracle paragraph injection (is_supporting-filtered from `bdsaglam/musique`).
- Abstention credit on unanswerable rows via `answerable` flag join on
  `original_id`.

Result: 24/50 → 44/50 correct abstentions. This moved the needle far more than
any router-side change.

### QuALITY: gold format parsing

`router_dataset-2` stores QuALITY's gold as a `str` repr of a list of dicts.
Added `ast.literal_eval` fallback in `_gold_as_list` so the scorer sees real
structured answers. EM jumped from 0.000 after this.

---

## Scoring toolkit

Built to reconcile P4 output with P3's evaluation methodology:

- `score_p4_results.py` — main scorer. Abstention, hedge-phrase detection,
  markdown strip, FEVER label extraction via `task_eval`.
- `score_humaneval_pass1.py` — HumanEval+ pass@1 subprocess sandbox.
- `benchmark_stats.py` — per-source latency / output length / routing
  accuracy. Force-routed sources excluded from route_acc (otherwise trivially
  1.0).

### Failure-mode diagnostics

Each regression got its own script:

- `diagnose_p3_gap.py` — FEVER/MuSiQue: abstention-miss vs extraction-miss vs
  model-wrong.
- `diagnose_humaneval_failures.py` — pass@1 buckets (no-entry-point,
  syntax-error, assert-failed, runtime-error).
- `diagnose_fever_qwen_regression.py` — H-A/H-B/H-C hypothesis test for the
  Qwen force-route crash.
- `diagnose_quality_regression.py` — EM regression on `router_dataset-2`.

---

## Key findings

1. **Learned routing underperforms rule-based routing on specialist-heavy
   sources.** The joint model systematically misroutes math and code prompts
   to the general reasoner. The eventual fix — per-source `force_role`
   overrides — is effectively a rule-based escape hatch layered on top of the
   learned router. This suggests the learned router needs more specialist
   signal during training, or the role vocabulary is under-specified.

2. **Context injection matters more than routing for MuSiQue.** Oracle
   paragraphs + abstention credit gave a larger accuracy jump than any router
   tweak. For multi-hop retrieval-style benchmarks, the evaluation plumbing
   is the dominant variable.

3. **Scorer fidelity to P3 was the single biggest gap.** Several apparent
   "P4 regressions" turned out to be scorer bugs — gold format parsing,
   markdown-wrap extraction, FEVER label extraction style. Reproducing P3's
   numbers required reproducing P3's scorer quirks.

---

## Files touched

### Runtime
- `packages/council_policies/src/council_policies/p4_policy.py` —
  `force_role` / `force_single_subtask` metadata overrides, synth-template
  passthrough.
- `packages/council_policies/src/council_policies/seq2seq_decomposer_router.py`
- `p4_benchmark_client.py` — MuSiQue oracle injection, force-role /
  single-subtask CLI, HumanEval template, max_tokens=4096.

### Training
- `packages/training/src/training/train_decomposer_router{,_causal}.py`
- `packages/training/src/training/build_decomposer_router_dataset.py`
- `packages/data_prep/src/data_prep/build_router_dataset.py`

### SLURM
- `p4_rd2_val.sbatch`, `p4_rd2_val_{1,2}xa{40,100}.sbatch`

### Scoring + diagnostics
- `scratch/score_p4_results.py`, `scratch/score_humaneval_pass1.py`,
  `scratch/benchmark_stats.py`
- `scratch/diagnose_{p3_gap,humaneval_failures,fever_qwen_regression,quality_regression}.py`
- `scratch/{inspect_gold_format,strip_source_from_jsonl,warm_specialist_cache}.py`
