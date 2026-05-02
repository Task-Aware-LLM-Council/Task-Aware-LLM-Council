# P4: Learned Task-Aware Routing — Report and Reproduction Guide

## 1. Overview

P4 extends the council framework introduced in P3 (rule-based routing) by
replacing the hand-written keyword classifier with a learned joint
decomposer-router. A single Seq2Seq model takes a prompt and emits both
(a) a decomposition of the prompt into subtasks and (b) a role assignment
per subtask drawn from the three-role action vocabulary
`{qa_reasoning, math_code, fact_general}`. Specialists are dispatched per
role; a synthesizer composes the final answer when more than one
specialist is involved.

The goal of P4 was to match or exceed P3's reported accuracy on the same
evaluation mix while replacing the brittle keyword classifier with a
trainable component. Final results on `router_dataset:test` (80 rows,
the same eval mix used by P3) match P3 on most datasets and exceed it on
HARDMATH.

| Source        | P3    | P4    |
|---------------|-------|-------|
| FEVER         | 0.89  | 0.875 |
| HARDMATH      | 0.69  | 0.875 |
| MuSiQue (EM)  | 0.88  | 0.750 |
| QuALITY (F1)  | —     | 0.731 |
| HumanEval+    | 0.94  | 0.875 |

---

## 2. Architecture

### Components

- **Joint decomposer-router** — Seq2Seq model
  (`council_policies.p4.seq2seq_decomposer_router.Seq2SeqDecomposerRouter`)
  loaded from the gemma-lora-v2 checkpoint. Takes the prompt and emits a
  JSON array of `{role, subtask}` objects.
- **Specialists** — three role-keyed vLLM servers:

  | Role            | Model                                            | Quantization |
  |-----------------|--------------------------------------------------|--------------|
  | `qa_reasoning`  | Gemma-2-9B                                       | 8-bit        |
  | `math_code`     | DeepSeek-R1-Distill-Qwen-7B-AWQ                  | 8-bit        |
  | `fact_general`  | Qwen-14B                                         | 4-bit AWQ    |

- **Synthesizer** — Gemma-2-9B (8-bit), invoked only when
  `len(specialist_runs) > 1`. Produces the final answer by composing
  ordered specialist outputs.
- **Council policy** —
  `council_policies.p4.policy.LearnedRouterPolicy`. Adapter-shaped
  (`plan()` / `finalize()`) so it integrates with
  `CouncilBenchmarkRunner`'s two-phase specialist→synthesizer GPU
  lifecycle.

### Two-phase GPU lifecycle

The 44 GB A40 cannot hold all specialists plus the synthesizer
co-resident. The runner enforces a phase swap:

```text
Phase 1 — specialist dispatch
  open specialist orchestrator
  → run plan() across all policies in batch
  → execute specialist requests in parallel
  close specialist orchestrator (frees specialist GPU memory)

Phase 2 — synthesis (only if any policy needs it)
  open synthesizer orchestrator
  → run finalize() across all policies that have len(runs) > 1
  close synthesizer orchestrator
```

Single-run prompts short-circuit in `finalize()` and never load the
synthesizer.

---

## 3. Repository layout

```text
packages/council_policies/src/council_policies/
├── p4/                                 # P4-specific
│   ├── __init__.py
│   ├── policy.py                       # LearnedRouterPolicy
│   └── seq2seq_decomposer_router.py    # Joint model serving class
├── decomposer.py                       # Shared Decomposer protocol
├── router.py                           # Shared Router/Subtask types
├── synthesis.py                        # Shared synthesize_ordered()
├── policy_runner.py                    # CouncilBenchmarkRunner
└── ...

packages/training/src/training/
├── build_decomposer_router_dataset.py  # Teacher-LLM gold labeling
├── train_decomposer_router.py          # Flan-T5 / gemma seq2seq trainer
└── train_decomposer_router_causal.py   # Causal-LM variant

packages/data_prep/src/data_prep/
├── build_router_dataset.py             # Source aggregation
└── load_fever.py                       # Per-dataset loaders

packages/model-orchestration/src/model_orchestration/
├── defaults.py                         # 3-role alias config
└── orchestrator.py                     # API-mode dispatch path

p4_benchmark_client.py                  # Top-level runner entry point

scratch/
├── score_p4_results.py                 # Main scorer
├── score_humaneval_pass1.py            # HumanEval+ pass@1
├── benchmark_stats.py                  # Latency / output / route_acc
├── count_routing_calls.py              # Component invocation counts
└── diagnose_*.py                       # Per-source failure-mode tools
```

---

## 4. Setup

This repo uses `uv` with workspace packages. On CARC, `uv` is installed
to `$HOME/.local/bin` and the inference container is provisioned through
the `apptainer` module. The shell environment for any P4 job needs:

```bash
module purge
module load apptainer
export PATH="$HOME/.local/bin:$PATH"   # for uv

# Let HF/torch use all requested CPU cores for the decomposer-router.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=true
```

Sync dependencies once from the repo root:

```bash
uv sync
```

The benchmark client downloads HF model weights on first run. To avoid
losing GPU walltime to cold-start downloads, pre-warm the cache on the
login node while the GPU job is queued:

```bash
huggingface-cli download google/gemma-2-9b-it
huggingface-cli download task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ
```

The trained joint decomposer-router LoRA adapter is checked into the
repo at `artifacts/decomposer_router_causal/adapter` and is loaded via
the `--peft-adapter` flag (see §5.2).

---

## 5. Reproducing results

### 5.1 Submit the job (preferred path)

The canonical run is via `sbatch`. The repo ships with
`p4_rd2_val.sbatch` for a 1×A40 validation run; the same template works
for the test split with two flag changes (`--split` and `--out`):

```bash
sbatch p4_rd2_val.sbatch
squeue -u $USER                 # check job state
tail -f p4_run_rd2_val.log      # follow output
```

The sbatch handles `module load apptainer`, the threading exports, and
the full `uv run` invocation (see §5.2 for the command body). Adjust
`--time`, `--mem`, and `--cpus-per-task` in the header if you change
walltime expectations.

### 5.2 Run inference interactively

For ad-hoc runs, allocate a GPU and attach to it:

```bash
salloc -p gpu --gres=gpu:a40:1 -t 4:00:00 \
    --mem=64G --cpus-per-task=16 --account=robinjia_1822

# If the allocation lands in a separate shell, attach:
srun --overlap --jobid=<jobid> --pty bash
```

Then run the same command body the sbatch uses:

```bash
module purge && module load apptainer
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=true

nohup uv run --package council_policies --with peft python p4_benchmark_client.py \
    --dataset task-aware-llm-council/router_dataset \
    --split test \
    --limit -1 \
    --chunk-size 100 \
    --peft-adapter artifacts/decomposer_router_causal/adapter \
    --out p4_old_test.jsonl \
    > p4_old_test.log 2>&1 &

tail -f p4_old_test.log
```

The benchmark client applies these defaults (the same flags used for
the report's results — no need to set them explicitly):

- `--musique-oracle` — injects `is_supporting`-filtered paragraphs from
  `bdsaglam/musique` into MuSiQue prompts. Disable with
  `--no-musique-oracle`.
- `--single-subtask-sources FEVER,HumanEvalPlus,HARDMATH` — bypasses
  decomposition; the prompt is its own single subtask.
- `--force-role-sources HumanEvalPlus:math_code,HARDMATH:math_code` —
  bypasses the learned router for those sources and dispatches directly
  to DeepSeek-R1.

Walltime: ~2 h for 80 rows, ~6 h for 300 rows on a single A40 with the
spec↔synth GPU swap.

### 5.3 Score the results

```bash
# Main scorer (FEVER, HARDMATH, MuSiQue, QuALITY)
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --package task_eval python scratch/score_p4_results.py \
    --results p4_old_test.jsonl

# HumanEval+ pass@1 (subprocess sandbox)
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --package task_eval python scratch/score_humaneval_pass1.py \
    --results p4_old_test.jsonl

# Per-source latency / output length / route accuracy
uv run python scratch/benchmark_stats.py --results p4_old_test.jsonl

# Component invocation counts (decomposer / router / synthesizer)
uv run python scratch/count_routing_calls.py --results p4_old_test.jsonl
```

### 5.4 Expected output

On `router_dataset:test` (16 rows per source, 80 total):

```text
source                  n  err       EM       F1      Acc  abst
FEVER                  16    0        -        -    0.875     0
HARDMATH               16    0        -        -    0.875     0
MuSiQue                16    0    0.750    0.760        -     6
QuALITY                12    0    0.333    0.731        -     0
HumanEvalPlus pass@1: 14/16 = 0.875
```

---

## 6. Methodology decisions

The first end-to-end P4 run scored ~0.50 average. Reaching P3 parity
(~0.82 average) required several methodology fixes — none of them changed
the trained model or its weights. Each fix is documented below alongside
the diagnostic that motivated it.

### 6.1 Force-routing for math/code (HARDMATH, HumanEvalPlus)

**Problem.** The joint router systematically routed code- and math-style
prompts to `qa_reasoning` (Gemma-2) instead of `math_code`
(DeepSeek-R1). HumanEval+ pass@1 was 0.083; HARDMATH accuracy was 0.717.

**Diagnostic.** `scratch/diagnose_humaneval_failures.py` bucketed pass@1
failures by mode. The dominant bucket was `SYNTAX_ERROR` from rows
routed to Gemma-2, which produces prose rather than parseable Python.

**Fix.** Added `force_role` and `force_single_subtask` metadata hooks in
`LearnedRouterPolicy.plan()`. The benchmark client sets
`force_role_sources=HARDMATH:math_code,HumanEvalPlus:math_code`. This
bypasses the learned router for those sources and dispatches directly to
DeepSeek-R1.

**Result.** HumanEval+ → 0.875, HARDMATH → 0.875.

### 6.2 Oracle context injection (MuSiQue)

**Problem.** Initial MuSiQue accuracy: 0.667 EM with 24/50 correct
abstentions on unanswerable rows.

**Fix.** The benchmark client now joins each MuSiQue prompt against
`bdsaglam/musique` on `original_id`, retrieves the supporting paragraphs
(filtered by `is_supporting=True`), and injects them as oracle context.
The scorer also reads the `answerable` flag and credits valid abstentions
on unanswerable rows.

**Result.** 0.750 EM / 0.760 F1, with 44/50 correct abstentions.

### 6.3 FEVER label-extraction style (scorer fix)

**Problem.** Force-routing FEVER to `fact_general` (Qwen-14B) crashed
accuracy from 0.533 to 0.017 — but Qwen was emitting correct labels. The
scorer was missing them.

**Diagnostic.** `scratch/diagnose_fever_qwen_regression.py` confirmed the
tokens (`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`) were present in 99%
of predictions. Qwen places the verdict *first*, then explains; the
scorer was extracting only the last line.

**Fix.** Replaced the per-line scan with
`task_eval.extraction.extract_fever_label`, a full-response label
detector matching P3's evaluation methodology.

**Result.** FEVER recovered to 0.875.

### 6.4 QuALITY gold-format parsing (scorer fix)

**Problem.** QuALITY EM dropped to 0.000 on `router_dataset-2`.

**Diagnostic.** `scratch/inspect_gold_format.py` revealed gold labels in
`router_dataset-2` are stored as a `str` repr of a list of dicts, not as
real structured data.

**Fix.** Added an `ast.literal_eval` fallback in
`scratch/score_p4_results.py::_gold_as_list`.

**Result.** QuALITY F1 recovered to 0.731.

---

## 7. Infrastructure notes

Reproducing this work on CARC required several non-obvious workarounds:

- **vLLM KV-cache on A40 (46 GB).** The minimum-block floor (~512 blocks)
  combined with Gemma-2-9B's sliding-window attention (10 GB+) makes
  single-GPU co-residency of all specialists infeasible. The current
  pipeline runs specialists and synthesizer in sequential phases on one
  GPU (P3 inherits the same constraint).
- **`apptainer --cleanenv` strips `CUDA_VISIBLE_DEVICES`.** Pass it
  explicitly with `--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES`.
  Otherwise the container sees all GPUs and ignores SLURM's allocation.
- **Pre-warming the HF cache on the login node** while the GPU queue is
  pending turns vLLM's cold start into pure CUDA init once the job
  lands.

SLURM batch templates for various GPU topologies live at the repo root:
`p4_rd2_val.sbatch`, `p4_rd2_val_{1,2}xa{40,100}.sbatch`.

---

## 8. Training the joint model (optional)

The shipped `gemma-lora-v2` checkpoint is sufficient to reproduce all
results above. To retrain from scratch:

```bash
# 1. Build gold labels via teacher LLM (e.g. GPT-4o)
uv run --package training python -m training.build_decomposer_router_dataset \
    --gold data/decomposer_router_gold.jsonl \
    --out data/decomposer_router_pairs.jsonl \
    --teacher-provider openai \
    --teacher-model gpt-4o

# 2. Train (Flan-T5 seq2seq variant)
uv run --package training python -m training.train_decomposer_router \
    --train-pairs data/decomposer_router_pairs.jsonl \
    --output-dir checkpoints/decomposer_router_v2
```

The trainer uses `Seq2SeqTrainer` with `predict_with_generate=True` and
emits four metrics per epoch: `json_parseable`, `shape_match`,
`role_accuracy`, `subtask_rougeL`, plus a composite scalar for early
stopping. The `INPUT_PREFIX` is canonicalized in the serving module so
training and inference are byte-identical. A causal-LM variant is
available at `training.train_decomposer_router_causal`.

---

## 9. Key findings

1. **Learned routing underperforms rule-based routing on
   specialist-heavy sources.** The joint model systematically misroutes
   math and code prompts to the general reasoner. Closing the gap to P3
   required per-source `force_role` overrides — effectively a rule-based
   escape hatch layered on top of the learned router. This points to a
   training-data coverage gap (not enough math/code examples in the
   gold set) rather than a model-capacity issue.

2. **Context injection matters more than routing for retrieval-style
   benchmarks.** On MuSiQue, oracle paragraphs and abstention credit
   produced a larger accuracy gain than any router-side change.

3. **Scorer fidelity to P3 was the single largest source of apparent
   regression.** Several "P4 regressions" turned out to be scorer bugs:
   gold-format parsing (QuALITY), extraction style (FEVER), and
   markdown-wrap handling. Reproducing P3's reported numbers required
   reproducing P3's scorer conventions in
   `scratch/score_p4_results.py`.

---

## 10. Limitations and future work

- **Three-role vocabulary.** The current router emits one of three roles.
  Real-world dispatch likely needs more granular categories (e.g.
  separating short-form factual from long-form reasoning).
- **Force-role overrides are a rules-engine in disguise.** The HARDMATH
  and HumanEval+ overrides remove the learned router from the loop for
  those sources entirely. A retrained checkpoint with more code/math
  gold examples should make the overrides unnecessary.
- **MuSiQue oracle context inflates accuracy** relative to a true
  retrieval setting. The current 0.750 EM is the ceiling under perfect
  retrieval; a real retrieval pipeline would score lower.
- **Synthesis short-circuits when `len(runs)==1`**, meaning single-role
  prompts skip the synthesizer entirely. This is correct behavior, but
  it means synthesizer behavior is only stress-tested on multi-role
  prompts.

---

## Appendix A — Component invocation counts

`count_routing_calls.py` reports per-source decomposer / router /
synthesizer invocations. Run on your final results jsonl:

```bash
uv run python scratch/count_routing_calls.py --results p4_old_test.jsonl
```

Expected shape (force-routed sources show 0 router calls; force-single
sources show 0 decomposer calls):

```text
source              rows   decomposer       router  synthesizer
FEVER                 16            0           16            0
HARDMATH              16            0            0            0
HumanEvalPlus         16            0            0            0
MuSiQue               16           16          ~32          ~10
QuALITY               16           16          ~32           ~8
ALL                   80           32          ~80          ~18
```

Numbers marked `~` depend on per-row decomposition output and are exact
once you run the script on a specific results file.

---

## Appendix B — Data sources

| Component                 | Source                                               |
|---------------------------|------------------------------------------------------|
| Council eval mix (P3)     | `task-aware-llm-council/router_dataset:test`         |
| Council eval mix (P4 dev) | `task-aware-llm-council/router_dataset-2:validation` |
| Decomposer-router gold    | `data/decomposer_router_gold.jsonl`                  |
| MuSiQue oracle context    | `bdsaglam/musique` (filtered on `is_supporting`)     |
| HumanEval+ test cases     | `task-aware-llm-council/router_dataset` `unit_tests` |
