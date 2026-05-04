# Task-Aware LLM Council

Monorepo for task-aware LLM benchmarking and council-style model policies. The
workspace includes reusable packages for provider access, benchmark execution,
task evaluation, local model orchestration, quantization, and policy runs.
Main steps to reproduce the results are mentioned here - [Steps To Reproduce Results](#steps-to-reproduce-results), but make sure to follow all the setup and prerequisite steps.

## Environment Setup

### Prerequisites

- Python `>=3.12`
- `uv`: https://docs.astral.sh/uv/getting-started/installation/
- For running on CARC, vLLM and Python apptainer images are required. Copy both of them from this location - `/project2/robinjia_1822/apptainer-compatible-images-dont-delete` to the root of repository.

### Install

Clone the repository, move into the project root, and sync every workspace
package:

```bash
uv sync --all-packages
```

Use `uv run` for commands so package entry points resolve from the workspace
environment:

```bash
uv run <command> --help
```

### Provider Credentials

Set only the keys needed for the provider you plan to use:

```bash
export OPENAI_API_KEY=<openai-key>
export OPENROUTER_API_KEY=<openrouter-key>
export HUGGINGFACE_API_KEY=<huggingface-key>
export NVIDIA_API_KEY=<nvidia-key>
```

`Local vLLM runs do not need an API key, but they do need local vLLM Apptainer images, as mentioned in `
[Prerequisites](#prerequisites)

## Package Overview

```text
packages/
|- common/                  Shared schemas, IDs, split helpers, and utilities
|- data_prep/               Dataset loaders and router dataset build/upload scripts
|- benchmarking_pipeline/   Prompt execution and prediction artifact persistence
|- benchmark_runner/        P1 benchmark suite runner and aggregate summaries
|- task_eval/               Dataset profiles, extraction, normalization, and scoring
|- llm_gateway/             OpenAI/OpenRouter/HF/local provider clients and vLLM runtime
|- model-orchestration/     Role-based model orchestration for council policies
|- council_policies/        Council policy runner
|- model-quantization/      Hugging Face model quantization and upload CLI
|- training/                Training-facing package scaffold
|- inference/               Inference package scaffold
```

Core flow:

```text
task_eval -> benchmarking_pipeline -> benchmark_runner
llm_gateway -> benchmarking_pipeline / model-orchestration
model-orchestration -> council_policies

```
# Steps To Reproduce Results
1. All these tests were done on CARC using a40, so make sure to get correct resources assigned.
`srun --pty -p gpu --gres=gpu:a40:1 --account=robinjia_1822 --time=04:00:00 --cpus-per-task=8 --mem=64G bash`
2. Run `module load apptainer` after resources are allocated. This is required to load apptainer runtime,
3. All the policies use these final models as specialists, so only these models should be used. 
Ignore any other models in the repository, since there are few other models that were part of initial pool of models from which specialists were chosen.
- "task-aware-llm-council/gemma-2-9b-it-GPTQ",
- "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
- "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2",

**Note** All the policies were run on single a40 GPU (since getting multiple GPUs was almost impossible), hence use a40 since it is the minimum spec GPU with enough vRAM to run 3 models and minimum CUDA Compute Capability to run vLLM.

## Single Model (P1) Benchmarking

P1 uses `benchmark_runner` to run model-dataset benchmark pairs and write
predictions, scores, per-pair summaries, and a flattened suite summary.

### Local vLLM Run

```bash
uv run benchmark-runner \
  --preset pilot \
  --provider vllm
```
`This will load 1 model, run all 5 datasets sequentially, and then unload and load second model (total 3 models). This whole process can take time, approx 1.5 hours/model so make sure to allocate resources accordingly. Or run this step-by-step using the following flags`

Run individual models:

```bash
uv run benchmark-runner \
  --preset pilot \
  --provider vllm \
  --models task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2 \
  --datasets musique fever \
```

Default P1 output root:

```text
/scratch1/$USER/results/benchmark_suite
```

Each run writes a suite directory containing `manifest.json`, prediction JSONL
files, score JSONL files, per-pair summaries, and `suite_metrics.json`.

`Once each run completes, suite_metrics.json is the file to check for summarized results. 
In case there was some failure during the run, intermediate results for each dataset are summarized in this location 
/scratch1/$USER/results/benchmark_suite/<run_name>/summaries`

### Slurm Wrapper (not fully tested)

Use the root wrapper when submitting a GPU job through Slurm:

```bash
./run_benchmark_job.sh \
  --model task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2 \
  --port 8000 \
  --partition gpu \
  --time 02:00:00 \
  --cpus-per-task 8 \
  --mem 64G
```

The wrapper syncs the workspace and runs:

```bash
uv run benchmark-runner \
  --preset pilot \
  --provider vllm \
  --models "$MODEL_NAME" \
  --local-launch-port "$PORT"
```

## Flat-Council (P2) Council Benchmarking

P2 uses `council_policies` to run the specialist/synthesizer council policy on
the mixed router dataset and score the synthesized answers.

`This will load all 3 specialists in one GPU, run the specialist phase, unload the specialists and then load synthesizer (single model). The whole process can take close to 1hr`

### Local vLLM Run

```bash
uv run council-policies \
  --policy p2 \
  --preset pilot \
  --provider vllm
```

Default P2 output root:

```text
/scratch1/$USER/results/council_policies
```

Each run writes `manifest.json`, prediction JSONL, score JSONL, summary files,
and `suite_metrics.json`.

`Once each run completes, suite_metrics.json is the file to check for summarized results. 
In case there was some failure during the run, intermediate results for each dataset are summarized in this location 
/scratch1/$USER/results/council_policies/<run_name>/summaries` In case you don't see the summary, best is to re-run the script.

## P3 Branch

P3 lives on a separate branch since there are multiple steps to run it.

```bash
git fetch origin
git checkout please_work
```

## P4 Branch

P4 (joint decomposer+router + quantized specialists + synthesizer) lives
on the `router-training` branch. End-to-end recipe for evaluating it on a
single A40 via Slurm follows.

### Switch to the P4 branch

```bash
git fetch origin
git checkout router-training
git pull --ff-only
uv sync --all-packages
```

Verify the P4 layout:

```bash
ls packages/council_policies/src/council_policies/p4/
# should show: policy.py, router.py, decomposer.py, synthesis.py,
#              policy_runner.py, types.py, ...
```

P4 specialists default to:

- `qa_reasoning` → `task-aware-llm-council/gemma-2-9b-it-GPTQ`
- `math_code` → `task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2`
- `fact_general` → `task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2`
- synthesizer → same DeepSeek-R1 AWQ as `math_code`

Joint decomposer+router runs in-process (CPU) on `google/gemma-2-2b-it`
with an optional LoRA adapter at `artifacts/decomposer_router_causal/adapter`.

### P4 environment setup

P4 runs inside an Apptainer-launched vLLM container managed by
`model-orchestration`. The login-node setup is:

```bash
module purge
module load apptainer

# uv lives under your home — make sure it is on PATH
export PATH="$HOME/.local/bin:$PATH"

# sanity-check the toolchain
which apptainer
which uv
```

> The vLLM image (`vllm-openai_latest.sif`) and HF cache live under
> `/scratch1/$USER/.cache`. The orchestrator binds that path into the
> container automatically — you do not need to bind-mount manually.

#### Pre-warm the HF cache (optional but strongly recommended)

The login node can download weights while the GPU allocation is still
queued, turning the eventual vLLM cold start into pure CUDA-init cost.
For P4 you want the specialist triad + the synthesizer + the decomposer
base on disk before submitting:

```bash
uv run --package council_policies python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "task-aware-llm-council/gemma-2-9b-it-GPTQ",
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2",
    "google/gemma-2-2b-it",
]:
    snapshot_download(repo)
PY
```

### Configure the run

The runner is `p4_benchmark_client.py` at the repo root. The two knobs
you will touch most are **how many rows to process** and **how often to
flush to disk**:

| flag | what it does | typical value |
| --- | --- | --- |
| `--dataset` | HF dataset for eval prompts | `task-aware-llm-council/router_dataset-2` |
| `--split` | dataset split | `validation` or `test` |
| `--limit` | cap rows; `-1` = run the whole split | `25` for smoke, `-1` for full eval |
| `--chunk-size` | flush + spec↔synth GPU swap every N prompts | `60`–`100` |
| `--out` | JSONL output path; resume reads from this | `p4_results.jsonl` |
| `--peft-adapter` | optional LoRA adapter on top of the base decomposer | `artifacts/decomposer_router_causal/adapter` |

#### Why `--chunk-size` matters

The 44 GB A40 cannot hold all 3 specialists *and* the synthesizer at the
same time. Each chunk pays one specialist↔synthesizer swap (≈ 8 min
cold start). Bigger chunks = fewer cold starts but a larger blast radius
if the job dies mid-chunk and the in-flight rows are lost. 60–100 is
the sweet spot.

Edit `p4_rd2_val.sbatch` (or copy it) and tune the bottom block:

```bash
uv run --package council_policies --with peft python p4_benchmark_client.py \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 100 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val.jsonl
```

### Smoke-test on the debug GPU first

Before burning a 2 h `gpu` allocation on a config you have not run
end-to-end yet, verify the pipeline on the **debug** partition with a
tiny `--limit`. This catches the cheap failures (missing image, wrong
adapter path, dataset typo, port collision) in 30 minutes of cheap wall
clock instead of waiting through the full queue.

#### Option A — interactive `salloc`

The fastest feedback loop. You get a shell on the GPU node and can
iterate on flags without resubmitting:

```bash
salloc --partition=debug --gres=gpu:a40:1 --cpus-per-task=8 \
       --mem=40G --time=00:30:00

# once the shell drops in:
module load apptainer
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi | head -5

uv run --package council_policies --with peft python p4_benchmark_client.py \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit 5 \
  --chunk-size 5 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_smoke.jsonl
```

`--limit 5 --chunk-size 5` runs exactly one specialist phase and one
synthesizer phase, which is what you want — it exercises the GPU swap
without paying for more than one of them. Tail the output and watch
for:

- decomposer loads (`HFCausalGenerate` log line),
- vLLM specialist server comes up on port 8000-8003,
- specialist phase completes,
- specialist vLLM exits cleanly,
- synthesizer vLLM comes up on port 8004,
- synthesizer phase completes,
- 5 lines land in `p4_smoke.jsonl`.

If any of those steps stall or error, fix it here — not in a queued
2 h job.

#### Option B — debug-partition sbatch loop

If `salloc` is queued and you'd rather batch-submit, copy
`p4_rd2_val.sbatch` to `p4_debug.sbatch` and change the header + the
run line:

```bash
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --job-name=p4_debug
#SBATCH --output=p4_debug.log

# ... (rest of the apptainer / PATH / threading setup unchanged) ...

uv run --package council_policies --with peft python p4_benchmark_client.py \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit 5 \
  --chunk-size 5 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_smoke.jsonl
```

Then drive iterations with the same resume property the full loop uses
— each rerun resumes against `p4_smoke.jsonl`, so if you grow `--limit`
across iterations the previous rows are reused:

```bash
#!/usr/bin/env bash
# loop_p4_debug.sh — keep resubmitting the debug job until it succeeds
set -euo pipefail
OUT=p4_smoke.jsonl
TARGET=5

while :; do
  done=$(wc -l <"$OUT" 2>/dev/null || echo 0)
  echo "[debug] $done / $TARGET rows done"
  if [[ "$done" -ge "$TARGET" ]]; then
    echo "[debug] smoke test complete — promote to full run."
    break
  fi
  jid=$(sbatch --parsable p4_debug.sbatch)
  echo "[debug] submitted job $jid; waiting…"
  while squeue -j "$jid" -h | grep -q .; do sleep 30; done
done
```

Once `p4_smoke.jsonl` has 5 healthy rows, delete it (or pick a
different `--out`) and move on to the full run below. **Never reuse the
smoke jsonl as the full-run output** — its rows came from `--limit 5`,
but the runner keys resume by `index` and would silently skip those
indices in the full split.

### Submit the full job

You have two partitions available. Pick one based on queue state right
now — the output JSONL is partition-agnostic so you can switch later if
you change your mind.

#### Option A — `gpu` partition (default, 2 h jobs)

The standard path. Submit and tail the log:

```bash
sbatch p4_rd2_val.sbatch
squeue -u $USER          # watch the queue
tail -f p4_run_rd2_val.log   # live log once the job starts
```

Each job is wall-time-capped at 2 h (see `#SBATCH --time=02:00:00`). On
a full split that is **not** enough to finish — which is the whole
point of the resume loop below.

#### Option B — `debug` partition (30 min jobs, congestion alternative)

If the `gpu` queue is congested but `debug` is open, you can run the
full eval entirely on debug compute. The walltime is 30 min instead of
2 h, so you submit ~4× as many jobs — but each one usually drops in
fast, and the resume loop below handles the multi-job pattern
transparently.

Copy `p4_debug.sbatch` (built in the smoke section above) to
`p4_debug_full.sbatch` and switch the run line back to a real `--limit`
and a real output path:

```bash
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --job-name=p4_debug_full
#SBATCH --output=p4_debug_full.log

# ... (apptainer / PATH / threading setup unchanged) ...

uv run --package council_policies --with peft python p4_benchmark_client.py \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 60 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val.jsonl
```

Submit and tail the same way as Option A:

```bash
sbatch p4_debug_full.sbatch
squeue -u $USER
tail -f p4_debug_full.log
```

Notes specific to running the full eval on debug compute:

- **Drop `--chunk-size` to 40–60.** A 2 h job at chunk-size 100 fits
  ~1 chunk per spec↔synth swap cycle. A 30 min job at chunk-size 100
  may not even finish *one* full chunk on slower rows, which means the
  job dies before its first JSONL flush and you make zero progress.
  60 is the safe ceiling; 40 if you're seeing late-chunk timeouts.
- **More cold-start tax.** Each submission pays the ~8-min spec↔synth
  swap. A 30 min job spends a meaningful fraction of its budget on
  cold start, so throughput per wall-clock-hour is lower than on `gpu`.
  The win is queue latency, not compute efficiency.
- **Mixing `gpu` and `debug` is fine.** The output JSONL is the only
  state, and resume-by-`index` doesn't care which partition wrote each
  row. You can switch partitions between resubmissions freely. Running
  one `gpu` and one `debug_full` job *concurrently* against the same
  `--out` is also possible but introduces a rare duplicate-row race
  (`_load_done_indices` is snapshotted once at startup, not refreshed
  mid-run) — fine if rare, run sequentially if you don't want to think
  about it.

### The resume loop (rerunning until done)

`p4_benchmark_client.py` is **idempotent against its own output file**.
On startup it reads `--out`, parses every JSONL line, and skips any
`index` it has already written:

```python
done_indices = _load_done_indices(args.out)
if done_indices:
    print(f"Resuming: {len(done_indices)} rows already in {args.out}, skipping.")
```

So the recovery pattern after a 2-hour timeout, OOM, or transient vLLM
crash is just **resubmit the same sbatch** — it picks up where the
previous run stopped, no flags to change.

#### Manual loop

```bash
sbatch p4_rd2_val.sbatch                       # job 1
# … wait for completion / timeout, check progress:
wc -l p4_gemma_lora_v2_rd2_val.jsonl           # rows done so far
sbatch p4_rd2_val.sbatch                       # job 2 — resumes
# repeat until line-count matches the split size
```

#### Automated loop

If you have the dataset row count handy (call it `TARGET=2500`), drive
the resubmissions from a small bash wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
OUT=p4_gemma_lora_v2_rd2_val.jsonl
TARGET=2500

while :; do
  done=$(wc -l <"$OUT" 2>/dev/null || echo 0)
  echo "[loop] $done / $TARGET rows done"
  if [[ "$done" -ge "$TARGET" ]]; then
    echo "[loop] complete."
    break
  fi
  jid=$(sbatch --parsable p4_rd2_val.sbatch)
  echo "[loop] submitted job $jid; waiting…"
  # block until that job leaves the queue
  while squeue -j "$jid" -h | grep -q .; do sleep 60; done
done
```

Save as `loop_p4.sh`, `chmod +x loop_p4.sh`, run on the login node.
Each iteration submits one 2 h job, waits for it, then re-checks the
JSONL line count — and exits cleanly once the file is full.

### Getting the results

The runner produces one JSONL with one row per prompt; everything below
operates on that file. Path examples assume
`OUT=p4_gemma_lora_v2_rd2_val.jsonl`.

#### Output schema

Each line is a flat JSON object:

| field | type | meaning |
| --- | --- | --- |
| `index` | int | row index in the source split (the resume key) |
| `source_dataset` | str | one of `MuSiQue`, `FEVER`, `QuALITY`, `HARDMATH`, `HumanEvalPlus` |
| `question` | str | prompt the policy actually saw (post-template) |
| `context` | str | retrieved/oracle context for QA-ish rows; `""` otherwise |
| `gold_answer` | str / list | reference answer(s); list-of-dicts on QuALITY (router_dataset-2 quirk) |
| `gold_label` | str | reference label, FEVER only |
| `force_single_subtask` | bool | runner skipped decomposition for this row |
| `force_role` | str / null | runner pinned the role (bypassed the learned router) |
| `p4_answer` | str | the synthesized response — what scoring extracts from |
| `predicted_route` | list[str] | the role(s) the router actually picked |
| `synthesis_used` | bool | true ⇔ more than one specialist run was fused |
| `error` | str / null | populated only if the chunk crashed mid-row |
| `latency_ms` | float | wall-time for the whole pipeline on this row |

#### Sanity-check one-liners

CARC login nodes don't ship `jq`; use Python stdlib instead:

```bash
OUT=p4_gemma_lora_v2_rd2_val.jsonl

# total rows
wc -l "$OUT"

# rows per source
python3 -c '
import json
from collections import Counter
c = Counter()
for line in open("'"$OUT"'"):
    c[json.loads(line)["source_dataset"]] += 1
for k, n in c.most_common():
    print(f"{n:5}  {k}")
'

# error count
python3 -c '
import json
n = sum(1 for line in open("'"$OUT"'") if json.loads(line).get("error"))
print(f"errors: {n}")
'
```

#### Score the non-code datasets

`scratch/score_p4_results.py` mirrors P3's grading methodology so P3↔P4
comparisons are apples-to-apples. It scores MuSiQue, QuALITY, FEVER,
and HARDMATH from a single JSONL and skips HumanEvalPlus (executed
separately):

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --package task_eval python scratch/score_p4_results.py \
  --results "$OUT"
```

Output is a single table:

```text
source                  n  err       EM       F1      Acc  abst
------------------------------------------------------------------
FEVER                 500    2        -        -    0.612     0
HARDMATH              500    0        -        -    0.388     0
MuSiQue               500    1    0.482    0.601        -    37
QuALITY               500    0    0.541    0.694        -     0
```

How to read it:

- **EM / F1** — exact-match and token-F1 (best-of-golds for multi-ref
  datasets), reported on extracted final answers.
- **Acc** — label or boxed-numeric accuracy for FEVER and HARDMATH.
- **abst** — MuSiQue rows where the model correctly said "NOT PRESENT
  IN CONTEXT" on an unanswerable row and got the abstention bonus
  (EM=F1=1). Pass `--no-abstention` to disable; without it you under-
  count P3-equivalent score.
- **err** — rows that crashed mid-chunk. If non-zero, rerun the sbatch
  to fill them in (resume by `index` is automatic).

#### Score HumanEvalPlus pass@1

Code execution lives in a separate script because it runs candidate
code in a subprocess with a timeout (the rest of the metrics are
pure-text):

```bash
uv run --package task_eval python scratch/score_humaneval_pass1.py \
  --results "$OUT" \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --timeout 10
```

Prints `pass@1 = K/N = X.XXX` and the first 15 failing `entry_point`
names. **Sandbox warning:** the script imports and executes the
model's generated Python in a subprocess. Only point it at trusted
public test suites (HumanEval is fine) — never at adversarial code.

#### Routing analysis (optional)

Two scripts dig into where the router spent its calls vs. how well it
classified:

```bash
# how many decomposer / router / synthesizer calls hit per source
uv run python scratch/count_routing_calls.py --results "$OUT"

# routing accuracy + latency / output-length per source
uv run python scratch/benchmark_stats.py --results "$OUT"
```

`benchmark_stats.py`'s "routing accuracy" column is the one to watch —
it tells you how often the learned router picked a *plausible* role
for each source (vs. the trivially-100% `force_role` rows which are
synthetic).

#### Diagnose specific failure modes

When a metric drops, the `scratch/diagnose_*.py` and `scratch/inspect_*.py`
scripts dump the rows that contributed:

| script | use when |
| --- | --- |
| `diagnose_fever_qwen_regression.py` | FEVER label-accuracy below baseline; surfaces which label the model picked vs. gold |
| `diagnose_quality_regression.py` | QuALITY token-F1 looks low; shows extracted-vs-gold pairs |
| `diagnose_humaneval_failures.py` | HumanEvalPlus pass@1 below baseline; runs the test, prints stderr for failing samples |
| `inspect_musique.py` | MuSiQue route distribution + sample rows |
| `inspect_musique_abstentions.py` | MuSiQue rows ending in `NOT PRESENT IN CONTEXT`; flags rows where the gold answer is in reasoning text but synth hedged |
| `audit_hardmath_extract.py` | HARDMath extraction buckets (match / no_pred_box / format_mismatch / genuine_miss) |
| `audit_musique_extract.py` | MuSiQue prediction quality buckets (match / high_f1 / mid_f1 / low_f1 / abstention) |
| `oracle_quality_audit.py` | how many MuSiQue rows actually got real oracle context vs. silent fallback |

All take `--results "$OUT"` and write summaries to stdout. Use them
*after* you have the headline scores from the previous two sections —
they are diagnostic, not headline.

#### Promoting the run

Once you have the JSONL plus the score table, the artifacts that
matter for a writeup are:

1. The raw JSONL (`$OUT`) — full provenance, never overwrite.
2. The score table from the non-code scoring step (paste into
   `docs/p4-report.md` or your run log).
3. The `pass@1` line from HumanEvalPlus.
4. The decomposer cache at `artifacts/p4_decomposer_cache.jsonl` — so
   the next run can skip its decomposer calls.

### Troubleshooting

- **`apptainer: command not found`** — you forgot
  `module load apptainer`. Add it to your `~/.bashrc` or always source
  it before submitting.
- **`vllm-openai_latest.sif` missing** — pull/build the image into
  your scratch dir; the orchestrator expects it under
  `/scratch1/$USER/.cache`.
- **OOM mid-chunk** — drop `--chunk-size` (try 40). The smaller chunk
  means more cold starts but smaller peak memory.
- **vLLM `max_model_len` mismatch** — gemma-2-9b-it caps at 8192
  positions. The runner only bumps `max_model_len=16384` for the
  DeepSeek (math_code) specialist via `_bump_max_model_len`; do not
  apply it globally or vLLM will refuse to launch gemma.
- **Resume not skipping rows** — verify each JSONL line has an
  `index` field. Old logs without it will be re-run; that is expected.
- **Synthesizer port collision** — both the specialist and synthesizer
  configs use distinct ports (8000-range vs 8004). If you run two P4
  benchmarks back-to-back, wait for the previous job's vLLM to fully
  exit before submitting the next.
- **MuSiQue oracle hit rate below 100%** — check the build log for
  `unflagged → empty`. The runner uses `dgslibisey/MuSiQue` upstream
  for full `is_supporting` coverage; if it falls back to
  `bdsaglam/musique`, ~12% of rows lose their supporting flags.

**NOTE** The following sections are not related to reproducing results, and just documents extra code that is part of the repo, like data prep etc.

## Tests

Run package tests from the workspace root:

```bash
uv run pytest packages/benchmark_runner/tests
uv run pytest packages/benchmarking_pipeline/tests
uv run pytest packages/task_eval/tests
uv run pytest packages/llm_gateway/tests -m unit
uv run pytest packages/model-orchestration/tests
uv run pytest packages/council_policies/tests
```

For live provider smoke tests, set the required provider API keys first:

```bash
uv run pytest packages/llm_gateway/tests/test_smoke.py -m smoke -rs
```

## Data Utilities

Dataset preparation utilities remain available under `data_prep`:

```bash
uv run -m data_prep.build_router_dataset
uv run -m data_prep.upload_dataset \
  --folder-path ./data/router_dataset \
  --commit-message "<commit message>"
```

`update_parquet_files.ipynb` can be used for manual parquet edits when needed.
