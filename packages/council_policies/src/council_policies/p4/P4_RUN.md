# Running P4 (LearnedRouterPolicy) on CARC

End-to-end recipe for evaluating P4 — the joint decomposer+router + quantized
specialists + synthesizer pipeline — on a single A40 via Slurm.

---

## 1. Switch to the P4 branch

P4 lives on `router-training`. From the repo root:

```bash
git fetch origin
git checkout router-training
git pull --ff-only
```

Verify you have the P4 layout:

```bash
ls packages/council_policies/src/council_policies/p4/
# should show: policy.py, router.py, decomposer.py, synthesis.py,
#              policy_runner.py, types.py, ...
```

---

## 2. Mandatory environment setup

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

Sync the workspace (only needed after a pull or first checkout):

```bash
uv sync --all-packages
```

> The vLLM image (`vllm-openai_latest.sif`) and HF cache live under
> `/scratch1/$USER/.cache`. The orchestrator binds that path into the
> container automatically — you do not need to bind-mount manually.

### Pre-warm the HF cache (optional but strongly recommended)

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

---

## 3. Configure the run

The runner is `p4_benchmark_client.py` at the repo root. The two knobs you
will touch most are **how many rows to process** and **how often to flush
to disk**:

| flag | what it does | typical value |
| --- | --- | --- |
| `--dataset` | HF dataset for eval prompts | `task-aware-llm-council/router_dataset-2` |
| `--split` | dataset split | `validation` or `test` |
| `--limit` | cap rows; `-1` = run the whole split | `25` for smoke, `-1` for full eval |
| `--chunk-size` | flush + spec↔synth GPU swap every N prompts | `60`–`100` |
| `--out` | JSONL output path; resume reads from this | `p4_results.jsonl` |
| `--peft-adapter` | optional LoRA adapter on top of the base decomposer | `artifacts/decomposer_router_causal/adapter` |

### Why `--chunk-size` matters

The 44 GB A40 cannot hold all 3 specialists *and* the synthesizer at the
same time. Each chunk pays one specialist↔synthesizer swap (≈ 8 min cold
start). Bigger chunks = fewer cold starts but a larger blast radius if
the job dies mid-chunk and the in-flight rows are lost. 60–100 is the
sweet spot.

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

---

## 4. Smoke-test on the debug GPU first

Before burning a 2 h `gpu` allocation on a config you have not run end-to-end
yet, verify the pipeline on the **debug** partition with a tiny `--limit`.
This catches the cheap failures (missing image, wrong adapter path, dataset
typo, port collision) in 30 minutes of cheap wall clock instead of waiting
through the full queue.

### Option A — interactive `salloc`

The fastest feedback loop. You get a shell on the GPU node and can iterate
on flags without resubmitting:

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
without paying for more than one of them. Tail the output and watch for:

- decomposer loads (`HFCausalGenerate` log line),
- vLLM specialist server comes up on port 8000-8003,
- specialist phase completes,
- specialist vLLM exits cleanly,
- synthesizer vLLM comes up on port 8004,
- synthesizer phase completes,
- 5 lines land in `p4_smoke.jsonl`.

If any of those steps stall or error, fix it here — not in a queued
2 h job.

### Option B — debug-partition sbatch loop

If `salloc` is queued and you'd rather batch-submit, copy `p4_rd2_val.sbatch`
to `p4_debug.sbatch` and change the header + the run line:

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

Then drive iterations with the same resume property the full loop uses —
each rerun resumes against `p4_smoke.jsonl`, so if you grow `--limit`
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

Once `p4_smoke.jsonl` has 5 healthy rows, delete it (or pick a different
`--out`) and move on to the full run below. **Never reuse the smoke jsonl
as the full-run output** — its rows came from `--limit 5`, but the runner
keys resume by `index` and would silently skip those indices in the full
split.

---

## 5. Submit the full job

```bash
sbatch p4_rd2_val.sbatch
squeue -u $USER          # watch the queue
tail -f p4_run_rd2_val.log   # live log once the job starts
```

Each job is wall-time-capped at 2 h (see `#SBATCH --time=02:00:00`). On a
full split that is **not** enough to finish — which is the whole point of
the resume loop below.

---

## 6. The resume loop (rerunning until done)

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

### Manual loop

```bash
sbatch p4_rd2_val.sbatch                       # job 1
# … wait for completion / timeout, check progress:
wc -l p4_gemma_lora_v2_rd2_val.jsonl           # rows done so far
sbatch p4_rd2_val.sbatch                       # job 2 — resumes
# repeat until line-count matches the split size
```

### Automated loop

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

---

## 7. Inspect results

```bash
wc -l p4_gemma_lora_v2_rd2_val.jsonl                  # total rows
jq -s 'length' p4_gemma_lora_v2_rd2_val.jsonl         # same, validating JSON
jq -c '{index, task_type, winner_role: .metadata.routed_role}' \
  p4_gemma_lora_v2_rd2_val.jsonl | head
```

Each line is one prompt's `CouncilResponse` plus provenance
(`routed_role`, `synthesizer_role`, latency, usage).

---

## Troubleshooting

- **`apptainer: command not found`** — you forgot `module load apptainer`.
  Add it to your `~/.bashrc` or always source it before submitting.
- **`vllm-openai_latest.sif` missing** — pull/build the image into your
  scratch dir; the orchestrator expects it under
  `/scratch1/$USER/.cache`.
- **OOM mid-chunk** — drop `--chunk-size` (try 40). The smaller chunk
  means more cold starts but smaller peak memory.
- **Resume not skipping rows** — verify each JSONL line has an `index`
  field. Old logs without it will be re-run; that is expected.
- **Synthesizer port collision** — both the specialist and synthesizer
  configs use distinct ports (8000-range vs 8004). If you run two P4
  benchmarks back-to-back, wait for the previous job's vLLM to fully
  exit before submitting the next.
