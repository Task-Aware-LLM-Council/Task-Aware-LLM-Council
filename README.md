# Task-Aware LLM Council

Monorepo for task-aware LLM benchmarking and council-style model policies. The
workspace includes reusable packages for provider access, benchmark execution,
task evaluation, local model orchestration, quantization, and policy runs.

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
# Steps To Reproduce results
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
git checkout <P3_BRANCH_NAME>
```

## P4 Branch

P4 lives on a separate branch since there are multiple steps to run it.

```bash
git fetch origin
git checkout <P4_BRANCH_NAME>
```

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
