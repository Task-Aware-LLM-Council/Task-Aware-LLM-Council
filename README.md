# Task-Aware LLM Council — P3 Branch

This branch (`please_work`) reproduces the rule-based routing (P3) council policy. The README only documents what is needed to reproduce P3 results — for P1, P2, and P4 use the corresponding branches.

## Environment Setup

### Prerequisites

- Python >=3.12
- `uv`: https://docs.astral.sh/uv/getting-started/installation/
- For running on CARC, vLLM and Python apptainer images are required. Copy both of them from this location - `/project2/robinjia_1822/apptainer-compatible-images-dont-delete` to the root of repository.

### Install

Clone the repository, check out this branch, move into the project root, and sync every workspace package:

```
git fetch origin
git checkout please_work
uv sync --all-packages
```

Use `uv run` for commands so package entry points resolve from the workspace environment:

```
uv run <command> --help
```

### Provider Credentials

P3 runs locally on vLLM and does not need an API key, but it does need the local vLLM Apptainer image (see Prerequisites). If you want to swap to a hosted provider, export only the keys you plan to use:

```
export OPENAI_API_KEY=<openai-key>
export OPENROUTER_API_KEY=<openrouter-key>
export HUGGINGFACE_API_KEY=<huggingface-key>
export NVIDIA_API_KEY=<nvidia-key>
```

## Steps To Reproduce Results

All these tests were done on CARC using a40, so make sure to get correct resources assigned. `srun --pty -p gpu --gres=gpu:a40:1 --account=robinjia_1822 --time=04:00:00 --cpus-per-task=8 --mem=64G bash`

Run `module load apptainer` after resources are allocated. This is required to load apptainer runtime.

P3 uses these final models as specialists, so only these models should be used. Ignore any other models in the repository, since there are few other models that were part of initial pool of models from which specialists were chosen.

```
"task-aware-llm-council/gemma-2-9b-it-GPTQ"
"task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
"task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"
```

**Note** P3 was run on single a40 GPU (since getting multiple GPUs was almost impossible), hence use a40 since it is the minimum spec GPU with enough vRAM to run 3 models and minimum CUDA Compute Capability to run vLLM.

### Rule-Based-Routing (P3) Council Benchmarking

P3 uses `council_policies` to route every question to a single specialist (no voting, no synthesis). The dataset name + keyword regexes over the question pick one of `qa` / `reasoning` / `math` / `code` / `fever`, that task type is mapped to a specialist role, and that role's model produces the final answer. The CLI attaches a per-dataset system prompt that constrains output shape (`Final Answer:` / `\boxed{}` / fenced ```python block / `SUPPORTS|REFUTES|NOT ENOUGH INFO`) and scores the response with the matching metric.

This will load all 3 specialists in one GPU and keep them loaded for the entire run — there is no swap phase like in P2. The whole process can take ~45–60 min on a single a40.

#### Local vLLM Run

```
uv run p3-policy
```

That is the whole command — model selection, routing, dataset, and scoring are all wired to the three council specialists and the `task-aware-llm-council/router_dataset` test split.

Default P3 output file (written to the current working directory):

```
results_p3_eval_local.jsonl
```

Each record contains `example_id`, `dataset_name`, `suite_id`, `question`, `system_prompt`, `context`, `task_type`, `routed_role`, `model`, `response_text`, `latency_ms`, `failed`, `error`, `usage`, `reference`, and a `metrics` block with the dataset's relevant scorer:

- `musique`, `quality` → `token_f1`, `exact_match`
- `fever`, `hardmath` → `accuracy`
- `humaneval` → `pass_at_1`

`metric_metadata.extracted_answer` records exactly what the scorer compared. At the end of the run the CLI prints a routing summary per dataset (question count, role distribution, task-type distribution, average latency).

If `task_eval.scoring.pass_at_1` is importable and apptainer is available, HumanEval predictions are executed inside the sandbox harness against the dataset's `test_code` + `entry_point`. Otherwise the scorer falls back to a bare `python3` exec.

## Tests

Run package tests from the workspace root:

```
uv run pytest packages/council_policies/tests
uv run pytest packages/task_eval/tests
uv run pytest packages/model-orchestration/tests
```
