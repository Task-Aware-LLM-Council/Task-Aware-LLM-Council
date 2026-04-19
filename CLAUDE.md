# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Task-Aware LLM Council** is a multi-model orchestration research system that routes tasks to specialized LLM roles and evaluates council policies (P1–P4) against five benchmark datasets (MuSiQue, QuALITY, FEVER, HardMath, HumanEval+).

Policy progression:
- **P1**: Single best-model baseline
- **P2**: Flat voting council
- **P3**: Rule-based task-routing to specialist roles
- **P4**: Learned router (partially implemented)

## Environment & Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # install all workspace packages and deps
```

After any `pyproject.toml` change, re-run `uv sync`.

## Common Commands

```bash
# Run all tests in a package
uv run pytest packages/<package>/tests -q

# Run a single test file
uv run pytest packages/<package>/tests/test_foo.py -q

# Run a single test
uv run pytest packages/<package>/tests/test_foo.py::test_name -q

# CLI entry points
uv run benchmark-runner          # top-level benchmark CLI
uv run p3-policy                 # P3 policy runner

# Run a module directly
uv run -m <package_name>.<script>
```

## Architecture

The workspace is organized as a monorepo under `packages/`:

| Package | Role |
|---|---|
| `common` | Shared schemas, utilities, ID generation |
| `llm_gateway` | OpenAI-compatible API client; supports OpenRouter and local vLLM |
| `model-orchestration` | Maps semantic roles → clients; owns per-role request defaults |
| `council_policies` | P2/P3/P4 policy logic (voting, synthesis, routing) |
| `task_eval` | Dataset profiles and metrics for each benchmark |
| `benchmark_runner` | Top-level orchestration CLI |
| `benchmarking_pipeline` | Low-level benchmark execution engine |
| `data_prep` | Dataset loaders and Hugging Face uploads |

### Request flow

```
PromptRequest
  → ModelOrchestrator.get_client(role)   # role: qa | math | code | reasoning | fever | general
    → llm_gateway client (async)
      → OrchestratorResponse
        → Policy aggregation (vote / synthesize)
          → CouncilResponse
```

### Key types

- `PromptRequest` — query container (defined in `llm_gateway`)
- `OrchestratorResponse` — model output with usage metadata
- `CouncilResponse` — final policy result including role provenance
- `EvaluationCase` — benchmark sample with dataset-specific answer extractor

### Conventions

- **Always use roles**, not direct client construction: `orchestrator.get_client(role)`.
- All client calls are **async**; sync wrappers exist but async is canonical.
- Per-role request defaults are merged automatically; explicit values in `PromptRequest` take precedence.
- Answer extraction is dataset-specific (regex/parsing); metrics are computed via `task_eval` profiles after extraction.
- Optional JSONL event logging via `OrchestratorCallRecord`.

## SLURM / HPC Scripts

- `run_benchmark_job.sh` — submits a vLLM-backed benchmark run to SLURM
- `run_p3_policy.sh` — submits P3 policy evaluation to SLURM

## Testing Notes

- Test framework: `pytest` with `pytest-asyncio`.
- Shared orchestrator stubs live in `packages/council_policies/tests/conftest.py`.
- Each package has its own `tests/` directory; run them per-package.
