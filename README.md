# NLP-Project

Monorepo for building, packaging, and consuming a router dataset for NLP workflows.

## Project Setup (uv)

### Prerequisites

- `uv` - https://docs.astral.sh/uv/getting-started/installation/

### Initial setup

- Clone the repository and move into the project root.
- Run `uv sync` to install dependencies and set up the workspace.

<!-- ### Recommended uv workflow

1. Use `uv sync` when dependencies change.
2. Use `uv run ...` to execute scripts without manually activating the environment.
3. Use workspace-aware sync/install options when working across all packages.

## Project Structure

```text
NLP-Project/
|- pyproject.toml              # Root uv workspace config
|- main.py                     # Root entry script (placeholder)
|- data/
|  |- router_dataset_v1/       # Local parquet exports
|- packages/
|  |- common/
|  |  |- src/common/           # Shared schemas, IO, IDs, split/util helpers
|  |- data_prep/
|  |  |- src/data_prep/        # Dataset loaders, dataset build/upload scripts
|  |- training/
|  |  |- src/training/         # Dataset consumption/training-facing scripts
|  |- inference/
|  |  |- src/inference/        # Inference package scaffold
|- update_parquet_files.ipynb  # Notebook utility for parquet updates
```

## Available Script Types

Fill in your preferred command format for each item below.

### Data preparation scripts

- Build merged router dataset from multiple sources.
- Source-specific dataset loaders (MuSiQue, QuALITY, FEVER, HARDMATH, HumanEvalPlus).
- Upload/publish dataset artifacts.

### Training scripts

- Consume/inspect prepared dataset splits for downstream training workflows.

### Inference scripts

- Inference package scaffold is present; runtime scripts can be added here.

### Shared utility modules (imported by scripts)

- Hugging Face dataset IO helpers.
- Router schema/types.
- ID generation and split utilities.

### Root-level scripts

- Project sanity-check or bootstrap script from `main.py`.

## Notes

- This repo uses a uv workspace with members in `packages/common`, `packages/data_prep`, `packages/training`, and `packages/inference`.
- Internal package linking is configured via workspace sources (for example, `common` used by `data_prep` and `training`). -->