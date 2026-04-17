# Task-Aware LLM Council

Monorepo for building, packaging, and consuming a router dataset for NLP workflows.

## Project Setup (uv)

### Prerequisites

- `uv` - https://docs.astral.sh/uv/getting-started/installation/

### Initial setup

- Clone the repository and move into the project root.
- Run `uv sync` to install dependencies and set up the workspace.

### Recommended uv workflow
- Keep everything as separate packages under `packages/` for better modularity and dependency management.
```
uv init --lib packages/<package_name>
```
- Use `uv sync` when dependencies change (from root folder).
- Use `uv add <dependency>` to add dependencies to specific packages.
```
cd packages/<package_1>
uv add <package_2>  # Adds package_2 as a dependency to package_1
cd ../../
uv lock  # Update the lockfile after adding dependencies
uv sync  # Sync the workspace to install new dependencies
```
- Use `uv run` to execute scripts without manually activating the environment.
```
 uv run -m <package_name>.<script_name>
```
- Use workspace-aware sync/install options when working across all packages.
- Make sure to select correct interpreters in VSCode (Ctrl+Shift+P > Python: Select Interpreter) to get proper linting and IntelliSense.

## Project Structure

```text
NLP-Project/
|- pyproject.toml              # Root uv workspace config
|- main.py                     # Root entry script (placeholder)
|- data/
|  |- router_dataset/       # Local parquet exports
|- packages/
|  |- common/
|  |  |- src/common/           # Shared schemas, IO, IDs, split/util helpers
|  |- data_prep/
|  |  |- src/data_prep/        # Dataset loaders, dataset build/upload scripts
|  |- training/
|  |  |- src/training/         # Dataset consumption/training-facing scripts
|  |- inference/
|  |  |- src/inference/        # Inference package scaffold
|  |- llm_gateway/
|  |  |- src/llm_gateway/        # client layer for text-generation style LLM calls
|- update_parquet_files.ipynb  # Notebook utility for parquet updates
```

## Available Script Types

Fill in your preferred command format for each item below.

### Data preparation scripts

- Build common dataset
```
uv run -m data_prep.build_router_dataset  
- Modify parquet files using the provided notebook (update_parquet_files.ipynb) or custom scripts as needed. Handle the parquet files as pandas dataframe and make modifications, then save back to the same location to update the dataset splits.
```
- Upload to Hugging Face Hub (ask for token if you don't have it)
```
uv run -m data_prep.upload_dataset --folder-path='./data/router_dataset' --commit-message="<commit message>"
```
<!-- ### Training scripts

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

## Metrics Analysis
See [analysis/README.md](analysis/README.md) for details on system-level metrics (tokens, calls, routing accuracy) and the analysis scripts.