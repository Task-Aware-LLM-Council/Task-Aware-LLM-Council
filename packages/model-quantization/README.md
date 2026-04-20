# model-quantization

`model-quantization` is a CLI-first utility for quantizing Hugging Face causal
language models with `llmcompressor` and optionally uploading the resulting
artifacts to Hugging Face Hub.

Current supported modes:

- `4-bit` via AWQ (`W4A16_ASYM`)
- `8-bit` via GPTQ (`W8A16`)

The current public interface is the CLI command:

```bash
uv run model-quantization --help
```

## What It Does

For a given source model, the command:

1. authenticates to Hugging Face using `HUGGINGFACE_API_KEY`
2. inspects the model architecture with `transformers.AutoConfig`
3. chooses an AWQ or GPTQ recipe based on `--bits`
4. runs `llmcompressor.oneshot(...)` to produce quantized files locally
5. optionally uploads the generated folder to a Hugging Face repo

Local output is written under:

```text
<temp-dir>/<model-name-with-slashes-replaced-by-underscores>
```

Example:

```text
/scratch1/$USER/quantized_temp/Qwen_Qwen2.5-7B-Instruct
```

## Real Environment Requirements

Before running this in a real environment, ensure:

- the workspace dependencies are installed and `uv` can run the package
- `HUGGINGFACE_API_KEY` is set in the shell environment
- the source model is accessible with that Hugging Face account/token
- the machine has enough GPU/CPU memory for the source model and calibration run
- the temp directory has enough free space for the full quantized output
- the calibration dataset is available through the configured environment

Set auth like this:

```bash
export HUGGINGFACE_API_KEY=hf_xxx
```

## Basic Usage

### 4-bit AWQ with upload

```bash
uv run model-quantization \
  --model Qwen/Qwen2.5-7B-Instruct \
  --repo-id your-username/Qwen2.5-7B-Instruct-AWQ \
  --bits 4
```

### 8-bit GPTQ with upload

```bash
uv run model-quantization \
  --model google/gemma-2-9b-it \
  --repo-id your-username/gemma-2-9b-it-gptq \
  --bits 8
```

### Generate local quantized files only

```bash
uv run model-quantization \
  --model Qwen/Qwen2.5-7B-Instruct \
  --repo-id your-username/Qwen2.5-7B-Instruct-AWQ \
  --bits 4 \
  --skip-upload
```

### Retry upload from existing local files

Use this when quantization already completed and the files already exist in the
derived temp directory for the model.

```bash
uv run model-quantization \
  --model Qwen/Qwen2.5-7B-Instruct \
  --repo-id your-username/Qwen2.5-7B-Instruct-AWQ \
  --bits 4 \
  --temp-dir /scratch1/$USER/quantized_temp \
  --skip-quantize
```

## CLI Reference

### `--model`

Source Hugging Face model id.

Example:

```text
Qwen/Qwen2.5-7B-Instruct
```

Required: yes

### `--repo-id`

Destination Hugging Face repo id for upload.

Example:

```text
your-username/Qwen2.5-7B-Instruct-AWQ
```

Required: yes

### `--dataset`

Calibration dataset passed to `llmcompressor.oneshot(...)`.

Default:

```text
open_platypus
```

### `--samples`

Number of calibration samples.

Default:

```text
256
```

### `--seq-len`

Maximum calibration sequence length.

Default:

```text
1024
```

### `--temp-dir`

Base local directory used to store generated quantized artifacts before upload.

Default:

```text
/scratch1/$USER/quantized_temp
```

The script appends a model-specific subdirectory automatically.

### `--bits`

Quantization mode.

Supported values:

- `4`: AWQ (`W4A16_ASYM`)
- `8`: GPTQ (`W8A16`)

Default:

```text
4
```

### `--skip-quantize`

Skip the quantization step and only perform the upload step.

Use this only when the quantized files already exist in the derived output
directory for the selected model.

### `--skip-upload`

Skip the Hugging Face upload step and keep the quantized files locally.

## Runtime Behavior

### Authentication

The command reads:

```text
HUGGINGFACE_API_KEY
```

If the variable is missing, execution fails before quantization starts.

### Architecture detection

The script loads the source model config and reads `config.architectures[0]`.

If the architecture name contains `ForCausalLM`, it derives a sequential target
layer by replacing:

```text
ForCausalLM -> DecoderLayer
```

Example:

```text
Gemma2ForCausalLM -> Gemma2DecoderLayer
```

If the architecture does not match that pattern, the command continues without
sequential targets.

### AWQ path

For `--bits 4`, the command:

- looks up architecture-specific mappings in `AWQ_MAPPING_REGISTRY`
- fails if no mapping exists for the detected architecture
- uses `AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])`

### GPTQ path

For `--bits 8`, the command uses:

- `GPTQModifier(targets=["Linear"], scheme="W8A16", ignore=["lm_head"])`

### Upload path

If upload is enabled, the command:

- creates the destination repo if needed
- uploads the entire generated output folder with `HfApi.upload_folder(...)`

## Operational Notes

- This package is currently built for causal language models only.
- `--skip-quantize` still computes the same model-specific temp subdirectory
  internally; it does not upload arbitrary folders.
- The script prints a cleanup message, but local artifact deletion is currently
  disabled in code.
- AWQ support depends on the architecture being present in
  `AWQ_MAPPING_REGISTRY`.
- Recipe customization is currently minimal and mostly fixed in code.

## Recommended Real-World Workflow

1. Run with `--skip-upload` first and confirm the local artifacts look correct.
2. If quantization succeeds but upload fails, rerun with `--skip-quantize`.
3. Use a scratch-backed `--temp-dir` with enough capacity for intermediate
   artifacts.
4. Prefer validating one model end to end before launching multiple long runs.
