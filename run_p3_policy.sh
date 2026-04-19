#!/usr/bin/env bash
# Submit a SLURM job to run P3 policy on all datasets using local vLLM.
set -euo pipefail

POLICY="${1:-p3}"   # p3 or p2
ACCOUNT="robinjia_1822"
PARTITION="gpu"
TIME="06:00:00"
CPUS="12"
MEM="64G"
GRES="gpu:a40:1"
SAMPLE_CAP="${SAMPLE_CAP:-500}"
WORKDIR="$PWD"

mkdir -p "$WORKDIR/logs"

sbatch \
  --job-name="${POLICY}-policy-run" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --account="$ACCOUNT" \
  --gres="$GRES" \
  --chdir="$WORKDIR" \
  --output="$WORKDIR/logs/${POLICY}-%j.out" \
  --error="$WORKDIR/logs/${POLICY}-%j.err" \
  --export=ALL,POLICY="$POLICY",SAMPLE_CAP="$SAMPLE_CAP" <<'SBATCH'
#!/bin/bash
set -euo pipefail

module load apptainer

echo "=== Starting ${POLICY}-policy run ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

uv sync --all-packages

OUTPUT_ROOT="/scratch1/$(whoami)/results"

uv run "${POLICY}-policy" \
  --provider local \
  --sample-cap "$SAMPLE_CAP" \
  --output-root "$OUTPUT_ROOT" \
  --verbose

echo "=== Done. Results in $OUTPUT_ROOT ==="
SBATCH
