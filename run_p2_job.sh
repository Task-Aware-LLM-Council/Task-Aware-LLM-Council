#!/usr/bin/env bash
set -euo pipefail

PARTITION="gpu" # Corrected from "gpi"
TIME="03:00:00"
CPUS="8"
MEM="64G"
ACCOUNT="robinjia_1822"
GRES="gpu:a40:1"
WORKDIR="$PWD"

usage() {
  cat <<'EOF'
Usage:
  ./submit_pilot.sh [options]

Options:
  --model MODEL
  --port PORT
  --partition PARTITION
  --time TIME
  --cpus-per-task N
  --mem MEM
  --account ACCOUNT
  --gres GRES
  --workdir DIR
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --cpus-per-task) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$WORKDIR/logs"

JOB_NAME="p2_policy"

sbatch \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --account="$ACCOUNT" \
  --gres="$GRES" \
  --chdir="$WORKDIR" \
  --output="$WORKDIR/logs/%x-%j.out" \
  --error="$WORKDIR/logs/%x-%j.err" \
  --export=ALL <<'EOF'
#!/bin/bash
set -euo pipefail

module load apptainer

# Install/sync dependencies first
uv sync --all-packages

uv run --package council-policies council-policies --provider vllm

EOF


# ./run_p2_job.sh