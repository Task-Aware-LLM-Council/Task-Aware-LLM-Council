#!/usr/bin/env bash
set -euo pipefail

MODEL="task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
PORT="8000"
PARTITION="gpu" # Corrected from "gpi"
TIME="01:00:00"
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
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
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

JOB_NAME="pilot-$(echo "${MODEL##*/}" | tr '/[:space:]' '-' | cut -c1-40)"

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
  --export=ALL,MODEL_NAME="$MODEL",PORT="$PORT" <<'EOF'
#!/bin/bash
set -euo pipefail

module load apptainer

echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"

# Install/sync dependencies first
uv sync --all-packages

uv run benchmark-runner \
  --preset pilot \
  --provider vllm \
  --models "$MODEL_NAME" \
  --local-launch-port "$PORT"
EOF


# ./run_benchmark_job.sh  \
#     --model task-aware-llm-council/gemma-2-9b-it-AWQ \
#     --port 8001  \
#     --partition gpu \
#     --time 02:00:00   \
#     --cpus-per-task 8 \
#     --mem 64G