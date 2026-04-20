#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────────
QA_MODEL="task-aware-llm-council/gemma-2-9b-it-GPTQ"
REASONING_MODEL="task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
GENERAL_MODEL="task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"
BASE_PORT="8000"
PARTITION="gpu"
TIME="02:00:00"
CPUS="16"
MEM="128G"
ACCOUNT="robinjia_1822"
GRES="gpu:a40:3"          # 3 GPUs — one per council role
N_PER_DATASET="50"
DATASETS=""               # empty = all 5 datasets
OUTPUT_ROOT=""            # empty = default scratch path
WORKDIR="$PWD"

usage() {
  cat <<'EOF'
Usage:
  ./run_p2_benchmark_job.sh [options]

Options:
  --qa-model MODEL
  --reasoning-model MODEL
  --general-model MODEL
  --base-port PORT          Base port; roles use PORT, PORT+1, PORT+2 (default: 8000)
  --partition PARTITION
  --time TIME
  --cpus-per-task N
  --mem MEM
  --account ACCOUNT
  --gres GRES
  --n-per-dataset N         Questions per dataset (default: 50)
  --datasets "d1 d2 ..."    Space-separated dataset names (default: all 5)
  --output-root DIR         Output directory (default: /scratch1/<user>/results/p2_benchmark)
  --workdir DIR
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qa-model)        QA_MODEL="$2";        shift 2 ;;
    --reasoning-model) REASONING_MODEL="$2"; shift 2 ;;
    --general-model)   GENERAL_MODEL="$2";   shift 2 ;;
    --base-port)       BASE_PORT="$2";       shift 2 ;;
    --partition)       PARTITION="$2";       shift 2 ;;
    --time)            TIME="$2";            shift 2 ;;
    --cpus-per-task)   CPUS="$2";            shift 2 ;;
    --mem)             MEM="$2";             shift 2 ;;
    --account)         ACCOUNT="$2";         shift 2 ;;
    --gres)            GRES="$2";            shift 2 ;;
    --n-per-dataset)   N_PER_DATASET="$2";   shift 2 ;;
    --datasets)        DATASETS="$2";        shift 2 ;;
    --output-root)     OUTPUT_ROOT="$2";     shift 2 ;;
    --workdir)         WORKDIR="$2";         shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$WORKDIR/logs"

JOB_NAME="p2-bench-$(date +%Y%m%dT%H%M%S)"

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
  --export=ALL,QA_MODEL="$QA_MODEL",REASONING_MODEL="$REASONING_MODEL",GENERAL_MODEL="$GENERAL_MODEL",BASE_PORT="$BASE_PORT",N_PER_DATASET="$N_PER_DATASET",DATASETS="$DATASETS",OUTPUT_ROOT="$OUTPUT_ROOT" \
  <<'EOF'
#!/bin/bash
set -euo pipefail

module load apptainer

echo "=== P2 Council Benchmark ==="
echo "QA model       : ${QA_MODEL}"
echo "Reasoning model: ${REASONING_MODEL}"
echo "General model  : ${GENERAL_MODEL}"
echo "Base port      : ${BASE_PORT}"
echo "N per dataset  : ${N_PER_DATASET}"
echo "Datasets       : ${DATASETS:-all}"

# Sync dependencies (skip torch which requires Linux-only wheels on mac but fine on CARC)
uv sync --all-packages

# Build dataset args
DATASET_ARGS=""
if [[ -n "${DATASETS}" ]]; then
  DATASET_ARGS="--datasets ${DATASETS}"
fi

# Build output-root args
OUTPUT_ARGS=""
if [[ -n "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ARGS="--output-root ${OUTPUT_ROOT}"
fi

uv run council-p2-bench \
  --provider local \
  --qa-model "${QA_MODEL}" \
  --reasoning-model "${REASONING_MODEL}" \
  --general-model "${GENERAL_MODEL}" \
  --base-port "${BASE_PORT}" \
  --n-per-dataset "${N_PER_DATASET}" \
  ${DATASET_ARGS} \
  ${OUTPUT_ARGS}

echo "=== P2 Benchmark Complete ==="
EOF

echo "Submitted job: $JOB_NAME"
echo "Logs: $WORKDIR/logs/$JOB_NAME-<jobid>.out"
