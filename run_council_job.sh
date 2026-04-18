#!/usr/bin/env bash
# SLURM job: starts 3 vLLM servers (one per GPU) and runs the council.
#
# Usage:
#   sbatch run_council_job.sh --question "What is the capital of France?"
#
# Or submit directly:
#   sbatch --export=ALL,QUESTION="What is the capital of France?" run_council_job.sh
#
#SBATCH --job-name=council-run
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --account=robinjia_1822
#SBATCH --output=logs/council-%j.out
#SBATCH --error=logs/council-%j.err

set -euo pipefail

VLLM_IMAGE="/project/robinjia_1822/Task-Aware-LLM-Council/vllm-openai_latest.sif"
CACHE_DIR="/scratch1/$(whoami)/.cache/huggingface"
PORTS=(8000 8001 8002)
MODELS=(
    "task-aware-llm-council/gemma-2-9b-it-GPTQ"
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"
)
QUESTION="${QUESTION:-What is the capital of France?}"

mkdir -p logs "$CACHE_DIR"

# Cleanup all background vLLM servers on exit
cleanup() {
    echo "Stopping vLLM servers..."
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    wait "${SERVER_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

module load apptainer

uv sync --package council-policies

# GPU layout:
#   GPU 0 (shared): gemma-2-9b-GPTQ + DeepSeek-7B-AWQ  (0.45 each, ~9GB + ~5GB)
#   GPU 1 (solo):   Qwen2.5-14B-AWQ                     (0.90, ~9GB)
GPU_DEVICES=(0 0 1)
GPU_UTIL=(0.45 0.45 0.90)

SERVER_PIDS=()
for i in 0 1 2; do
    echo "Starting vLLM server for ${MODELS[$i]} on port ${PORTS[$i]} (GPU ${GPU_DEVICES[$i]})..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} apptainer run --nv --cleanenv \
        --bind "${CACHE_DIR}:${CACHE_DIR}" \
        --env "HF_HOME=${CACHE_DIR}" \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        "$VLLM_IMAGE" \
        --model "${MODELS[$i]}" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "${PORTS[$i]}" \
        --gpu-memory-utilization "${GPU_UTIL[$i]}" &
    SERVER_PIDS+=($!)
done

# Wait for all 3 servers to be ready
echo "Waiting for vLLM servers to be ready..."
for port in "${PORTS[@]}"; do
    until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        sleep 5
    done
    echo "  Port ${port} ready."
done

echo "All servers ready. Running council..."
uv run python run_council.py \
    --ports "${PORTS[@]}" \
    --question "$QUESTION"
