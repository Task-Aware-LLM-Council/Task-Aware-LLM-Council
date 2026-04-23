#!/bin/bash
#
# P4 router_dataset-2:validation on 2×A40 with specialists + synth co-resident.
# Same topology as the 2×A100 variant but A40 has a much shorter queue on CARC.
#
# Layout (2 × 44GB = 88GB total):
#   GPU 0: qa(8001) + reasoning(8002) + general(8003)   -- gpu_mem_util 0.35 each
#   GPU 1: synth(8004) + in-process decomposer (cuda:1) -- synth gpu_mem_util 0.35
# 3 specs (~19GB weights) + 3 KV caches on GPU 0, synth (~5GB) + decomposer
# (~5GB) on GPU 1. No spec<->synth GPU swap, no cold-start between chunks.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=p4_rd2_val_2xa40
#SBATCH --output=p4_run_rd2_val_2xa40.log

set -uo pipefail

cd $SLURM_SUBMIT_DIR

module purge
module load apptainer
export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=true

VLLM_IMAGE="${VLLM_IMAGE:-/project/robinjia_1822/Task-Aware-LLM-Council/vllm-openai_latest.sif}"
if [[ ! -f "$VLLM_IMAGE" ]]; then
    VLLM_IMAGE="$SLURM_SUBMIT_DIR/vllm-openai_latest.sif"
fi
CACHE_DIR="/scratch1/$(whoami)/.cache/huggingface"
mkdir -p "$CACHE_DIR"

echo "image=$VLLM_IMAGE  cache=$CACHE_DIR  cpus=$SLURM_CPUS_PER_TASK"
which apptainer && which uv
nvidia-smi | head -25

MODELS=(
    "task-aware-llm-council/gemma-2-9b-it-GPTQ"
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
)
PORTS=(8001 8002 8003 8004)
GPUS=(0 0 0 1)
GPU_UTIL=(0.35 0.35 0.35 0.35)
QUANT=("compressed-tensors" "compressed-tensors" "compressed-tensors" "compressed-tensors")

SERVER_PIDS=()
cleanup() {
    echo "Stopping vLLM servers..."
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait "${SERVER_PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT

for i in 0 1 2 3; do
    echo "Starting ${MODELS[$i]} on port ${PORTS[$i]} (GPU ${GPUS[$i]}, util ${GPU_UTIL[$i]})..."
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} apptainer run --nv --cleanenv \
        --bind "${CACHE_DIR}:${CACHE_DIR}" \
        --env "HF_HOME=${CACHE_DIR}" \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        "$VLLM_IMAGE" \
        --model "${MODELS[$i]}" \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port "${PORTS[$i]}" \
        --gpu-memory-utilization "${GPU_UTIL[$i]}" \
        --quantization "${QUANT[$i]}" \
        --max-model-len 8192 \
        > "p4_vllm_${PORTS[$i]}.log" 2>&1 &
    SERVER_PIDS+=($!)
done

echo "Waiting for all 4 vLLM servers..."
for port in "${PORTS[@]}"; do
    deadline=$((SECONDS + 900))
    until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        if (( SECONDS > deadline )); then
            echo "  port ${port} failed to start within 15min — tail log:"
            tail -n 80 "p4_vllm_${port}.log" || true
            exit 1
        fi
        sleep 5
    done
    echo "  port ${port} ready."
done

echo "All 4 servers up. Running P4 client..."
uv run --package council_policies --with peft python p4_benchmark_client.py \
  --pre-launched \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 300 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val_2xa40.jsonl
