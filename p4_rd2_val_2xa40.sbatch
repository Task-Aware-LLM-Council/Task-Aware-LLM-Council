#!/bin/bash
#
# P4 router_dataset-2:validation on 2×A40 with specialists + synth co-resident.
# Same topology as the 2×A100 variant but A40 has a much shorter queue on CARC.
#
# Layout (2 × 46GB = 92GB total). Only 3 vLLM servers; synth reuses the
# reasoning server (port 8002) via the client's _PRE_LAUNCHED_SYNTH_PORT
# alias — avoids duplicating 8GB of DeepSeek-7B weights.
#
# Spread across BOTH GPUs so vLLM's minimum KV-cache reservation fits:
#   GPU 0: qa(8001, gemma, util 0.42) + general(8003, qwen, util 0.40) ≈ 37.7GB, 8GB headroom
#   GPU 1: reasoning+synth(8002, DS-7B, util 0.40) ≈ 18.4GB + in-process decomposer on cuda:1 (~5GB)
#
# No spec<->synth GPU swap, no cold-start between chunks. Launches staggered
# by 30s. max-model-len=2048; QuALITY >2048-token contexts truncate.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=p4_rd2_val_2xa40
#SBATCH --output=p4_run_rd2_val_2xa40.log

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load apptainer
export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=true

VLLM_IMAGE="${VLLM_IMAGE:-/project/robinjia_1822/Task-Aware-LLM-Council/vllm-openai_latest.sif}"
if [[ ! -f "$VLLM_IMAGE" ]]; then
    VLLM_IMAGE="${SLURM_SUBMIT_DIR:-$PWD}/vllm-openai_latest.sif"
fi
CACHE_DIR="/scratch1/$(whoami)/.cache/huggingface"
mkdir -p "$CACHE_DIR"

echo "image=$VLLM_IMAGE  cache=$CACHE_DIR  cpus=${SLURM_CPUS_PER_TASK:-?}"
which apptainer && which uv
nvidia-smi | head -25

MODELS=(
    "task-aware-llm-council/gemma-2-9b-it-GPTQ"
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"
)
PORTS=(8001 8002 8003)
# Spread 3 servers across BOTH GPUs so each gets proper KV headroom.
# Packing all 3 on one 46GB A40 fails even at high util because vLLM
# reserves a minimum num_gpu_blocks (~512) whose memory cost scales
# with hidden-dim * layers, not just utilization.
#
# GPU 0: qa (gemma 9.5GB) + general (qwen 10GB)  — util 0.42 + 0.40 ≈ 37.7GB, 8GB headroom
# GPU 1: reasoning+synth (DS 8GB) solo            — util 0.40 ≈ 18.4GB, + decomposer ~5GB = 23GB used, 23GB headroom
GPUS=(0 1 0)
GPU_UTIL=(0.42 0.40 0.40)
QUANT=("compressed-tensors" "compressed-tensors" "compressed-tensors")

SERVER_PIDS=()
cleanup() {
    echo "Stopping vLLM servers..."
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait "${SERVER_PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT

for i in 0 1 2; do
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
        --max-model-len 2048 \
        > "p4_vllm_${PORTS[$i]}.log" 2>&1 &
    SERVER_PIDS+=($!)
    # Stagger launches so each server finishes reserving GPU memory before
    # the next starts its free-memory probe. Concurrent launches race and
    # one loses with "No available memory for the cache blocks".
    sleep 30
done

echo "Waiting for all 3 vLLM servers..."
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

echo "All 3 servers up. Running P4 client..."
uv run --package council_policies --with peft python p4_benchmark_client.py \
  --pre-launched \
  --decomposer-device cuda:1 \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 300 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val_2xa40.jsonl
