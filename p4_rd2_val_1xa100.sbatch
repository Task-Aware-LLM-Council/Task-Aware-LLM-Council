#!/bin/bash
#
# P4 router_dataset-2:validation on 1×A100 with specialists + synth co-resident.
# Faster to schedule than the 2×A100 variant when a100:2 queue is contested.
#
# All 4 vLLM servers share a single 80GB A100:
#   port 8001 qa          (gemma-2-9b-it-GPTQ,       ~5GB weights)
#   port 8002 reasoning   (DeepSeek-R1-Qwen-7B-AWQ,  ~5GB weights)
#   port 8003 general     (Qwen2.5-14B-AWQ,          ~9GB weights)
#   port 8004 synth       (DeepSeek-R1-Qwen-7B-AWQ,  ~5GB weights)
#   + in-process decomposer on cuda:0 (Gemma-2-2B,    ~5GB)
# = ~29GB weights + per-server KV caches at gpu_memory_utilization 0.18 each
#   (4 × 0.18 = 0.72 of 80GB ≈ 58GB) + decomposer ~5GB ≈ 63GB. ~17GB headroom.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=p4_rd2_val_1xa100
#SBATCH --output=p4_run_rd2_val_1xa100.log
##SBATCH --account=robinjia_1822   # uncomment + set if your A100 access is under a shared account

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
GPU_UTIL=(0.18 0.18 0.18 0.18)
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
    echo "Starting ${MODELS[$i]} on port ${PORTS[$i]} (GPU 0, util ${GPU_UTIL[$i]})..."
    apptainer run --nv --cleanenv \
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
  --decomposer-device cuda:0 \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 300 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val_1xa100.jsonl
