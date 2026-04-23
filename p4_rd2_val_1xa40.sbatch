#!/bin/bash
#
# P4 router_dataset-2:validation on a single 46GB A40 with specialists + synth
# + decomposer all co-resident on GPU 0. Fastest queue (a40:1 backfills into
# most fragments).
#
# Layout (all on GPU 0, 46GB total):
#   port 8001 qa          (gemma-2-9b-it-GPTQ,       ~5GB weights)
#   port 8002 reasoning   (DeepSeek-R1-Qwen-7B-AWQ,  ~5GB weights)
#   port 8003 general     (Qwen2.5-14B-AWQ,          ~9GB weights)
#   port 8004 synth       (DeepSeek-R1-Qwen-7B-AWQ,  ~5GB weights)
#   + in-process decomposer on CPU (Gemma-2-2B)   -- keeps GPU free for vLLM
# Budget: 4 × 0.22 × 46GB ≈ 40GB vLLM, ~6GB headroom. Launches staggered by
# 30s to avoid concurrent memory-reservation races. Max model len lowered to
# 4096 so KV cache fits — QuALITY >4096-token contexts will be truncated; if
# scoring later shows QuALITY regression vs A100 baseline, revert that one.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=p4_rd2_val_1xa40
#SBATCH --output=p4_run_rd2_val_1xa40.log

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
# 4 servers share one 46GB A40. Budget: 4 * 0.22 * 46 ~= 40GB, ~6GB headroom.
# Decomposer moves to CPU in this mode to avoid a 5th load on the same card.
GPU_UTIL=(0.22 0.22 0.22 0.22)
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
        --max-model-len 4096 \
        > "p4_vllm_${PORTS[$i]}.log" 2>&1 &
    SERVER_PIDS+=($!)
    # Stagger launches so each server finishes reserving GPU memory before the
    # next starts its free-memory probe. Concurrent launches race and lose.
    sleep 30
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
  --decomposer-device cpu \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 300 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val_1xa40.jsonl
