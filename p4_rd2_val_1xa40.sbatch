#!/bin/bash
#
# P4 router_dataset-2:validation on a single 46GB A40 with specialists + synth
# + decomposer all co-resident on GPU 0. Fastest queue (a40:1 backfills into
# most fragments).
#
# Layout (all on GPU 0, 46GB total). ONLY 3 vLLM servers — synth reuses the
# reasoning server on port 8002 (same DS-R1-Qwen-7B model) to avoid
# duplicating ~8GB of weights on a card where every GB matters:
#   port 8001 qa         (gemma-2-9b-it-GPTQ,      9.5GB weights, util 0.32)
#   port 8002 reasoning+synth (DS-R1-Qwen-7B-AWQ,   8GB weights, util 0.26)
#   port 8003 general    (Qwen2.5-14B-AWQ,         10GB weights, util 0.28)
#   + in-process decomposer on CPU (Gemma-2-2B) — keeps GPU free for vLLM
# Budget: 0.86 × 46GB ≈ 39.6GB vLLM, ~6.4GB headroom. Staggered 30s apart
# to dodge concurrent memory-reservation races. max-model-len=2048; at 4096
# vLLM needed ~9.6GB KV cache for gemma-2-9B alone (more than the weights!),
# which couldn't fit. QuALITY contexts >2048 tokens will truncate; revisit
# if scoring flags QuALITY regression.
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
)
PORTS=(8001 8002 8003)
# Per-model util tuned to observed weight sizes. Sum = 0.86 × 46 ≈ 39.6GB.
# The reasoning server (port 8002) also serves synth requests via the client
# config's _PRE_LAUNCHED_SYNTH_PORT alias.
GPU_UTIL=(0.32 0.26 0.28)
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
        --max-model-len 2048 \
        > "p4_vllm_${PORTS[$i]}.log" 2>&1 &
    SERVER_PIDS+=($!)
    # Stagger launches so each server finishes reserving GPU memory before the
    # next starts its free-memory probe. Concurrent launches race and lose.
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
  --decomposer-device cpu \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 300 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val_1xa40.jsonl
