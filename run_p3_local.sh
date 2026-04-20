#!/usr/bin/env bash
# Run P3 policy locally using a single local vLLM server (NVIDIA GPU).
# All specialist roles (qa, reasoning, general) share one model/server.
#
# Usage:
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash run_p3_local.sh
#   SAMPLE_CAP=10 DATASETS="musique fever" MODEL=... bash run_p3_local.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
SAMPLE_CAP="${SAMPLE_CAP:-5}"
DATASETS="${DATASETS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/p3_local_results}"
API_BASE="http://localhost:$PORT/v1"

echo "=== P3 local GPU run ==="
echo "Model      : $MODEL"
echo "Port       : $PORT"
echo "Sample cap : $SAMPLE_CAP per dataset"
echo "Output     : $OUTPUT_ROOT"
echo ""

uv sync --all-packages

# Start vLLM server in background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 &
VLLM_PID=$!

# Kill server on exit
trap "echo 'Stopping vLLM...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# Wait for server to be ready
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
  if curl -sf "$API_BASE/models" > /dev/null 2>&1; then
    echo "vLLM ready."
    break
  fi
  sleep 5
  if [ $i -eq 60 ]; then
    echo "ERROR: vLLM did not start after 5 minutes."
    exit 1
  fi
done

DATASET_ARGS=""
if [ -n "$DATASETS" ]; then
  DATASET_ARGS="--datasets $DATASETS"
fi

# All three specialist roles point to the same model on the same server
uv run p3-policy \
  --provider openai-compatible \
  --api-base "$API_BASE" \
  --qa-model "$MODEL" \
  --reasoning-model "$MODEL" \
  --general-model "$MODEL" \
  --sample-cap "$SAMPLE_CAP" \
  --output-root "$OUTPUT_ROOT" \
  --max-concurrency 10 \
  $DATASET_ARGS \
  --verbose

echo ""
echo "=== Done. Results in $OUTPUT_ROOT ==="
