#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=p4_rd2_val
#SBATCH --output=p4_run_rd2_val.log

cd $SLURM_SUBMIT_DIR

uv run --package council_policies --with peft python p4_benchmark_client.py \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --chunk-size 100 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val.jsonl
