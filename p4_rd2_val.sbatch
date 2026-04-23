#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=p4_rd2_val
#SBATCH --output=p4_run_rd2_val.log

cd $SLURM_SUBMIT_DIR

module purge
module load apptainer
export PATH="$HOME/.local/bin:$PATH"   # for uv

# Let HF/torch use all requested cores for CPU-bound decomposer generate().
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=true

which apptainer
which uv
nvidia-smi | head -20
echo "CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}  OMP=${OMP_NUM_THREADS}"

uv run --package council_policies --with peft python p4_benchmark_client.py \
  --dataset task-aware-llm-council/router_dataset-2 \
  --split validation \
  --limit -1 \
  --chunk-size 100 \
  --peft-adapter artifacts/decomposer_router_causal/adapter \
  --out p4_gemma_lora_v2_rd2_val.jsonl
