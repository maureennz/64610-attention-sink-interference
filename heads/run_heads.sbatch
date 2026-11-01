#!/bin/bash
# Sink identification + retrieval scores for one model on a single GPU.
# Submit from the repo root:  sbatch heads/run_heads.sbatch [qwen2.5-1.5b|llama-3.2-1b]
#SBATCH -J heads
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail
MODEL=${1:-qwen2.5-1.5b}

module load miniforge/25.11.0-0
eval "$(conda shell.bash hook)"
conda activate attn-sink

nvidia-smi --query-gpu=name,memory.total --format=csv
cd heads
python find_sinks.py --model "$MODEL"
python retrieval_score.py --model "$MODEL"
