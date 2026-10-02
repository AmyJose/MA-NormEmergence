#!/bin/bash
#SBATCH --job-name=ethics-full-pilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=ethics-full-pilot-%j.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

nvidia-smi --list-gpus

time python -u code/run_condition.py \
    --model qwen3_8b \
    --prompt cooperative \
    --rule-policy cooperative \
    --seed 1 \
    --max-steps 75 \
    --output-root saved_runs/pilots/full_length
