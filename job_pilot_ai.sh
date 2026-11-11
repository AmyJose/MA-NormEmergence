#!/bin/bash
#SBATCH --job-name=norm-pilot-ai
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=norm-pilot-ai-%j.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

nvidia-smi --list-gpus

python -u code/run_condition.py \
    --model qwen3_8b \
    --prompt unframed \
    --rule-policy self_interested \
    --seed 43 \
    --max-steps 3
