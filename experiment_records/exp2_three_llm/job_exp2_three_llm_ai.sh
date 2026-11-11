#!/bin/bash
#SBATCH --job-name=ethics-three-llm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-119%8
#SBATCH --output=experiment_records/exp2_three_llm/logs/mixed-%A_%a.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

echo "Started: $(date -Is)"
nvidia-smi --list-gpus

time python -u code/run_exp2_task.py \
    --kind mixed \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --llm-agent-ids 0 1 2 \
    --output-root saved_runs/exp2_three_llm

echo "Finished: $(date -Is)"