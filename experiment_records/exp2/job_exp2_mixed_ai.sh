#!/bin/bash
#SBATCH --job-name=ethics-exp2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-119%4
#SBATCH --output=experiment_records/exp2/logs/mixed-%A_%a.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

echo "Job: $SLURM_JOB_ID"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date -Is)"
nvidia-smi --list-gpus

time python -u code/run_exp2_task.py \
    --kind mixed \
    --task-id "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date -Is)"