#!/bin/bash
#SBATCH --job-name=ethics-four-llm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=0-39%4
#SBATCH --output=experiment_records/exp2_four_llm/logs/mixed-%A_%a.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

# Select the first rule-policy block for each framing.
# No rule agents exist, so the policy argument is unused.
task_id=$(( (SLURM_ARRAY_TASK_ID / 10) * 30 + SLURM_ARRAY_TASK_ID % 10 ))

echo "Array task: $SLURM_ARRAY_TASK_ID; manifest task: $task_id"
echo "Started: $(date -Is)"
nvidia-smi --list-gpus

time python -u code/run_exp2_task.py \
    --kind mixed \
    --task-id "$task_id" \
    --llm-agent-ids 0 1 2 3 \
    --output-root saved_runs/exp2_four_llm

echo "Finished: $(date -Is)"