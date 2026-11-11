#!/bin/bash
#SBATCH --job-name=ethics-exp2-rules
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=experiment_records/exp2/logs/rules-%j.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

echo "Started: $(date -Is)"

for task_id in {0..29}; do
    python -u code/run_exp2_task.py \
        --kind rule_only \
        --task-id "$task_id"
done

echo "Finished: $(date -Is)"