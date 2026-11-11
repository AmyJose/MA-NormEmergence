#!/bin/bash
#SBATCH --job-name=llm-ethics-test
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=llm-ethics-test-ai-%j.out

set -euo pipefail

cd "$HOME/MA-NormEmergence"
module load cray-python
source norm-env-ai/bin/activate

nvidia-smi --list-gpus

#python code/smoke_two_llms.py
python code/smoke_logging.py