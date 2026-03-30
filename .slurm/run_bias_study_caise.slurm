#!/bin/bash
#SBATCH --job-name=caise_study
#SBATCH --output=.scripts/logs/%A/log_%a.out
#SBATCH --error=.scripts/logs/%A/log_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --spread-job
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --account=eberlenn
#SBATCH --time=1-00:00:00

source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate drifts-and-complexity

# One array task per scenario (match --array upper bound to last index).
scenarios=(
    synthetic_base
    real_base
)

python -m scripts.bias_study.run_bias_study --scenarios "${scenarios[$SLURM_ARRAY_TASK_ID]}"
