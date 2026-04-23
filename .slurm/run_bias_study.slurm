#!/bin/bash
#SBATCH --job-name=caise_study
#SBATCH --output=.slurm/logs/%A/log_%a.out
#SBATCH --error=.slurm/logs/%A/log_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --spread-job
#SBATCH --cpus-per-task=50
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --account=eberlenn
#SBATCH --time=1-00:00:00

set -euo pipefail

source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate drifts-and-complexity

# One array task per scenario (match --array upper bound to last index).
scenarios=(
    synthetic_base
    real_base
)

scenario="${scenarios[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script start (scenario=${scenario})"
python -m scripts.bias_study.run_bias_study --scenarios "${scenario}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run bias study complete(scenario=${scenario})"
python -m scripts.bias_study.generate_latex --scenarios "${scenario}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script complete (scenario=${scenario})"
