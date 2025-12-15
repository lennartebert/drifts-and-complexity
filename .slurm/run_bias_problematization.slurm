#!/bin/bash
#SBATCH --job-name=run_bias_problematization
#SBATCH --output=.scripts/logs/%A/log_%A_%a.out
#SBATCH --error=.scripts/logs/%A/log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --account=eberlenn
#SBATCH --time=1-00:00:00 # increase MaxTime to 24hours instead of 4 hours


source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate drifts-and-complexity

python run_bias_problematization.py
