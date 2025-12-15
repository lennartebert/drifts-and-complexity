#!/bin/bash
#SBATCH --job-name=assess_datasets
#SBATCH --output=.scripts/logs/%A/log_%A_%a.out
#SBATCH --error=.scripts/logs/%A/log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --account=eberlenn

source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate drifts-and-complexity

python assess_datasets.py
