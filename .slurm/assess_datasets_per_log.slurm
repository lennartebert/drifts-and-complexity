#!/bin/bash
#SBATCH --job-name=assess_datasets
#SBATCH --output=.slurm/logs/%A/log_%A_%a.out
#SBATCH --error=.slurm/logs/%A/log_%A_%a.err
#SBATCH --array=0-22
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --spread-job
#SBATCH --distribution=block:block
#SBATCH --time=3-00:00:00 # increase MaxTime to 3 days instead of 4 hours

source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate drifts-and-complexity

# Get project root (assuming script is in .slurm/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

logs=(ACCRE BPIC12 BPIC13_cp BPIC13_i BPIC14 BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5 BPIC17 BPIC18 BPIC19 BPIC20_dd BPIC20_id BPIC20_ptc BPIC20_rfp BPIC20_tpd CALL ITHD MOBIS RTFMP SEPSIS)

python "$PROJECT_ROOT/scripts/change_study/computations/assess_datasets.py" --datasets ${logs[$SLURM_ARRAY_TASK_ID]}
