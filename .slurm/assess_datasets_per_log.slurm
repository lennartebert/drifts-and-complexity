#!/bin/bash
#SBATCH --job-name=assess_datasets
#SBATCH --output=.scripts/logs/%A/log_%A_%a.out
#SBATCH --error=.scripts/logs/%A/log_%A_%a.err
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

echo "Python path:"
which python

echo "Python executable:"
python -c "import sys; print(sys.executable)"

echo "PyYAML version:"
python -c "import yaml; print(yaml.__version__)"

# Define your list of logs
logs=(ACCRE BPIC12 BPIC13_cp BPIC13_i BPIC14 BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5 BPIC17 BPIC18 BPIC19 BPIC20_dd BPIC20_id BPIC20_ptc BPIC20_rfp BPIC20_tpd CALL ITHD MOBIS RTFMP SEPSIS)

# Select log for this array task
LOG_LABEL=${logs[$SLURM_ARRAY_TASK_ID]}

echo "Processing log: $LOG_LABEL"
python assess_datasets.py --datasets "$LOG_LABEL" --plot-coverage-curves
