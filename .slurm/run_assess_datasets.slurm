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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Sourcing conda..."
source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Activating conda environment 'drifts-and-complexity'..."
conda activate drifts-and-complexity

# Debug: Verify conda environment is activated
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Active conda environment: ${CONDA_DEFAULT_ENV:-NOT_SET}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Python executable: $(which python)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Python version: $(python --version 2>&1)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Working directory: $(pwd)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

logs=(ACCRE BPIC12 BPIC13_cp BPIC13_i BPIC14 BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5 BPIC17 BPIC18 BPIC19 BPIC20_dd BPIC20_id BPIC20_ptc BPIC20_rfp BPIC20_tpd CALL ITHD MOBIS RTFMP SEPSIS)

CURRENT_LOG=${logs[$SLURM_ARRAY_TASK_ID]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting assessment for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID})"

# Test if the module can be imported
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Testing Python module import..."
python -c "import scripts.change_study.computations.assess_datasets; print('Module import successful')" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to import module!"
    exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script..."
python -m scripts.change_study.computations.assess_datasets --datasets ${CURRENT_LOG} --fig-format pdf
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed assessment for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) - Exit code: ${EXIT_CODE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Assessment failed for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
