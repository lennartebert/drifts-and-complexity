#!/bin/bash
#SBATCH --job-name=assess_datasets_synthetic_kraus
#SBATCH --output=.slurm/logs/%A/log_%A_%a.out
#SBATCH --error=.slurm/logs/%A/log_%A_%a.err
#SBATCH --array=0-59
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

logs=(KA25_1_0_S KA25_1_20_S KA25_1_40_S KA25_2_0_S KA25_2_20_S KA25_2_40_S KA25_3_0_S KA25_3_20_S KA25_3_40_S KA25_4_0_S KA25_4_20_S KA25_4_40_S KA25_5_0_S KA25_5_20_S KA25_5_40_S KA25_6_0_S KA25_6_20_S KA25_6_40_S KA25_7_0_S KA25_7_20_S KA25_7_40_S KA25_8_0_S KA25_8_20_S KA25_8_40_S KA25_9_0_S KA25_9_20_S KA25_9_40_S KA25_10_0_S KA25_10_20_S KA25_10_40_S KA25_11_0_S KA25_11_20_S KA25_11_40_S KA25_12_0_S KA25_12_20_S KA25_12_40_S KA25_13_0_S KA25_13_20_S KA25_13_40_S KA25_14_0_S KA25_14_20_S KA25_14_40_S KA25_15_0_S KA25_15_20_S KA25_15_40_S KA25_16_0_S KA25_16_20_S KA25_16_40_S KA25_17_0_S KA25_17_20_S KA25_17_40_S KA25_18_0_S KA25_18_20_S KA25_18_40_S KA25_19_0_S KA25_19_20_S KA25_19_40_S KA25_20_0_S KA25_20_20_S KA25_20_40_S)

CURRENT_LOG=${logs[$SLURM_ARRAY_TASK_ID]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting assessment for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID})"

# Test if the module can be imported
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Testing Python module import..."
python -c "import scripts.change_study.computations.assess_datasets; print('Module import successful')" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to import module!"
    exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script with ground-truth-drifts mode..."
python -m scripts.change_study.computations.assess_datasets --datasets ${CURRENT_LOG} --mode ground-truth-drifts
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed assessment for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) - Exit code: ${EXIT_CODE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Assessment failed for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
