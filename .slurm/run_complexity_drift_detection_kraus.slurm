#!/bin/bash
#SBATCH --job-name=complexity_drift_kraus
#SBATCH --output=.slurm/logs/%A/log_%A.out
#SBATCH --error=.slurm/logs/%A/log_%A.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

# Change to the cdrift_evaluation directory
cd plugins/cdrift_evaluation

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Changed to directory: $(pwd)"

# Test if the module can be imported
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Testing Python module import..."
python -c "from cdrift.approaches import complexity_drift_detection; print('Module import successful')" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to import module!"
    exit 1
}

# Use SLURM_CPUS_PER_TASK for the number of cores, but cap at 16 to avoid memory issues
NUM_CORES=${SLURM_CPUS_PER_TASK:-16}
if [ $NUM_CORES -gt 16 ]; then
    NUM_CORES=16
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running ComplexityDriftDetection evaluation on Kraus logs with ${NUM_CORES} cores..."
python testAll_reproducibility.py --kraus-only --cores ${NUM_CORES}
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ComplexityDriftDetection evaluation - Exit code: ${EXIT_CODE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ComplexityDriftDetection evaluation failed - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
