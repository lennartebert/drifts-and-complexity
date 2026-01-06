#!/bin/bash
#SBATCH --job-name=combine_results_synthetic_kraus
#SBATCH --output=.slurm/logs/%A/log_%A.out
#SBATCH --error=.slurm/logs/%A/log_%A.err
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting to combine results from all Kraus synthetic datasets"

# Test if the module can be imported
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Testing Python module import..."
python -c "import scripts.change_study.computations.combine_results; print('Module import successful')" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to import module!"
    exit 1
}

# Track overall exit code
OVERALL_EXIT_CODE=0

# Function to run combine_results for a given noise level and parameter setting
run_combine_results() {
    local noise_flag=$1
    local noise_subfolder=$2
    local cp_setting=$3

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running combine_results for ${cp_setting}__cp_default with ${noise_subfolder}..."
    python -m scripts.change_study.computations.combine_results \
        "${noise_flag}" \
        --results-subfolder "${noise_subfolder}" \
        --cp-parameter-setting "${cp_setting}"
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed combining results for ${cp_setting}__cp_default (${noise_subfolder}) - Exit code: ${EXIT_CODE}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Combining results for ${cp_setting}__cp_default (${noise_subfolder}) failed - Exit code: ${EXIT_CODE}"
        OVERALL_EXIT_CODE=$EXIT_CODE
    fi
    return $EXIT_CODE
}

# Run for each noise level and each parameter setting
# No noise
run_combine_results "--synthetic-no-noise" "synthetic_kraus_no_noise" "processGraphsPDefaultWDefault"
run_combine_results "--synthetic-no-noise" "synthetic_kraus_no_noise" "groundtruth"

# 20% noise
run_combine_results "--synthetic-20-noise" "synthetic_kraus_20_noise" "processGraphsPDefaultWDefault"
run_combine_results "--synthetic-20-noise" "synthetic_kraus_20_noise" "groundtruth"

# 40% noise
run_combine_results "--synthetic-40-noise" "synthetic_kraus_40_noise" "processGraphsPDefaultWDefault"
run_combine_results "--synthetic-40-noise" "synthetic_kraus_40_noise" "groundtruth"

# Exit with the worst exit code encountered
if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All combine_results runs completed successfully"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: One or more combine_results runs failed - Exit code: ${OVERALL_EXIT_CODE}"
    exit $OVERALL_EXIT_CODE
fi
