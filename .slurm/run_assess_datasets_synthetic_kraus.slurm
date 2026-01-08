#!/bin/bash
#SBATCH --job-name=assess_datasets_synthetic_kraus
#SBATCH --output=.slurm/logs/%A/log_%A_%a.out
#SBATCH --error=.slurm/logs/%A/log_%A_%a.err
#SBATCH --array=0-299
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

# Generate all 300 datasets: 100 logs (1-100) × 3 noise levels (0, 20, 40)
# Format with 3-digit log numbers: KA25_001_00_S, KA25_002_00_S, ..., KA25_100_00_S
logs=()
for log_num in {1..100}; do
    log_num_padded=$(printf "%03d" ${log_num})
    logs+=("KA25_${log_num_padded}_00_S" "KA25_${log_num_padded}_20_S" "KA25_${log_num_padded}_40_S")
done

CURRENT_LOG=${logs[$SLURM_ARRAY_TASK_ID]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting assessment for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID})"

# Test if the module can be imported
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Testing Python module import..."
python -c "import scripts.change_study.computations.assess_datasets; print('Module import successful')" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to import module!"
    exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script with ground-truth-drifts mode..."
python -m scripts.change_study.computations.assess_datasets --datasets ${CURRENT_LOG} --mode ground-truth-drifts --fig-format pdf
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Assessment failed for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) in ground-truth-drifts mode - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script with all mode..."
python -m scripts.change_study.computations.assess_datasets --datasets ${CURRENT_LOG} --mode all --fig-format pdf
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed assessment for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) - Exit code: ${EXIT_CODE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Assessment failed for log: ${CURRENT_LOG} (Array task ID: ${SLURM_ARRAY_TASK_ID}) in all mode - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
