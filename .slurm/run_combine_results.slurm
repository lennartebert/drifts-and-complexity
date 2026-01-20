#!/bin/bash
#SBATCH --job-name=combine_results
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

# Define the same dataset list as in run_assess_datasets.slurm
logs=(ACCRE BPIC11 BPIC12 BPIC13_cp BPIC13_i BPIC14 BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5 BPIC17 BPIC18 BPIC19 BPIC20_dd BPIC20_id BPIC20_ptc BPIC20_rfp BPIC20_tpd CALL ITHD MOBIS RTFMP SEPSIS)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting to combine results from all datasets (${#logs[@]} datasets)"

# Test if the module can be imported
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Testing Python module import..."
python -c "import scripts.change_study.computations.combine_results; print('Module import successful')" || {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to import module!"
    exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script for processGraphsPDefaultWDefault..."
python -m scripts.change_study.computations.combine_results --datasets "${logs[@]}" --results-subfolder real_processGraphsPDefaultWDefault --cp-parameter-setting processGraphsPDefaultWDefault

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script for processGraphsPDefaultWDefault..."
python -m scripts.change_study.computations.combine_results --datasets "${logs[@]}" --results-subfolder real_emd200 --cp-parameter-setting emd200

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed combining results - Exit code: ${EXIT_CODE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Combining results failed - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
