#!/bin/bash
#SBATCH --job-name=generate_collection_of_logs
#SBATCH --output=.slurm/logs/%A/log_%A.out
#SBATCH --error=.slurm/logs/%A/log_%A.err
#SBATCH --nodelist=gruenau4
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Sourcing conda..."
source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Activating conda environment 'cdlg_tool'..."
conda activate cdlg_tool

# Debug: Verify conda environment is activated
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Active conda environment: ${CONDA_DEFAULT_ENV:-NOT_SET}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Python executable: $(which python)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Python version: $(python --version 2>&1)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Working directory: $(pwd)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

# Change to cdlg_tool directory
cd plugins/cdlg_tool

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Changed to directory: $(pwd)"

# Test if the script file exists
SCRIPT_PATH="generate_collection_of_logs.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Script file not found: $SCRIPT_PATH"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Running Python script..."
python "$SCRIPT_PATH"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed generate_collection_of_logs - Exit code: ${EXIT_CODE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: generate_collection_of_logs failed - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
