#!/bin/bash
#SBATCH --job-name=signal_noise_study
#SBATCH --output=.slurm/logs/%A/log_%A.out
#SBATCH --error=.slurm/logs/%A/log_%A.err
#SBATCH --nodelist=gruenau5
#SBATCH --cpus-per-task=50
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00

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

# Set PYTHONPATH to project root so imports work
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ---------------------------------------------------------------------------
# Skip-to logic:
#   SKIP_TO=0 (default) — run all steps (1 → 2 → 3)
#   SKIP_TO=1           — skip step 1, start from step 2 (2 → 3)
#   SKIP_TO=2           — skip steps 1 & 2, start from step 3 (3)
#
# Usage:
#   sbatch .slurm/run_signal_noise_study.slurm                        # all steps
#   sbatch --export=ALL,SKIP_TO=1 .slurm/run_signal_noise_study.slurm # skip mining
#   sbatch --export=ALL,SKIP_TO=2 .slurm/run_signal_noise_study.slurm # only AUC
# ---------------------------------------------------------------------------
SKIP_TO="${SKIP_TO:-0}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: SKIP_TO=${SKIP_TO} (steps 1..${SKIP_TO} will be skipped)"

# ---------------------------------------------------------------------------
# Step 1: Mine complexity values (01_mine_complexity_values.py)
# ---------------------------------------------------------------------------
SCRIPT_01="scripts/signal_noise_study/01_mine_complexity_values.py"
if [ "$SKIP_TO" -ge 1 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: SKIPPED (SKIP_TO=${SKIP_TO}) ==="
else
    if [ ! -f "$SCRIPT_01" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Script file not found: $SCRIPT_01"
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: Mining complexity values ==="
    python "$SCRIPT_01" --n-jobs="${SLURM_CPUS_PER_TASK:-1}"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Step 1 (mining) failed - Exit code: ${EXIT_CODE}"
        exit $EXIT_CODE
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1 completed successfully."
fi

# ---------------------------------------------------------------------------
# Step 2: Analyze complexity values (02_analyze_complexity_values.py)
# ---------------------------------------------------------------------------
SCRIPT_02="scripts/signal_noise_study/02_analyze_complexity_values.py"
if [ "$SKIP_TO" -ge 2 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 2: SKIPPED (SKIP_TO=${SKIP_TO}) ==="
else
    if [ ! -f "$SCRIPT_02" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Script file not found: $SCRIPT_02"
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 2: Analyzing complexity values ==="
    python "$SCRIPT_02"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Step 2 (analyze complexity) failed - Exit code: ${EXIT_CODE}"
        exit $EXIT_CODE
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2 completed successfully."
fi

# ---------------------------------------------------------------------------
# Step 3: AUC separability analysis (03_analyze_auc.py)
# ---------------------------------------------------------------------------
SCRIPT_03="scripts/signal_noise_study/03_analyze_auc.py"
if [ ! -f "$SCRIPT_03" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Script file not found: $SCRIPT_03"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 3: AUC separability analysis ==="
python "$SCRIPT_03"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Step 3 (AUC analysis) failed - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3 completed successfully."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All steps finished."
