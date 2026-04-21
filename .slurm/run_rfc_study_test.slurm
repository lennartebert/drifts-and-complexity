#!/bin/bash
#SBATCH --job-name=run_rfc_study_test
#SBATCH --output=.slurm/logs/%j_rfc_study_test.out
#SBATCH --error=.slurm/logs/%j_rfc_study_test.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

# Smoke test: one small dataset (TEST_BPIC12) before running .slurm/run_rfc_study.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: PYTHONPATH=${PYTHONPATH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Sourcing conda..."
source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Activating conda environment 'drifts-and-complexity'..."
conda activate drifts-and-complexity

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Active conda environment: ${CONDA_DEFAULT_ENV:-NOT_SET}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Python executable: $(which python)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Python version: $(python --version 2>&1)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Working directory: $(pwd)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"

# Truncated TEST_BPIC12 log only (data_dictionary key; see data/real/TEST_BPIC12/)
logs=(TEST_BPIC12)

# Matches scripts/rfc_study/main.py --concept choices
concepts=(variants activities dfrs)

MAIN_SCRIPT=scripts/rfc_study/main.py
if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Missing ${MAIN_SCRIPT} (run sbatch from repository root)" >&2
  exit 1
fi

mkdir -p .slurm/logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RFC study TEST job: ${#logs[@]} dataset(s), ${#concepts[@]} concepts (SLURM_JOB_ID=${SLURM_JOB_ID:-N/A})"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Smoke-test main entrypoint..."
python "${MAIN_SCRIPT}" --help >/dev/null || {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: main.py --help failed"
  exit 1
}

for CONCEPT in "${concepts[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: python ${MAIN_SCRIPT} --datasets (test) --concept ${CONCEPT} ..."
  python "${MAIN_SCRIPT}" --datasets "${logs[@]}" --concept "${CONCEPT}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done concept ${CONCEPT} for test dataset(s)"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed RFC study TEST (TEST_BPIC12, all concepts) job_id=${SLURM_JOB_ID:-N/A}"
