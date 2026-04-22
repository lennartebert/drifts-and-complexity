#!/bin/bash
#SBATCH --job-name=run_rfc_study
#SBATCH --output=.slurm/logs/%A_%a_rfc_study.out
#SBATCH --error=.slurm/logs/%A_%a_rfc_study.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2

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

logs=(ACCRE BPIC11 BPIC12 BPIC13_cp BPIC13_i BPIC14 BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5 BPIC17 BPIC18 BPIC19 BPIC20_dd BPIC20_id BPIC20_ptc BPIC20_rfp BPIC20_tpd CALL ITHD MOBIS RTFMP SEPSIS)

# Matches scripts/rfc_study/main.py --concept choices
concepts=(variants activities dfrs)

MAIN_SCRIPT=scripts/rfc_study/main.py
if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Missing ${MAIN_SCRIPT} (run sbatch from repository root)" >&2
  exit 1
fi

mkdir -p .slurm/logs
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: SLURM_ARRAY_TASK_ID is not set" >&2
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#concepts[@]} )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0,$((${#concepts[@]}-1))]" >&2
  exit 1
fi

CONCEPT="${concepts[SLURM_ARRAY_TASK_ID]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RFC study for concept=${CONCEPT} across ${#logs[@]} datasets (array_task=${SLURM_ARRAY_TASK_ID}, job_id=${SLURM_JOB_ID:-N/A})"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Smoke-test main entrypoint..."
python "${MAIN_SCRIPT}" --help >/dev/null || {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: main.py --help failed"
  exit 1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: python ${MAIN_SCRIPT} --datasets (all) --concept ${CONCEPT} ..."
python "${MAIN_SCRIPT}" --datasets "${logs[@]}" --concept "${CONCEPT}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done concept ${CONCEPT} for all datasets"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed RFC study for concept=${CONCEPT} (array_task=${SLURM_ARRAY_TASK_ID}, job_id=${SLURM_JOB_ID:-N/A})"
