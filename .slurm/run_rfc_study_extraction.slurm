#!/bin/bash
#SBATCH --job-name=rfc_extract
#SBATCH --output=.slurm/logs/%A_%a_rfc_extract.out
#SBATCH --error=.slurm/logs/%A_%a_rfc_extract.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-24

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Sourcing conda..."
source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: Activating conda environment 'drifts-and-complexity'..."
conda activate drifts-and-complexity

logs=(ACCRE BPIC11 BPIC12 BPIC13_cp BPIC13_i BPIC14 BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5 BPIC17 BPIC18 BPIC19 BPIC20_dd BPIC20_id BPIC20_ptc BPIC20_rfp BPIC20_tpd CALL ITHD MOBIS RTFMP SEPSIS)
concepts=(variants activities dfrs)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: SLURM_ARRAY_TASK_ID is not set" >&2
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#logs[@]} )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0,$((${#logs[@]}-1))]" >&2
  exit 1
fi

DATASET="${logs[SLURM_ARRAY_TASK_ID]}"

EXTRACT_SCRIPT=scripts/rfc_study/extract_attachments.py
if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Missing ${EXTRACT_SCRIPT} (run sbatch from repository root)" >&2
  exit 1
fi

mkdir -p .slurm/logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting extraction for dataset=${DATASET} (array_task=${SLURM_ARRAY_TASK_ID}, job_id=${SLURM_JOB_ID:-N/A})"

for CONCEPT in "${concepts[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running extraction: dataset=${DATASET}, concept=${CONCEPT}"
  python "${EXTRACT_SCRIPT}" --datasets "${DATASET}" --concepts "${CONCEPT}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed extraction for dataset=${DATASET} (array_task=${SLURM_ARRAY_TASK_ID})"
