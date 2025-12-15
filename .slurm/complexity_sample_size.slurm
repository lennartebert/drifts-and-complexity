#!/bin/bash
#SBATCH --job-name=complexity_sample_size
#SBATCH --output=.scripts/logs/%A/log_%A_%a.out
#SBATCH --error=.scripts/logs/%A/log_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --spread-job
#SBATCH --distribution=block:block
#SBATCH --time=3-00:00:00 # increase MaxTime to 3 days instead of 4 hours

source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate drifts-and-complexity

echo "Python path:"
which python

echo "Python executable:"
python -c "import sys; print(sys.executable)"

echo "PyYAML version:"
python -c "import yaml; print(yaml.__version__)"

python complexity_sample_size_correlation_analysis.py