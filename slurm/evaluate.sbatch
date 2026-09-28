#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# One trained run in one environment: slurm/submit.sh slurm/evaluate.sbatch <eval config> <run dir> [--set key=value ...] [--tag name]
# The predictive-sampling cells run in the second environment: CONDA_ENV=fmpcc-mjx slurm/submit.sh slurm/evaluate.sbatch ...
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh "$@"
python scripts/evaluate.py "$1" --run "$2" "${@:3}"
