#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# One model, one seed: slurm/submit.sh slurm/train.sbatch <train config> <model> <seed> [--set key=value ...] [--resume] [--wandb]
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh "$@"
python scripts/train.py "$1" --model "$2" --seed "$3" "${@:4}"
