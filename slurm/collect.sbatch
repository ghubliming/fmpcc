#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
# Quadrotor demonstrations of one scene (CPU only): slurm/submit.sh slurm/collect.sbatch <corridor|s_curve> [--trials N] [--seed S]
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/common.sh "$@"
export MUJOCO_GL=disable
python scripts/collect_uav_demonstrations.py --scene "$1" "${@:2}"
