#!/bin/bash
# slurm/submit.sh <job script> [args...]: submits with the job name of the script and a dated log file
# logs/slurm/<date>/<time>_<name>_<jobid>.log; the arguments are passed to the job script.
if [ -z "$1" ]; then
    echo "usage: $0 <job script> [args...]"
    exit 1
fi
SCRIPT="$1"
shift
NAME="$(basename "${SCRIPT%.*}")"
DIR="logs/slurm/$(date +%Y-%m-%d)"
mkdir -p "$DIR"
sbatch --parsable --job-name="$NAME" --output="$DIR/$(date +%H_%M_%S)_%x_%j.log" --error="$DIR/$(date +%H_%M_%S)_%x_%j.log" \
    --export=ALL "$SCRIPT" "$@"
