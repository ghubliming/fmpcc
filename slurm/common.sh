# Sourced by every job script. Cluster-specific values first.
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fmpcc}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_api_key}"

set -e
echo "========================================"
echo "JOB:  ${SLURM_JOB_NAME:-local} ${SLURM_JOB_ID:-}"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "GIT:  $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "ARGS: $*"
echo "========================================"
trap 'echo "JOB END: $(date)"' EXIT

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export MPLBACKEND=agg
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
    export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
    echo "GPU:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
    if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
        echo "EGL device differs from the allocated GPU -- aborting"
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi
