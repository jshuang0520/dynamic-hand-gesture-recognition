#!/usr/bin/env bash
#SBATCH --job-name=00_subset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/project_output_resnet_lstm/dev/logs/00_prep-%j.out

set -e

# --- 1. Load and Clean Environment ---
# This ensures we pull paths from .env and strip any hidden Windows characters
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "❌ ERROR: .env file missing in $(pwd)"
    exit 1
fi

# Clean paths (removes invisible characters and extra spaces)
ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)
PROJECT_ROOT_CLEAN=$(echo "${PROJECT_ROOT}" | tr -d '\r' | xargs)

# --- 2. Activate Virtual Environment ---
ACTIVATE_PATH="${ENV_DIR_CLEAN}/bin/activate"

if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
    # Append the project root to PYTHONPATH so 'src' and 'utilities' are importable
    export PYTHONPATH="${PROJECT_ROOT_CLEAN}:${PYTHONPATH}"
else
    echo "❌ ERROR: Cannot find virtualenv at: $ACTIVATE_PATH"
    echo "Check if ENV_DIR in .env is correct and that the environment is built."
    exit 1
fi

# --- 3. HPC Execution Config ---
export ENV="prod"
echo "🙂 [INFO] Environment active. Running 00_prepare_subdataset.py..."

# --- 4. Execution ---
# Using srun ensures the task is properly managed by the SLURM step manager
srun -u python mains/00_prepare_subdataset.py

echo "✅ [SUCCESS] SLURM Job Complete: Subset dataset done."