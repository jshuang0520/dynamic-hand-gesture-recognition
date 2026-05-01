#!/usr/bin/env bash
#SBATCH --job-name=04_evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/shan_project/slurm_logs/05_eval-%j.out

set -e

# --- 1. Load and Clean Environment ---
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "❌ ERROR: .env file missing in $(pwd)"
    exit 1
fi

# Clean paths (strips hidden characters/carriage returns)
ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)
PROJECT_ROOT_CLEAN=$(echo "${PROJECT_ROOT}" | tr -d '\r' | xargs)

# --- 2. Activate Virtual Environment ---
ACTIVATE_PATH="${ENV_DIR_CLEAN}/bin/activate"

if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
    export PYTHONPATH="${PROJECT_ROOT_CLEAN}:${PYTHONPATH}"
else
    echo "❌ ERROR: Cannot find virtualenv at: $ACTIVATE_PATH"
    exit 1
fi

# --- 3. Configuration & Hardware Check ---
SCRIPT_NAME="05_evaluate_models.py"
export ENV="dev"

echo "ℹ️ [INFO] Initializing SLURM Job: ${SCRIPT_NAME}"
echo "🙂 [INFO] Environment active. Checking Hardware..."

# Hardware check (Optional but good for GPU jobs)
nvidia-smi || echo "⚠️ Warning: nvidia-smi failed, checking execution anyway..."

# --- 4. Execution ---
srun -u python "mains/${SCRIPT_NAME}"

echo "✅ [SUCCESS] SLURM Job Complete: Evaluation Phase."