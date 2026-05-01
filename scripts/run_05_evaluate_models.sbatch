#!/usr/bin/env bash
#SBATCH --job-name=05_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# --- CRITICAL FIX: Logs go to scratch! ---
#SBATCH --output=/home/%u/scratch.msml640/project_output_resnet_lstm/dev/logs/05_eval-%j.out

set -e

TARGET_ENV="dev" 

ENV_FILE=".env.${TARGET_ENV}"
if [ -f "$ENV_FILE" ]; then
    echo "ℹ️ [INFO] Loading environment variables from $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a
else
    echo "❌ ERROR: Environment file $ENV_FILE missing in $(pwd)"
    exit 1
fi

ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)
PROJECT_ROOT_CLEAN=$(echo "${PROJECT_ROOT}" | tr -d '\r' | xargs)

ACTIVATE_PATH="${ENV_DIR_CLEAN}/bin/activate"
if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
    export PYTHONPATH="${PROJECT_ROOT_CLEAN}:${PYTHONPATH}"
else
    echo "❌ ERROR: Cannot find virtualenv at: $ACTIVATE_PATH"
    exit 1
fi

SCRIPT_NAME="05_evaluate_models.py"

cd "$PROJECT_ROOT_CLEAN"

echo "ℹ️ [INFO] Initializing SLURM Job: ${SCRIPT_NAME} in ${TARGET_ENV^^} mode"
nvidia-smi || echo "⚠️ Warning: nvidia-smi failed"

srun -u python "mains/${SCRIPT_NAME}"

echo "✅ [SUCCESS] SLURM Job Complete: Evaluation."