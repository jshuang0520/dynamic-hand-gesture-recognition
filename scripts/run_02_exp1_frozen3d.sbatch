#!/usr/bin/env bash
#SBATCH --job-name=01_exp1_frozen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/scratch.msml640/project_output_resnet_lstm/prod/logs/02_exp1-%j.out

set -e

# --- Define which environment we are running ---
# Change this to "dev" when you want to run the micro-test!
TARGET_ENV="prod" 

# --- 1. Load and Clean Environment ---
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
SCRIPT_NAME="02_train_exp1_frozen3d.py"

# --- CRITICAL FIX: MOVE TO PROJECT ROOT ---
cd "$PROJECT_ROOT_CLEAN"

echo "ℹ️ [INFO] Initializing SLURM Job: ${SCRIPT_NAME} in ${TARGET_ENV^^} mode"
nvidia-smi || echo "⚠️ Warning: nvidia-smi failed"

# --- 4. Execution ---
srun -u python "mains/${SCRIPT_NAME}"

echo "✅ [SUCCESS] SLURM Job Complete."