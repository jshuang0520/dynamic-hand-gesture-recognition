#!/usr/bin/env bash
#SBATCH --job-name=05_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/scratch.msml640/project_output_resnet_lstm/prod/logs/05_eval-%j.out

set -euo pipefail

# 1. Navigate to Project Root
cd "$SLURM_SUBMIT_DIR"

# 2. Load .env dynamically
if [ -f ".env" ]; then
    echo "ℹ️ [INFO] Loading .env file..."
    set -a; source .env; set +a
else
    echo "❌ ERROR: .env file missing in $(pwd)"
    exit 1
fi

ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)

# 3. Activate virtual environment
ACTIVATE_PATH="${ENV_DIR_CLEAN}/bin/activate"
if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
    export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH}"
    echo "✅ [SUCCESS] Virtual environment activated at $ENV_DIR_CLEAN"
else
    echo "❌ ERROR: Cannot find virtualenv at: $ACTIVATE_PATH"
    exit 1
fi

# 4. Target Environment
export ENV="prod"

# 5. Check hardware
echo "ℹ️ [INFO] Starting Model Evaluation..."
nvidia-smi || true

# 6. Execute Python Script
srun -u python3 mains/05_evaluate_models.py

echo "✅ [SUCCESS] SLURM Job Complete: Evaluation."