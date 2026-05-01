#!/usr/bin/env bash
#SBATCH --job-name=01_exp1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/scratch.msml640/project_output_resnet_lstm/prod/logs/02_exp1-%j.out

set -euo pipefail

# 1. Navigate to the folder you submitted this from (Project Root)
cd "$SLURM_SUBMIT_DIR"

# 2. Load the .env file dynamically
if [ -f ".env" ]; then
    echo "ℹ️ [INFO] Loading .env file..."
    set -a; source .env; set +a
else
    echo "❌ ERROR: .env file missing in $(pwd)"
    exit 1
fi

# Clean the path just in case of hidden carriage returns
ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)

# 3. Activate environment dynamically
ACTIVATE_PATH="${ENV_DIR_CLEAN}/bin/activate"
if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
    export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH}"
    echo "✅ [SUCCESS] Virtual environment activated at $ENV_DIR_CLEAN"
else
    echo "❌ ERROR: Cannot find virtualenv at: $ACTIVATE_PATH"
    exit 1
fi

# 4. Tell Python we want the PROD config
export ENV="prod"

# 5. Check hardware
echo "ℹ️ [INFO] Starting Experiment 1 Training..."
nvidia-smi || true

# 6. Run the script
srun -u python3 mains/02_exp1_frozen3d.py

echo "✅ [SUCCESS] SLURM Job Complete."