#!/usr/bin/env bash
#SBATCH --job-name=05_eval_cpu
#SBATCH --partition=standard      # Swapped to the CPU queue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # Maximize parallel data loading/processing
#SBATCH --mem=64G                 # Generous memory for batch handling
#SBATCH --time=24:00:00           # Massive time bump for CPU backpropagation
#SBATCH --output=/scratch/zt-msml640/user/%u/project_output_resnet_lstm/dev/logs/05_eval-cpu-%j.out


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

# --- 3. Configuration ---
SCRIPT_NAME="05_evaluate_models.py"
export ENV="dev"

echo "ℹ️ [INFO] Initializing SLURM Job: ${SCRIPT_NAME} (CPU MODE)"
echo "🙂 [INFO] Environment active. Running on standard partition..."

# Note: nvidia-smi has been removed as it will fail on CPU nodes.

# --- 4. Execution ---
# srun -u ensures the Python output prints to your log file in real-time
srun -u python "mains/${SCRIPT_NAME}"

echo "✅ [SUCCESS] SLURM Job Complete: Evaluation Phase."