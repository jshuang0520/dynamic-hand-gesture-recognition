#!/usr/bin/env bash
#SBATCH --job-name=01_prep_cpu
#SBATCH --partition=standard      # Use 'standard' or 'debug' instead of 'gpu'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # High CPU count for parallel image processing
#SBATCH --mem=64G                 # High RAM for buffer handling
#SBATCH --time=04:00:00           
#SBATCH --output=/home/%u/project_output_resnet_lstm/dev/logs/01-cpu-%j.out

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

# --- 3. Configuration & Execution ---
SCRIPT_NAME="01_preprocess_data.py"
export ENV="dev"

echo "ℹ️ [INFO] Initializing SLURM Job: ${SCRIPT_NAME} (CPU MODE)"
echo "🙂 [INFO] Environment active. Running on standard partition..."

# --- 4. Execution ---
# Note: srun -u ensures unbuffered output so you see logs in real-time
srun -u python "mains/${SCRIPT_NAME}"

echo "✅ [SUCCESS] SLURM Job Complete: Preprocess data done."