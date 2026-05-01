#!/usr/bin/env bash
#SBATCH --job-name=01_exp1_frozen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/shan_project/slurm_logs/01_exp1-%j.out

set -e

# 1. Load environment variables dynamically
set -a
source .env
set +a

SCRIPT_NAME="02_train_exp1_frozen3d.py"

echo "ℹ️ [INFO] Initializing SLURM Job: ${SCRIPT_NAME}"

# 2. Activate the executing user's environment
source "${ENV_DIR}/bin/activate"

# 3. Setup python paths dynamically
export PYTHONPATH="${PROJECT_ROOT}"

# 4. Enforce Production Configuration for HPC
export ENV="dev"

# 5. Hardware check & Execution
nvidia-smi || true
srun -u python "mains/${SCRIPT_NAME}"

echo "✅ [SUCCESS] SLURM Job Complete: Experiment 1."