#!/usr/bin/env bash
#SBATCH --job-name=03_exp3_hybrid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/shan_project/slurm_logs/03_exp3-%j.out

set -e

# 1. Load environment variables dynamically
set -a
source .env
set +a

echo "ℹ️ [INFO] Initializing SLURM Job: 03_train_exp3_hybrid"

# 2. Activate the executing user's environment
source "${ENV_DIR}/bin/activate"

# 3. Setup python paths dynamically
export PYTHONPATH="${PROJECT_ROOT}"

# 4. Enforce Production Configuration for HPC
export ENV="prod"

# 5. Hardware check & Execution
nvidia-smi || true
srun -u python mains/03_train_exp3_hybrid.py

echo "✅ [SUCCESS] SLURM Job Complete: Experiment 3."