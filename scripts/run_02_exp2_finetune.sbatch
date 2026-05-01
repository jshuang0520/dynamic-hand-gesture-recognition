#!/usr/bin/env bash
#SBATCH --job-name=02_exp2_tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/shan_project/slurm_logs/02_exp2-%j.out

set -e

# 1. Load environment variables dynamically
set -a
source .env
set +a

echo "ℹ️ [INFO] Initializing SLURM Job: 02_train_exp2_finetune"

# 2. Activate the executing user's environment
source "${ENV_DIR}/bin/activate"

# 3. Setup python paths dynamically
export PYTHONPATH="${PROJECT_ROOT}"

# 4. Enforce Production Configuration for HPC
export ENV="prod"

# 5. Hardware check & Execution
nvidia-smi || true
srun -u python mains/02_train_exp2_finetune.py

echo "✅ [SUCCESS] SLURM Job Complete: Experiment 2."