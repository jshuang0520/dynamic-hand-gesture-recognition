#!/usr/bin/env bash
#SBATCH --job-name=00_subset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/shan_project/slurm_logs/00_prep-%j.out

set -e

# 1. Load environment variables dynamically
set -a
source .env
set +a

echo "ℹ️ [INFO] Initializing SLURM Job: 00_prepare_datasets"

# 2. Activate the executing user's environment
source "${ENV_DIR}/bin/activate"

# 3. Setup python paths dynamically
export PYTHONPATH="${PROJECT_ROOT}"

# 4. Environment Configuration for HPC
export ENV="dev"

# 5. Execution
srun -u python mains/00_prepare_subdataset.py

echo "✅ [SUCCESS] SLURM Job Complete: Subset dataset done."