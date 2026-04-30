#!/usr/bin/env bash
#SBATCH --job-name=exp3_hybrid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/shhuang/hybrid_project/slurm_logs/exp3-%j.out

echo "[INFO] Initializing SLURM Job for Experiment 3 (Hybrid Model)"

# 1. Activate the isolated Python 3.11 environment
source ${HOME}/SHELL.msml640/bin/activate

# 2. Setup paths
export PYTHONPATH="${HOME}/dynamic-hand-gesture-recognition"

# 3. Hardware Sanity Check
nvidia-smi || true

# 4. Execute the Python Payload
srun -u python mains/03_train_exp3_hybrid.py

echo "[SUCCESS] SLURM Job Complete."