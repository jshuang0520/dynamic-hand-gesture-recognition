#!/usr/bin/env bash
#SBATCH --job-name=shan-cnn-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1          # FIX: Explicitly request 1 H100 GPU
#SBATCH --cpus-per-task=8          # 8 CPUs is perfect for feeding 1 GPU
#SBATCH --mem=32G                  # 32GB RAM
#SBATCH --time=08:00:00            
#SBATCH --output=/home/shhuang/shan_project/slurm_logs/train-%j.out

PROJECT_ROOT="${HOME}/dynamic-hand-gesture-recognition"

set -euo pipefail

mkdir -p ${HOME}/shan_project/slurm_logs

# 1. Dynamically load the config
source ${PROJECT_ROOT}/.env

# ---------------------------------------------------------
# THE CURE: Scrub SLURM's toxic snapshot and block ghost paths
# ---------------------------------------------------------
unset VIRTUAL_ENV
unset PYTHONHOME
export PYTHONNOUSERSITE=1  

# Activate the environment cleanly
source ${VENV_DIR}/bin/activate
# ---------------------------------------------------------

# ---------------------------------------------------------
# GPU OPTIMIZATIONS
# ---------------------------------------------------------
# Prevent CPU thread fighting
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Prevent PyTorch memory fragmentation on large batches
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.9

# Enable Tensor Cores for massive speedups on A100/H100 GPUs
export NVIDIA_TF32_OVERRIDE=1
# ---------------------------------------------------------

echo "Starting PyTorch CNN Training Phase..."
export PYTHONPATH="${PROJECT_ROOT}"

# THE SANITY CHECK: Log the GPU status before PyTorch boots
nvidia-smi || true
python -c "
import torch
print('🤖 CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🤖 GPU Device:', torch.cuda.get_device_name(0))
"

# Run the PyTorch training code!
# We use 'srun -u' here to force unbuffered logs so you can watch your epochs train live
srun -u python "${PROJECT_ROOT}/src/models/cnn_baseline.py"

echo "Training Complete!"
