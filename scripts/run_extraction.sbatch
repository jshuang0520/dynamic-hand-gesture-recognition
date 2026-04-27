#!/usr/bin/env bash
#SBATCH --job-name=shan-etl
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x/shan_project/slurm_logs/etl-%j.out
# #SBATCH --account=<your_allocation>   # uncomment and fill if SLURM complains

set -euo pipefail

# Ensure the shared logs directory exists before running
mkdir -p ${HOME}/shan_project/slurm_logs

# Activate the environment
source scripts/setup_env.sh

echo "Starting Parallel MediaPipe Extraction..."
python mains/01_extract_keypoints.py
echo "Extraction Complete!"