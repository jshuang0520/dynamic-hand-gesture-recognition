#!/usr/bin/env bash
#SBATCH --job-name=shan-etl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Notice the path below: logs go to the shared folder, not the code repo!
#SBATCH --output=%x/shan_project/slurm_logs/etl-%j.out

set -euo pipefail

# Ensure the shared logs directory exists before running
mkdir -p ${HOME}/shan_project/slurm_logs

# Activate the environment (Assuming you submit this from dynamic-hand-gesture-recognition)
source scripts/setup_env.sh

echo "Starting Parallel MediaPipe Extraction..."
python mains/01_extract_keypoints.py
echo "Extraction Complete!"