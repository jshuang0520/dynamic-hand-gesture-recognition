#!/usr/bin/env bash
#SBATCH --job-name=shan-etl
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00 
#SBATCH --output=/home/shhuang/shan_project/slurm_logs/etl-%j.out

set -euo pipefail

mkdir -p ${HOME}/shan_project/slurm_logs

# 1. Dynamically load the user's personal config file
source ${HOME}/dynamic-hand-gesture-recognition/.env

# ---------------------------------------------------------
# THE CURE: Scrub SLURM's toxic snapshot and block ghost paths
# ---------------------------------------------------------
unset VIRTUAL_ENV
unset PYTHONHOME
export PYTHONNOUSERSITE=1  # Completely ignores ~/.local ghost packages

# Now that the air is clean, safely activate using your dynamic variable
source ${VENV_DIR}/bin/activate
# ---------------------------------------------------------

echo "Starting Parallel MediaPipe Extraction (16 Workers)..."
export PYTHONPATH="${HOME}/dynamic-hand-gesture-recognition"

# THE LIE DETECTOR: Force the compute node to print its version to your log file
python -c "import mediapipe; print('🤖 COMPUTE NODE MEDIAPIPE VERSION:', mediapipe.__version__)"

# Run the code normally (because we activated cleanly, 'python' is now perfectly safe)
python mains/01_extract_keypoints.py

echo "Extraction Complete!"