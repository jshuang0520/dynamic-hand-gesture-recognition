#!/usr/bin/env bash
#SBATCH --job-name=shan-etl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/etl-%j.out

set -euo pipefail
source scripts/setup_env.sh
python mains/01_extract_keypoints.py