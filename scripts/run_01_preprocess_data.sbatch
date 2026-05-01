#!/usr/bin/env bash
#SBATCH --job-name=01_preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/project_output_resnet_lstm/dev/logs/01-%j.out

set -a; source .env; set +a
source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}"
export ENV="dev"

srun -u python mains/01_preprocess_data.py