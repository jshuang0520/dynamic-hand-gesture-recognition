#!/usr/bin/env bash
#SBATCH --job-name=06_demo
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/%u/scratch.msml640/project_output_resnet_lstm/prod/logs/06_demo-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)
source "${ENV_DIR_CLEAN}/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "🚀 Starting Demo..."
SCRIPT_NAME="06_demo_single_video.py"
srun -u python "mains/${SCRIPT_NAME}" --video prod/raw_data/videos/test/137745 --model exp3  # for demo

echo "✅ [SUCCESS] SLURM Job Complete: Demo for single video."