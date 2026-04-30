```bash
# activate virtual env
source ~/gesture_rec_env/bin/activate

source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh && \
python ~/dynamic-hand-gesture-recognition/src/models/cnn_baseline.py && \
sbatch /home/shhuang/dynamic-hand-gesture-recognition/scripts/run_training.sbatch && \
sbatch ~/dynamic-hand-gesture-recognition/scripts/run_extraction.sbatch && \
squeue -u $USER --start >> job_estimated_wait_time.txt && \
ls -t ${HOME}/shan_project/slurm_logs/*.out | head -1 | xargs tail -f


# Check job status
squeue -u $USER
squeue -u $USER --start >> job_estimated_wait_time.txt
ls -t ${HOME}/shan_project/slurm_logs/*.out | head -1 | xargs tail -f

# Check available GPUs
sinfo -p gpu -t idle -o "%n %G"


# To "follow" the the newest/latest updated log file in real-time
ls -t ${HOME}/shan_project/slurm_logs/*.out | head -1 | xargs tail -f


# check file size - show the top 10 largest files
du -ah . --max-depth=1 | sort -rh | head -n 10


# Cancel all your jobs
scancel -u $USER
```