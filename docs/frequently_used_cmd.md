```bash
source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh

# Check job status
squeue -u $USER

# Check available GPUs
sinfo -p gpu -t idle -o "%n %G"


# To "follow" the the newest/latest updated log file in real-time
ls -t ${HOME}/shan_project/slurm_logs/*.out | head -1 | xargs tail -f


# check file size - show the top 10 largest files
du -ah . --max-depth=1 | sort -rh | head -n 10


# Cancel all your jobs
scancel -u $USER
```