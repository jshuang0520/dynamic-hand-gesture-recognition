# Some useful commands in HPC Linux environment

## on your local machine

### connect to HPC
```bash
ssh shhuang@login.zaratan.umd.edu
```

### upload file or folder from your local machine to HPC

- note. `-r` for uploading a folder; without this param for uploading a file
```bash
# pattern like this
scp -r LOCAL_DIR_FULL_PATH HPC_ACCOUNT@login.zaratan.umd.edu:HPC_DESTINATION_DIR_FULL_PATH

# e.g.
scp -r /Users/johnson.huang/Downloads/Meta-Llama-3-8B-Instruct shhuang@login.zaratan.umd.edu:/home/shhuang/scratch/Comparative-Analysis-of-Gun-Violence-Coverage-Across-News-Outlets/hf-cache 
```

### download file or folder from HPC to your local machine
```bash
scp -r shhuang@login.zaratan.umd.edu:/home/shhuang/scratch/Comparative-Analysis-of-Gun-Violence-Coverage-Across-News-Outlets/data/outputs/analysis ~/py_ds/Comparative-Analysis-of-Gun-Violence-Coverage-Across-News-Outlets/data/outputs
```
