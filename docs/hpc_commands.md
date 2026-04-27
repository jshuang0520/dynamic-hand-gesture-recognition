# Some useful commands in HPC Linux environment

------------------------------------------------------------------------------------------

## On your local machine

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

------------------------------------------------------------------------------------------

## As long as in Linux environment


### ls

- only list dirs but not files, and each sub-directory layer, only show 10 dirs from them
```bash
find . -maxdepth 3 -type d | awk -F/ '{parent=$0; sub(/\/[^\/]+$/, "", parent)} count[parent]++ < 10' | sed -e 's/[^-][^\/]*\// |/g' -e 's/|/|--/g'
```
```
.
 |--jester_splits
 |--20bn-jester-v1
 |-- |--119126
 |-- |--89507
 |-- |--29296
 |-- |--106212
 |-- |--55256
 |-- |--72795
 |-- |--16876
 |-- |--142223
 |-- |--73026
 |-- |--19297
```


### file size

- show the top 10 largest files
```bash
du -ah . | sort -rh | head -n 10
```

- show number of dirs where one is larger than a specific size
```bash
du -h --max-depth=1 --threshold=20K | wc -l
```

- show number of files where one is larger than a specific size
```bash
find . -type f -size +5k | wc -l

The `-size` flag supports several units of measurement: 
- `c`: Bytes
- `k`: Kilobytes (1024 bytes)
- `M`: Megabytes (1024 kilobytes)
- `G`: Gigabytes (1024 megabytes) 
```

### change file / dir permission

- grants full read, write, and execute permissions to everyone: the owner (u), the group (g), and all other users (o). 

- rwx are $2^2, 2^1, 2^0$, respectively

- 7 means $2^2 + 2^1 + 2^0$, with all r (read), w (write), x (execute) permissions

- 754 means: owner (u) has rwx permission; group (g) has rx permission; all other users (o) has r permission

```bash
# pattern like this
chmod  7  7  7 YOUR_DIR
      (u)(g)(o)

# e.g.
chmod 755 /home/shhuang
```
