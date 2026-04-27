# dynamic-hand-gesture-recognition

## [!IMPORTANT] project directory structures

```
/home/shhuang
|_ dynamic-hand-gesture-recognition (code)
|_ manually_downloaded_pkg
|_ .cache (manually downloaded models, like InceptionV3)
|_ jester_uncompressed (our raw dataset; the sub-dataset from the original huge dataset; already train-test split)
|_ shan_project (shared data/output directory, including metadata and logs; this shared data folder is not under the project root because the project root is only for the code, and it will be frequently re-uploaded from local machine to HPC)
```

## data preprocessing

### entry point
```
01_generate_14k_subsample.ipynb
```

Since the original file size (22.8 GB after compression) is too large, there is a bottle neck for us to upload it to HPC. 
We subset the dataset as the following spec:
- overall 14,000 videos (folders) in 14 classes (dynamic gestures)
    - training set: 11,200 videos (with 800 videos per class)
    - validation set: 1,400 videos (with 100 videos per class)
    - test set: 1,400 videos (with 100 videos per class)

## Code Entry Points & Execution Flow

Our project utilizes a strictly decoupled, modular pipeline to ensure performance on the Zaratan HPC cluster. Execution is handled via the following entry points:

### 1. Python Main Scripts (`mains/`)
* **`01_extract_keypoints.py`**: The offline ETL pipeline. Uses `multiprocessing` across 8 CPU cores to read raw JESTER frames, extract 21 geometric hand keypoints via MediaPipe, and cache them as lightweight PyTorch tensors (`.pt`). This completely decouples CPU-bound extraction from GPU-bound training, preventing major bottlenecks.
* **`02_train.py`**: *(Pending Phase 2)* The Distributed Data Parallel (DDP) training loop. Handles PyTorch AMP (Automatic Mixed Precision) and auto-resume checkpointing to gracefully survive HPC time limits.
* **`03_evaluate.py`**: *(Pending Phase 3)* The evaluation and metrics script. Loads the `best_model.pth` and generates classification reports and confusion matrices to analyze model performance on mirror-gestures.

### 2. SLURM Bash Scripts (`scripts/`)
* **`setup_env.sh`**: Universal environment manager. Automatically builds a Python 3.11 virtual environment, installs HPC-optimized CUDA 12.6 wheels, and dynamically resolves paths so any team member can run it natively via `source scripts/setup_env.sh`.
* **`run_extraction.sbatch`**: Submits the `01_extract_keypoints.py` script to the HPC CPU partition.
* **`run_train_baseline.sbatch`**: *(Pending Phase 2)* Submits the Experiment 1 (InceptionV3 pure CNN baseline) job to the 4x H100 GPU partition.
* **`run_train_shan.sbatch`**: *(Pending Phase 2)* Submits the Experiment 2 (Full SHAN Multi-Modal Architecture) job to the 4x H100 GPU partition.

---

## plans


```
# src/models/temporal_tracker.py
"""
HOLD BACK: Vision Transformer Encoder.
Do not implement until InceptionV3 spatial tokens ([Batch, 16, 2048]) are verified.
"""

# src/models/skeleton_encoder.py
"""
HOLD BACK: MediaPipe Structural MLP.
Do not implement until the offline extraction finishes and we verify the .pt files.
"""

# src/models/fusion_brain.py
"""
HOLD BACK: Late-Fusion Concatenation & Classifier.
Wait for temporal_tracker and skeleton_encoder dimensions to be finalized.
"""

# src/models/shan.py
"""
HOLD BACK: The SHAN Wrapper.
Do not implement until all individual components (Lego blocks) are tested.
"""

# src/trainer/engine.py
"""
HOLD BACK: Training/Validation Engine.
Depends on the final output shape of the SHAN model.
"""

# mains/02_train.py
"""
HOLD BACK: DDP Training Entry Point.
Wait for the Trainer Engine to be built.
"""

# mains/03_evaluate.py
"""
HOLD BACK: Inference & Metrics calculation.
Cannot be written until a best_model.pth is generated.
"""
```

---

## Miscellaneous

### upload file or folder from your local machine to HPC

- `scp`

```bash
scp -r /Users/johnson.huang/py_ds/dynamic-hand-gesture-recognition shhuang@login.zaratan.umd.edu:/home/shhuang
```

- [!NOTE] `rsync` to exclude some files

```bash
rsync -avz --exclude '.git' /Users/johnson.huang/py_ds/dynamic-hand-gesture-recognition shhuang@login.zaratan.umd.edu:/home/shhuang/
```

### download models manually and upload to HPC

- checkout this dir `/home/shhuang/.cache/torch/hub/checkpoints` on HPC

```bash
mkdir -p ~/.cache/torch/hub/checkpoints/

wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth -P ~/.cache/torch/hub/checkpoints/

scp /Users/johnson.huang/.cache/torch/hub/checkpoints/inception_v3_google-1a9a5a14.pth shhuang@login.zaratan.umd.edu:/home/shhuang/.cache/torch/hub/checkpoints
```

- load in python
```python
import torch
from torchvision import models

# Load model and point to the local file if necessary
model = models.inception_v3(pretrained=True)
model.eval()
```

- download the packages manually and then upload to HPC
```bash
# Step 1: Download Packages (On Internet-Connected Machine)
mkdir -p /Users/johnson.huang/py_ds/manually_downloaded_pkg/torch_offline
cd /Users/johnson.huang/py_ds/manually_downloaded_pkg/torch_offline
# Download exactly Python 3.11 and Linux x86_64 for HPC (Red Hat VERSION=8.10)
# NOTE. even if not using python 3.11 environment, this works, too!
pip download \
  --index-url https://download.pytorch.org/whl/cu126 \
  --only-binary=:all: \
  --platform manylinux_2_28_x86_64 \
  --python-version 311 \
  --implementation cp \
  --abi cp311 \
  torch==2.9.1+cu126 \
  torchvision==0.24.1+cu126 \
  torchaudio==2.9.1+cu126


# Step 2: Transfer Files to HPC
scp -r /Users/johnson.huang/py_ds/manually_downloaded_pkg shhuang@login.zaratan.umd.edu:/home/shhuang
```


## 11. HPC Commands

### Job Submission

```bash
# Submit training job
sbatch run_train.sbatch

# Output: Submitted batch job JOBID
```

### Monitoring

- note. specify all jobids and tail those corresponding logs

```bash
squeue -u $USER -h -t RUNNING -o "slurm-%i.out" | xargs -r tail -f
```

```bash
# Check job status
squeue -u $USER

# Check available GPUs
sinfo -p gpu -t idle -o "%n %G"

# Monitor log in real-time
tail -f slurm-JOBID.out

# Check file sizes
du -h --max-depth=1
```


### Job Management

```bash
# Cancel job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Job details
scontrol show job JOBID
```

### HPC Configuration

Current setup (`run_train.sbatch`):
```bash
#SBATCH --gres=gpu:h100:4      # 4× H100 GPUs
#SBATCH --cpus-per-task=8      # 8 CPU cores
#SBATCH --mem=48G              # 48GB RAM
#SBATCH --time=08:00:00        # 8 hour limit
```

**Optimizations enabled**:
- cuDNN benchmarking (faster convolutions)
- TF32 precision (faster on H100/A100)
- Parallel data loading (4 workers)
- Pinned memory (faster GPU transfer)
- Automatic Mixed Precision (AMP)

---

## temp

actions:

1.⁠ ⁠build python env on HPC

- Test the Env: Run `source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh`

- After package updates: Run `source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh --update`

2.⁠ ⁠⁠read the images, turn it into tensors that are recognizable by the models like InceptionV3 or ResNet for the recognition task

- Test the Model: Run `python ~/dynamic-hand-gesture-recognition/src/models/cnn_baseline.py`

3.⁠ ⁠run InceptionV3 as our baseline (Experiment 1) - refer to 01_generate_14k_subsample.ipynb last cells: train a simple inception baseline

- Start the ETL: Run `sbatch ~/dynamic-hand-gesture-recognition/scripts/run_extraction.sbatch`
