# dynamic-hand-gesture-recognition

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

---

## temp

actions:

1.⁠ ⁠build python env on HPC

- Test the Env: Run `source scripts/setup_env.sh`

2.⁠ ⁠⁠read the images, turn it into tensors that are recognizable by the models like InceptionV3 or ResNet for the recognition task

- Test the Model: Run `python src/models/cnn_baseline.py`

3.⁠ ⁠run InceptionV3 as our baseline (Experiment 1) - refer to 01_generate_14k_subsample.ipynb last cells: train a simple inception baseline

- Start the ETL: `Run sbatch scripts/run_extraction.sbatch`
