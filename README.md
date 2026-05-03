# Spatio-Temporal Modeling for Dynamic Hand Gesture Recognition

This repository contains a high-performance, decoupled pipeline for evaluating video-based gesture recognition. The project benchmarks a **Hybrid ResNet-50 + LSTM** architecture against **3D ResNet** baselines using the 20bn-jester-v1 dataset.

To ensure 100% scientific reproducibility and bypass the I/O bottlenecks of reading thousands of small JPEG files, this pipeline utilizes **Offline Preprocessing**. Videos are uniformly sampled, transformed (noise/pad/crop), and saved as binary tensors before training begins.

---

## 📂 Project Structure

```text
/home/shhuang
project_output_resnet_lstm/               # Global Output Root (Absolute paths in .env)
├── dev/                                  # Development Environment (GPU enabled)
│   ├── logs/                             # SLURM and training execution logs
│   ├── raw_data/
│   │   ├── annotations/                  # Output 00: train.csv, val.csv, test.csv
│   │   └── videos/
│   │       ├── train/                    # 1086/ (contains .jpg frames)
│   │       ├── val/                      # 1037/ (contains .jpg frames)
│   │       └── test/                     # 49884/ (contains .jpg frames)
│   ├── metadata/
│   │   └── data_processed/               # Output 01: Pre-computed .pt tensors (e.g. noise/cropping)
│   └── models/                           # Output 02-04: Trained weight binaries (.pth) for ResNet-3D-18, Finetuned ResNet-3D-18, and Hybrid
└── prod/                                 # Production Environment (Mirror structure)
│
|_ dynamic-hand-gesture-recognition  # Project Root (Code)
    ├── .env                         # User paths (ENV_DIR, OUTPUT_ROOT, etc.)
    ├── configs/
    │   ├── dev/config.yaml          # Subsampled testing parameters
    │   └── prod/config.yaml         # Production/Full-scale parameters
    ├── scripts/
    │   ├── init_dirs.sh             # Directory structure initializer
    │   ├── setup_env.sh             # Environment builder (Python 3.11)
    │   └── run_0X_...sbatch         # SLURM scripts (1-to-1 mapping with mains/)
    ├── mains/
    │   ├── 00_prepare_subdataset.py # Subsampling and CSV split generation
    │   ├── 01_preprocess_data.py    # Temporal sampling and Offline Augmentation
    │   ├── 02_exp1_frozen3d.py
    │   ├── 03_exp2_finetune3d.py
    │   ├── 04_exp3_train_hybrid.py  # Primary Research Entry Point
    │   └── 05_evaluate_models.py    # Benchmark and Classification Reports
    ├── src/
    │   ├── data/
    │   │   ├── loader.py            # High-speed .pt Tensor DataLoader
    │   │   └── transforms.py        # Offline Pad/Crop/Noise logic
    │   ├── models/
    │   │   └── hybrid.py            # Custom ResNet50 + LSTM architecture
    │   └── trainer/
    │       └── engine.py            # Hardware-agnostic training loop engine
    └── utilities/
        ├── config_parser.py         # Multi-env YAML & .env parser
        ├── logger.py                # Timestamped monitoring
        └── reproducibility.py       # Global Seed Locker (20260430)
```

- Note. In this dataset, we only target 4 selected classes: ["Stop Sign", "Swiping Left", "Sliding Two Fingers Down", "Thumb Up"]

---

## Teammate Setup

1. **Initialize Paths:** Configure your `.env` file based on `.env.example`.
2. **Create Directories:** Initialize the output folder structure.
```bash
chmod +x scripts/init_dirs.sh
./scripts/init_dirs.sh
```
3. **Build Environment:** Build the Python 3.11 venv at the path specified in your `.env`.
```bash
bash scripts/setup_env.sh
```

---

## Execution Commands

### 1. Data Preparation Pipeline
```bash
# # Generate CSV splits (train/val/test) based on current raw_data
# sbatch scripts/run_00_prepare_subdataset.sbatch

# Preprocess (Temporal sampling + Offline Noise -> .pt tensors)
sbatch scripts/run_01_preprocess_data.sbatch
sbatch scripts/run_01_preprocess_data_cpu.sbatch
```

### 2. Model Training Experiments
```bash
# Train Baseline 1 (Frozen 3D ResNet)
sbatch scripts/run_02_exp1_frozen3d.sbatch
sbatch scripts/run_02_exp1_frozen3d_cpu.sbatch

# Train Baseline 2 (Fine-tuned 3D ResNet)
sbatch scripts/run_03_exp2_finetune3d.sbatch
sbatch scripts/run_03_exp2_finetune3d_cpu.sbatch

# Train Proposed Model (Hybrid ResNet+LSTM)
sbatch scripts/run_04_exp3_train_hybrid.sbatch
sbatch scripts/run_04_exp3_train_hybrid_cpu.sbatch
```

### 3. Final Evaluation
```bash
# Benchmark all models and generate classification reports
sbatch scripts/run_05_evaluate_models.sbatch
sbatch scripts/run_05_evaluate_models_cpu.sbatch
```

## Monitoring Jobs

```bash
# Check Job Status
squeue -u $USER

# View logs
tail -f $(find ~/project_output_resnet_lstm/dev/logs/ -name "*-$(squeue -u $USER -h -t RUNNING -n 1 -o "%i").out")  # latest job only
squeue -u $USER -h -t RUNNING -o "%i" | xargs -I {} find ~/project_output_resnet_lstm/dev/logs/ -name "*-{}.out" | xargs -r tail -f
# View Real-time Logs
tail -f ~/project_output_resnet_lstm/dev/logs/01-JOB_ID.out

# Check available GPUs
sinfo -p gpu -t idle -o "%n %G"

# Cancel a Job
scancel JOB_ID
```

## For Development Phase
```bash
# init settings
bash scripts/init_dirs.sh
cp -r jester_subsampled_data/dev/raw_data/ project_output_resnet_lstm/dev/

# start testing dev scripts
sbatch scripts/run_01_preprocess_data_cpu.sbatch
sbatch scripts/run_02_exp1_frozen3d_cpu.sbatch
sbatch scripts/run_03_exp2_finetune3d_cpu.sbatch
sbatch scripts/run_04_exp3_train_hybrid_cpu.sbatch
sbatch scripts/run_05_evaluate_models_cpu.sbatch

# check space
du -ah . --max-depth=1 | sort -rh | head -n 10


# prod run
# 1. Run the first job and capture its Job ID
JOB1=$(sbatch --parsable scripts/run_01_preprocess_data.sbatch)
# 2. Run the next 3 jobs in parallel, waiting for JOB1 to finish successfully
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_02_exp1_frozen3d.sbatch)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_03_exp2_finetune3d.sbatch)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_04_exp3_train_hybrid.sbatch)
# 3. Run the final job only after all three middle jobs (JOB2, JOB3, and JOB4) finish successfully
sbatch --dependency=afterok:$JOB2:$JOB3:$JOB4 scripts/run_05_evaluate_models.sbatch

# # when run_01_preprocess_data is already done successfully
# JOB2=$(sbatch --parsable scripts/run_02_exp1_frozen3d.sbatch)
# JOB3=$(sbatch --parsable scripts/run_03_exp2_finetune3d.sbatch)
# JOB4=$(sbatch --parsable scripts/run_04_exp3_train_hybrid.sbatch)
# sbatch --dependency=afterok:$JOB2:$JOB3:$JOB4 scripts/run_05_evaluate_models.sbatch


squeue --me
sinfo -p gpu -t idle -o "%n %G"
scancel --me

login-1:~$ find project_output_resnet_lstm/ -type d | awk -F/ 'count[$(NF-1)]++ < 10' | sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'

 project_output_resnet_lstm
 |-
 |-dev
 |---models
 |---metadata
 |-----data_processed
 |---logs
 |---raw_data
 |-----annotations
 |-----videos
 |-------train
 |---------72057
 |---------134871
 |---------23472
 |---------1086
 |---------21313
 |---------116268
 |---------140191
 |---------9808
 |-------val
 |---------33871
 |---------73400
 |---------124343
 |---------1037
 |-------test
 |---------108558
 |---------142573
 |---------52722
 |---------49884
 |-prod
 |---models
 |---metadata
 |-----data_processed
 |---logs
 |---raw_data
 |-----annotations
 |-----videos
 |-------train
 |-------val
 |-------test
```

---

## Execution Pipeline Details

### Phase 1: Data Preparation
1. **Subsampling (Step 00):** Extracts a manageable subset of the Jester dataset and generates splits.
2. **Preprocessing (Step 01):** - **Temporal Sampling:** Uses uniform distribution (linspace) to capture the full gesture motion into 16 frames.
   - **Offline Augmentation:** Applies padding, center-cropping, and Gaussian noise.
   - **Binary I/O:** Saves results as `.pt` files for ultra-fast GPU training.

### Phase 2: Training Experiments
3. **Experiment 1 (Frozen 3D):** Trains only the classification head of an R3D-18 model.
4. **Experiment 2 (Fine-Tuned 3D):** Updates the entire 3D backbone.
5. **Experiment 3 (Hybrid ResNet+LSTM):** Evaluates the proposed spatial-temporal architecture.

### Phase 3: Benchmarking
6. **Evaluation (Step 05):** Pulls best weights from the `models/` directory (as defined in `config.yaml`) and generates comparative metrics.

---

## Core Features
- **Deterministic Experiments:** Moving noise and transforms to an offline step ensures all models see identical input data.
- **Dynamic Temporal Sampling:** Our sampler ensures the model always receives 16 frames representing the full duration of the action, solving variable-length issues.
- **Zero-Hardcode Policy:** All paths and weights are dynamically resolved via `.env` and YAML configurations.
