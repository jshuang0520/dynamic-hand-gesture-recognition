# Spatio-Temporal Modeling for Dynamic Hand Gesture Recognition

This repository contains a high-performance, decoupled pipeline for evaluating video-based gesture recognition. The project benchmarks a **Hybrid ResNet-50 + LSTM** architecture against **3D ResNet** baselines using the 20bn-jester-v1 dataset.

To ensure 100% scientific reproducibility and bypass the I/O bottlenecks of reading thousands of small JPEG files, this pipeline utilizes **Offline Preprocessing**. Videos are uniformly sampled, transformed (noise/pad/crop), and saved as binary tensors before training begins.

---

## 📂 Project Structure

```text
/home/shhuang
|_ project_output_resnet_lstm        # Global Output Root (Absolute paths in .env)
│   ├── dev/                         # Development Environment (GPU enabled)
│   │   ├── raw_data/                # Output 00: Subsampled .jpg folder structures
│   │   ├── metadata/
│   │   │   ├── splits/              # Output 00: train.csv, val.csv, test.csv
│   │   │   └── data_processed/      # Output 01: Pre-computed .pt tensors
│   │   ├── models/                  # Output 02-04: Trained weight binaries (.pth)
│   │   └── logs/                    # SLURM and training execution logs
│   └── prod/                        # Production Environment (Mirror structure)
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
    │   ├── 02_train_exp1_frozen3d.py
    │   ├── 03_train_exp2_finetune.py
    │   ├── 04_train_exp3_hybrid.py   # Primary Research Entry Point
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
# Generate CSV splits (train/val/test) based on current raw_data
sbatch scripts/run_00_prepare_subdataset.sbatch

# Preprocess (Temporal sampling + Offline Noise -> .pt tensors)
sbatch scripts/run_01_preprocess_data.sbatch
```

### 2. Model Training Experiments
```bash
# Train Baseline 1 (Frozen 3D ResNet)
sbatch scripts/run_02_train_exp1_frozen3d.sbatch

# Train Baseline 2 (Fine-tuned 3D ResNet)
sbatch scripts/run_03_train_exp2_finetune.sbatch

# Train Proposed Model (Hybrid ResNet+LSTM)
sbatch scripts/run_04_train_exp3_hybrid.sbatch
```

### 3. Final Evaluation
```bash
# Benchmark all models and generate classification reports
sbatch scripts/run_05_evaluate_models.sbatch
```

## Monitoring Jobs

```bash
# Check Job Status
squeue -u shhuang

# View Real-time Logs
tail -f ~/project_output_resnet_lstm/dev/logs/01-JOB_ID.out

# Cancel a Job
scancel JOB_ID
```

## For Development Phase
- Before running the big `sbatch` jobs for training (02-04), it's a good idea to run a "quicktest" locally to make sure the data loading works:
```bash
export ENV="dev"
python mains/01_preprocess_data.py  # Run one or two videos locally first
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