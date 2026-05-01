# Spatio-Temporal Modeling for Dynamic Hand Gesture Recognition

This repository contains a decoupled, production-grade pipeline for evaluating hand gesture recognition. The project compares a **Hybrid ResNet-50 + LSTM** architecture against **3D ResNet** baselines using an offline-preprocessed subsample of the 20bn-jester-v1 dataset.

---

## 📂 Project Structure

```text
/home/shhuang
|_ project_output_resnet_lstm        # Shared Output Root (Excluded from Git)
│   ├── dev/                         # Development Environment
│   │   ├── raw_data/                # Output of 00: Subsampled .jpg folders
│   │   ├── metadata/
│   │   │   ├── splits/              # Output of 00: train/val/test CSVs
│   │   │   ├── data_processed/      # Output of 01: Pre-computed .pt tensors
│   │   │   └── weights/             # Output of 02, 03, 04: Model checkpoints
│   │   └── logs/                    # SLURM and training logs
│   └── prod/                        # Production Environment (Same structure)
│
|_ dynamic-hand-gesture-recognition  # Project Root (Git-tracked)
    ├── .env                         # User-specific paths (ENV_DIR, OUTPUT_ROOT)
    ├── configs/
    │   ├── dev/config.yaml          # GPU-enabled testing parameters
    │   └── prod/config.yaml         # Production/Full-scale parameters
    ├── scripts/
    │   ├── setup_env.sh             # Error-handled environment builder
    │   └── run_0X_...sbatch         # SLURM submission scripts (1-to-1 mapping)
    ├── mains/
    │   ├── 00_prepare_subdataset.py # Task: Subsample & Train/Test Split
    │   ├── 01_preprocess_data.py    # Task: Offline Noise, Pad, Crop -> .pt Tensors
    │   ├── 02_train_exp1_frozen3d.py
    │   ├── 03_train_exp2_finetune.py
    │   ├── 04_train_exp3_hybrid.py   # Primary Entry Point
    │   └── 05_evaluate_models.py    # Final Benchmarking
    ├── src/
    │   ├── data/
    │   │   ├── loader.py            # Tensor-based PyTorch DataLoader
    │   │   └── transforms.py        # Offline spatial & noise processing logic
    │   ├── models/
    │   │   └── hybrid.py            # Hybrid ResNet50 + LSTM architecture
    │   └── trainer/
    │       └── engine.py            # Hardware-agnostic training loop
    └── utilities/
        ├── config_parser.py         # Multi-env YAML & .env parser
        ├── logger.py                # Timestamped stdout monitoring
        └── reproducibility.py       # Global Seed Locker (20260430)
```

---

## 🚀 Getting Started: Teammate Setup

1. **Build the Environment:** Run this script once to build your Python 3.11 virtual environment using the path defined in your `.env`.
    ```bash
    bash scripts/setup_env.sh
    ```
2. **Configure Paths:** Ensure your `.env` points `OUTPUT_ROOT` to `/home/shhuang/project_output_resnet_lstm`.

---

## 🧪 Local/Dev Execution (GPU Enabled)

In this version, even the `dev` environment is configured for GPU execution. The `dev` config uses the subsampled `raw_data` to ensure rapid iteration.

**Primary Entry Point:** `mains/04_train_exp3_hybrid.py`
> **What this does:** Loads pre-processed tensors from `metadata/data_processed` and trains the Hybrid model. Since the heavy lifting (resizing/noise) is already done in Step 01, training is significantly faster.

**Execution Command:**
```bash
export ENV="dev"
python mains/04_train_exp3_hybrid.py
```

---

## ⚡ HPC Deployment (Zaratan Production)

Follow this sequence to execute the full pipeline. All SLURM scripts route logs to `project_output_resnet_lstm/[env]/logs`.

1. **Prepare Subdataset (Task 1):** Subsamples original Jester data into `raw_data` and creates CSV splits.
    ```bash
    sbatch scripts/run_00_prepare_subdataset.sbatch
    ```

2. **Preprocess Data (Offline Augmentation):** Reads JPGs from `raw_data`, applies padding, cropping, and noise, then saves as binary `.pt` files.
    ```bash
    sbatch scripts/run_01_preprocess_data.sbatch
    ```

3. **Train Baseline 1: Frozen 3D ResNet:**
    ```bash
    sbatch scripts/run_02_train_exp1_frozen3d.sbatch
    ```

4. **Train Baseline 2: Fine-Tuned 3D ResNet:**
    ```bash
    sbatch scripts/run_03_train_exp2_finetune.sbatch
    ```

5. **Train Proposed Model: Hybrid ResNet+LSTM:**
    ```bash
    sbatch scripts/run_04_train_exp3_hybrid.sbatch
    ```

6. **Evaluate Models:** Benchmarks accuracy, precision, and F1-score across all experiments.
    ```bash
    sbatch scripts/run_05_evaluate_models.sbatch
    ```

---

## 🛠 Features for Reproducibility
- **Fixed Noise:** Moving noise addition to `01_preprocess_data.py` ensures all models see identical input data.
- **Binary I/O:** Reading `.pt` tensors from the `metadata` directory bypasses the bottleneck of decoding thousands of small `.jpg` files during training.
- **Universal Seed:** Global seed `20260430` is enforced across all five pipeline stages.