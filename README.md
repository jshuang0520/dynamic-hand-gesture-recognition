# Spatio-Temporal Modeling for Dynamic Hand Gesture Recognition

This repository contains the official, decoupled implementation for evaluating dynamic hand gesture recognition architectures. The project tests a proposed **Hybrid ResNet-50 + LSTM** model against established **ResNet-3D** baselines. It utilizes a heavily constrained subsample of the 20bn-jester-v1 dataset to operate within strict High-Performance Computing (HPC) limitations.

---

## 📂 Project Structure

```text
/home/shhuang
|_ shan_project (Metadata, Splits, Weights, and Slurm Logs)
|_ dynamic-hand-gesture-recognition (Project Root)
    ├── .env                         # Local environment paths (Teammate-specific)
    ├── configs/
    │   ├── dev/config.yaml          # Development/Testing parameters
    │   └── prod/config.yaml         # Production/HPC parameters
    ├── scripts/
    │   ├── setup_env.sh             # Environment builder
    │   └── run_0X_...sbatch         # SLURM submission scripts (1-to-1 mapping)
    ├── mains/
    │   ├── 00_prepare_datasets.py   # Physical ETL and Subsampling
    │   ├── 01_train_exp1_frozen3d.py
    │   ├── 02_train_exp2_finetune.py
    │   ├── 03_train_exp3_hybrid.py   # Primary Entry Point
    │   └── 04_evaluate_models.py    # Final Benchmarking
    ├── src/
    │   ├── data/
    │   │   ├── dataset.py           # PyTorch Dataset & Loader logic
    │   │   └── add_noise.py         # Online Data Augmentation
    │   ├── models/
    │   │   └── hybrid.py            # Hybrid ResNet50 + LSTM
    │   └── trainer/
    │       └── engine.py            # Hardware-agnostic training loop
    └── utilities/
        ├── config_parser.py         # Multi-env YAML & .env parser
        ├── logger.py                # Timestamped stdout monitoring
        └── reproducibility.py       # Global Seed Locker (20260430)
```

---

## 🚀 Getting Started: Teammate Setup

Because we are working collaboratively on Zaratan, you must configure your local paths before running any code.

1. **Build the Environment:** Run this script **ONCE** to build your Python 3.11 virtual environment at `${HOME}/SHELL.msml640/gesture_rec_env`.

    ```bash
    bash scripts/setup_env.sh
    ```

2. **Configure Paths:**
    * Copy the template: `cp .env.example .env`
    * Edit `.env` to match your specific `/home/username/` directories.

---

## 🧪 Local Execution (Quick Test)

Before burning HPC allocation, verify the architecture locally on a CPU using the `dev` config.

**Primary Entry Point:** `mains/03_train_exp3_hybrid.py`
> **What this does:** Initializes the Hybrid ResNet-50 + LSTM model. In `dev` mode, it uses a small sampled dataset (8/1/1 split) and truncates the training loop to 2 batches for a rapid "smoke test" of the tensor flow.

**Execution Command:**

```bash
# Env defaults to 'dev' if unset
export ENV="dev"
python mains/03_train_exp3_hybrid.py
```

---

## ⚡ HPC Deployment (Zaratan Production)

When you are ready to train on the H100 GPUs using the actual Jester dataset, follow the sequence below. All SLURM scripts internally set `export ENV="prod"` to ensure the 400/50/50 split and GPU acceleration are active.

1. **Generate Data Splits (Task 1):** Physically constructs the `jester_uncompressed` directory structure and generates the `train.csv`, `val.csv`, and `test.csv` splits based on the classes and counts in `prod/config.yaml`.

    ```bash
    sbatch scripts/run_00_prepare_datasets.sbatch
    ```

2. **Train Baseline 1: Frozen 3D ResNet (Task 3):** Trains only the final classification head of a pre-trained ResNet-3D-18.

    ```bash
    sbatch scripts/run_01_exp1_frozen3d.sbatch
    ```

3. **Train Baseline 2: Fine-Tuned 3D ResNet (Task 4):** Updates the entire spatial-temporal network of the ResNet-3D-18.

    ```bash
    sbatch scripts/run_02_exp2_finetune.sbatch
    ```

4. **Train Proposed Model: Hybrid ResNet+LSTM (Task 5):** Trains the custom ResNet-50 spatial extractor + LSTM temporal fusion layer.

    ```bash
    sbatch scripts/run_03_exp3_hybrid.sbatch
    ```

5. **Evaluate Models (Benchmarking):** Loads the best weights from all three experiments and calculates Accuracy, Precision, Recall, and F1-Scores.

    ```bash
    sbatch scripts/run_04_evaluate_models.sbatch
    ```

---

## 🛠 Features for Reproducibility
- **Universal Seed:** Every task (ETL, Training, Eval) pulls `20260430` from the config.
- **Dynamic Logging:** All output files are timestamped and routed to `${SHAN_PROJECT_DIR}/slurm_logs/` using the SLURM `%u` and `%j` variables.
- **Isolated I/O:** The Project Root contains only code. All large data artifacts and logs are kept in a separate sibling directory to simplify Git synchronization.