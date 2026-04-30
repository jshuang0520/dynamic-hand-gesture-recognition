# Dynamic Hand Gesture Recognition

This repository contains the official implementation for evaluating a proposed Hybrid ResNet-50 + LSTM architecture against baseline 3D ResNets.

## Overall Code Structure

* **`configs/` & `.env`**: Centralized configuration management. Hyperparameters are tracked in `config.yaml`, while user-specific HPC directory paths are kept securely in local `.env` files.
* **`scripts/`**: Shell and SLURM scripts for environment setup and job submission.
* **`mains/`**: The executable entry points. These scripts utilize the `config_parser` to dynamically set up the training pipelines.
* **`src/`**: Houses the decoupled logic for data augmentation, model architectures, and the hardware-agnostic training loop engine.

## Getting Started: Teammate Setup

Because we are working across different user accounts on Zaratan, **you must configure your local paths before running any code.**

1.  **Environment Setup:** Run `bash scripts/setup_env.sh` ONCE to build your Python 3.11 virtual environment.
2.  **Configure Paths:**
    * Copy the template: `cp .env.example .env`
    * Edit `.env` to match your specific `/home/username/` directories.
3.  **Activate:** `source ${HOME}/SHELL.msml640/bin/activate`

## Executing a Quick Test Locally

You can verify the architecture on a CPU before burning HPC allocation.

1.  Open `configs/config.yaml` and ensure:
    ```yaml
    quick_test: true
    execution_mode: "cpu"
    ```
2.  Run the pipeline: `python mains/03_train_exp3_hybrid.py`

## HPC Deployment (Zaratan)

When ready to train on the full Jester dataset using the H100 GPUs:
1.  Generate splits: `python mains/00_prepare_datasets.py`
2.  Update `config.yaml` to `quick_test: false` and `execution_mode: "gpu"`.
3.  Submit the job: `sbatch scripts/run_exp3_hybrid.sbatch`