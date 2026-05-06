# Spatio-Temporal Modeling for Dynamic Hand Gesture Recognition

This repository contains a high-performance, decoupled pipeline for evaluating video-based gesture recognition. The project benchmarks a **Hybrid ResNet-50 + LSTM** architecture against **3D ResNet** baselines using the 20bn-jester-v1 dataset.

To ensure 100% scientific reproducibility and bypass the I/O bottlenecks of reading thousands of small JPEG files, this pipeline utilizes **Offline Preprocessing**. Videos are uniformly sampled, transformed (noise/pad/crop), and saved as binary tensors before training begins.

---

## 📊 Dataset & Preprocessing

**Dataset:** The models are trained and evaluated on a subset of the **20BN-Jester Dataset**, a large-scale collection of densely labeled video clips showing humans performing basic pre-defined hand gestures.
* **Target Classes:** Focused on 4 distinct gesture classes to evaluate spatiotemporal modeling (e.g., *Swiping Left, Sliding Two Fingers Down, Stop Sign, Thumb Up*).

**Data Preprocessing Pipeline:**
To optimize for GPU memory and cluster training (Zaratan HPC), the raw video MP4s/JPEGs are heavily preprocessed into standardized PyTorch tensors before training:
1.  **Temporal Subsampling:** Videos are uniformly subsampled (or padded) to exactly **16 frames** per sequence.
2.  **Spatial Resizing:** Frames are center-cropped and resized to **224x224** pixels.
3.  **Tensor Serialization:** Preprocessed sequences are saved as `.pt` tensor files for high-speed I/O loading during training, bypassing heavy video-decoding bottlenecks.
4.  **Robust Dataloading:** The custom `JesterTensorDataset` dynamically drops empty/corrupted rows, securely maps string labels to integer indices, and falls back to zero-tensors if a sequence is missing, ensuring uninterrupted cluster jobs.

## Methodology & Architectures

We evaluated three distinct architectural approaches to understand the trade-offs between spatial and temporal feature extraction:

### 1. Baseline 1: Frozen ResNet-3D-18
* **Architecture:** Uses a pre-trained `r3d_18` model. The 3D convolutional backbone is entirely frozen, and only a newly initialized fully connected (FC) classification head is trained.
* **Purpose:** Establishes a baseline to see if out-of-the-box 3D spatiotemporal features learned from generalized video datasets (like Kinetics) map well to specific hand gestures.

### 2. Baseline 2: Fine-Tuned ResNet-3D-18
* **Architecture:** The same `r3d_18` architecture, but the entire network is unfrozen and fine-tuned.
* **Purpose:** Allows the 3D convolutional kernels to adapt specifically to the gradient flow of hand movement. 

### 3. Proposed Hybrid: ResNet-50 + LSTM
* **Architecture:** A 2D spatial extractor (`ResNet-50`) applied frame-by-frame, followed by a Temporal Sequence Modeler (`LSTM`) to classify the gesture across time.
* **Current Observation:** This model currently underperforms compared to the 3D networks. 
* **Theoretical Cause:** ResNet-50 is a 2D network; it processes every frame as an isolated, static photograph. By the time it passes its 2048-dimensional feature vector to the LSTM, the rich motion data (the "swipe") is drowned out by the static background noise of the room. Conversely, 3D kernels ($x, y, t$) natively track pixel gradients *over time*, naturally zeroing out static background noise.

## 🛡️ Pipeline Engineering & Robustness

The training pipeline is built with enterprise MLOps standards to prevent overfitting and track cluster experiments:
* **Experiment Tracking:** Uses dynamic `PIPELINE_RUN_ID` environment variables. Model weights are kept static for easy downstream evaluation, while all loss curves, confusion matrices, and prediction `.csv` files are automatically routed to timestamped run folders.
* **Overfitting Safeguards:** Implemented Early Stopping (patience=5), Gradient Clipping (max_norm=1.0) to prevent exploding gradients in the LSTM, Dropout (p=0.5), and $L_2$ Regularization (Weight Decay).
* **Error Analysis:** The evaluation script natively tracks test-set `video_ids`, mapping predictions vs. actuals into CSVs so researchers can visually inspect the specific videos the model failed on.

## Future Work & Improvements

To address the performance gap of the Hybrid model and push accuracy higher, the following techniques are proposed for future iterations:

1.  **Optical Flow / Frame Differencing (Data Level):**
    * Instead of passing raw RGB frames to the ResNet-50, we will pass *differences* between consecutive frames, or calculate dense Optical Flow. This explicitly forces the network to look only at the moving pixels (the hand) and completely eliminates static background interference.
2.  **Bi-Directional Deep LSTMs (Temporal Level):**
    * Upgrading the 1-layer LSTM to a 2-layer Bi-Directional LSTM. Reading the video sequence forwards and backwards allows the network to understand the complete trajectory of the hand before committing to a classification.
3.  **Feature Normalization:**
    * Injecting `nn.LayerNorm` between the ResNet-50 and the LSTM to act as a "volume control," preventing massive static image features from overwhelming the LSTM's sensitive forget-gates.
4.  **Test-Time Augmentation (TTA) & Ensembling:**
    * Evaluating overlapping 16-frame windows during inference and averaging the probabilities across the 3D and Hybrid models to create a highly robust ensemble classifier.

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
# cp -r jester_subsampled_data/dev/raw_data/ project_output_resnet_lstm/dev/

# # start testing dev scripts
# sbatch scripts/run_01_preprocess_data_cpu.sbatch
# sbatch scripts/run_02_exp1_frozen3d_cpu.sbatch
# sbatch scripts/run_03_exp2_finetune3d_cpu.sbatch
# sbatch scripts/run_04_exp3_train_hybrid_cpu.sbatch
# sbatch scripts/run_05_evaluate_models_cpu.sbatch

# check space
du -sh * .[^.]* | sort -rh | head -n 10  # this command checks everything in your home dir, including hidden files like .conda or .cache
du -ah . --max-depth=1 | sort -rh | head -n 10


# prod run
# 0. Generate the timestamp once (e.g., res_since_20260503_1701) and export it
export PIPELINE_RUN_ID="res_since_$(date +%Y%m%d_%H%M)"
# 2. Launch your jobs! They will all inherit that exact same PIPELINE_RUN_ID
# 1. Run the first job and capture its Job ID
JOB1=$(sbatch --parsable scripts/run_01_preprocess_data.sbatch)
# 2. Run the next 3 jobs in parallel, waiting for JOB1 to finish successfully
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_02_exp1_frozen3d.sbatch)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_03_exp2_finetune3d.sbatch)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_04_exp3_train_hybrid.sbatch)
# 3. Run the final job only after all three middle jobs (JOB2, JOB3, and JOB4) finish successfully
sbatch --dependency=afterok:$JOB2:$JOB3:$JOB4 scripts/run_05_evaluate_models.sbatch


# when run_01_preprocess_data is already done successfully
export PIPELINE_RUN_ID="res_since_$(date +%Y%m%d_%H%M)"
JOB2=$(sbatch --parsable scripts/run_02_exp1_frozen3d.sbatch)
JOB3=$(sbatch --parsable scripts/run_03_exp2_finetune3d.sbatch)
JOB4=$(sbatch --parsable scripts/run_04_exp3_train_hybrid.sbatch)
sbatch --dependency=afterok:$JOB2:$JOB3:$JOB4 scripts/run_05_evaluate_models.sbatch


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
