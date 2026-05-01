#!/usr/bin/env bash

# --- 1. Load and Clean OUTPUT_ROOT from .env ---
if [ -f .env ]; then
    set -a; source .env; set +a
    # Clean the variable: strip \r and trim whitespace
    OUTPUT_ROOT_CLEAN=$(echo "${OUTPUT_ROOT}" | tr -d '\r' | xargs)
else
    echo "⚠️  Warning: .env file not found."
fi

# --- 2. Resolve Base Directory ---
BASE_DIR=${OUTPUT_ROOT_CLEAN:-"/home/shhuang/project_output_resnet_lstm"}

echo "📂 Initializing project directory structure at: $BASE_DIR"

# --- 3. Define the sub-environments ---
ENVS=("dev" "prod")

# --- 4. Loop and Create ---
for ENV in "${ENVS[@]}"; do
    echo "  🏗️  Creating directories for [$ENV] environment..."
    
    # 1. New Raw Data Structure
    mkdir -p "$BASE_DIR/$ENV/raw_data/annotations"
    mkdir -p "$BASE_DIR/$ENV/raw_data/videos/train"
    mkdir -p "$BASE_DIR/$ENV/raw_data/videos/val"
    mkdir -p "$BASE_DIR/$ENV/raw_data/videos/test"
    
    # 2. Processed Data and Models
    mkdir -p "$BASE_DIR/$ENV/metadata/data_processed"
    mkdir -p "$BASE_DIR/$ENV/models"
    mkdir -p "$BASE_DIR/$ENV/logs"
done

echo "---"
echo "✅ Directory structure is ready!"
echo "📍 Location: $BASE_DIR"

# --- Usage Reminder ---
# chmod +x scripts/init_dirs.sh
# ./scripts/init_dirs.sh