#!/usr/bin/env bash

# 1. Try to load the OUTPUT_ROOT from .env
if [ -f .env ]; then
    # Sourcing only the line we need to avoid side effects
    OUTPUT_ROOT=$(grep '^OUTPUT_ROOT=' .env | cut -d '=' -f2)
fi

# 2. Fallback to the provided path if .env didn't have it
BASE_DIR=${OUTPUT_ROOT:-"/home/shhuang/project_output_resnet_lstm"}

echo "📂 Initializing project directory structure at: $BASE_DIR"

# 3. Define the sub-environments
ENVS=("dev" "prod")

# 4. Loop and create
for ENV in "${ENVS[@]}"; do
    echo "  🏗️ Creating directories for [$ENV] environment..."
    
    mkdir -p "$BASE_DIR/$ENV/raw_data"
    mkdir -p "$BASE_DIR/$ENV/metadata/splits"
    mkdir -p "$BASE_DIR/$ENV/metadata/data_processed"
    mkdir -p "$BASE_DIR/$ENV/models"
    mkdir -p "$BASE_DIR/$ENV/logs"
done

echo "---"
echo "✅ Directory structure is ready!"
echo "📍 Location: $BASE_DIR"


# chmod +x scripts/init_dirs.sh
# ./scripts/init_dirs.sh