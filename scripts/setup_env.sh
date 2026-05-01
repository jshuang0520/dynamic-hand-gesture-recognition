#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Define the error handler
handle_error() {
    echo "❌ [ERROR] Failed to build the SHAN Environment. Please check the logs above."
    exit 1
}

# Trap any errors
trap 'handle_error' ERR

# 1. Load variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "❌ [ERROR] .env file not found. Ensure PROJECT_ROOT and ENV_DIR are defined."
    exit 1
fi

echo "ℹ️ [INFO] Building Environment at: ${ENV_DIR}"

# 2. Create the virtual environment
python3.11 -m venv "${ENV_DIR}"

# 3. Upgrade basic tools
"${ENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

# 4. Install Requirements
# The --extra-index-url in the requirements file will now be recognized
echo "ℹ️ [INFO] Installing packages (this may take several minutes)..."
"${ENV_DIR}/bin/pip" install -r requirements_hpc.txt

echo "✅ [SUCCESS] SHAN Environment is fully built and active!"
echo "📍 [PATH] Environment installed at: ${ENV_DIR}"