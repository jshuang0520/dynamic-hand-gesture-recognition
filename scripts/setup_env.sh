#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Define the error handler
handle_error() {
    echo "❌ [ERROR] Failed to build the SHAN Environment. Please check the logs above."
    exit 1
}

# Trap any errors and route them to the error handler
trap 'handle_error' ERR

# 1. Load variables securely from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "❌ [ERROR] .env file not found in the current directory."
    exit 1
fi

echo "ℹ️ [INFO] Building MSML640 Virtual Environment (Python 3.11)..."

# 2. Create the virtual environment using the dynamic path
python3.11 -m venv "${ENV_DIR}"

# 3. Install dependencies
echo "ℹ️ [INFO] Upgrading pip and installing requirements..."
"${ENV_DIR}/bin/pip" install --upgrade pip
"${ENV_DIR}/bin/pip" install -r requirements_hpc.txt

# This will only print if everything succeeded
echo "✅ [SUCCESS] SHAN Environment is fully built and active!"
echo "📍 [PATH] Environment installed at: ${ENV_DIR}"