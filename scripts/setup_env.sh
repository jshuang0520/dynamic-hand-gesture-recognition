#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Define the error handler
handle_error() {
    echo "❌ [ERROR] Failed to build the RESNET_LSTM Environment. Please check the logs above."
    exit 1
}

# Trap any errors
trap 'handle_error' ERR

# --- 1. Load and Clean Environment ---
if [ -f .env ]; then
    set -a; source .env; set +a
else
    echo "❌ [ERROR] .env file not found. Ensure PROJECT_ROOT and ENV_DIR are defined."
    exit 1
fi

# Clean paths (removes invisible characters and extra spaces)
ENV_DIR_CLEAN=$(echo "${ENV_DIR}" | tr -d '\r' | xargs)

echo "ℹ️ [INFO] Building Environment at: ${ENV_DIR_CLEAN}"

# --- 2. Create the virtual environment ---
# We use the cleaned path to ensure the folder name is literal and predictable
python3.11 -m venv "${ENV_DIR_CLEAN}"

# --- 3. FIX PERMISSIONS FOR COMPUTE NODES ---
echo "🔐 [INFO] Setting permissions for SLURM access..."

# # Ensure the compute node can 'pass through' your home directory
# chmod +x "$HOME"

# Set permissions for the environment folder and its parents
# We use '|| true' because if a folder is a shared class folder, 
# you might not own it, and we don't want the script to crash.
chmod +x "$(dirname "${ENV_DIR_CLEAN}")" || true
chmod -R 755 "${ENV_DIR_CLEAN}"

echo "✅ Permissions updated."

# --- 4. Upgrade basic tools ---
# Using the absolute path to the newly created bin to ensure we are in the venv
"${ENV_DIR_CLEAN}/bin/pip" install --upgrade pip setuptools wheel

# --- 5. Install Requirements ---
echo "ℹ️ [INFO] Installing packages (this may take several minutes)..."
# Note: Ensure requirements_hpc.txt is in the project root
"${ENV_DIR_CLEAN}/bin/pip" install -r requirements_hpc.txt

echo "✅ [SUCCESS] RESNET_LSTM Environment is fully built and active!"
echo "📍 [PATH] Environment installed at: ${ENV_DIR_CLEAN}"