#!/usr/bin/env bash
# Make sure requirements_hpc.txt includes pandas, pillow, torch, torchvision, python-dotenv, pyyaml
echo "[INFO] Building MSML640 Virtual Environment (Python 3.11)..."
ENV_DIR="${HOME}/SHELL.msml640"
python3.11 -m venv ${ENV_DIR}
${ENV_DIR}/bin/pip install --upgrade pip
${ENV_DIR}/bin/pip install -r requirements_hpc.txt
echo "[SUCCESS] Environment built."