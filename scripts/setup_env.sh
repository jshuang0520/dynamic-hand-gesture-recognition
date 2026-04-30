#!/usr/bin/env bash
# [INFO] Run this strictly ONCE to build the environment for your user account.

echo "[INFO] Building MSML640 Virtual Environment (Python 3.11)..."
ENV_DIR="${HOME}/SHELL.msml640"

python3.11 -m venv ${ENV_DIR}

echo "[INFO] Upgrading pip and installing requirements..."
${ENV_DIR}/bin/pip install --upgrade pip
${ENV_DIR}/bin/pip install -r requirements_hpc.txt

echo "[SUCCESS] Environment built at ${ENV_DIR}. Do not source this file directly."