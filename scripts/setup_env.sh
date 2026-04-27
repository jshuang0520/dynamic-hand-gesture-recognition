#!/usr/bin/env bash

# 1. Dynamically get the absolute path of the directory containing this script
# Then step up one level to get the project_root. 
# This means you can run this from ANY folder on the HPC!
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_DIR="$PROJECT_ROOT/gesture_rec_env"
REQ_FILE="$PROJECT_ROOT/requirements_hpc.txt"

# 2. Check for the --update parameter
UPDATE_REQS=false
if [[ "$1" == "--update" || "$1" == "-u" ]]; then
    UPDATE_REQS=true
fi

# 3. Check if the virtual environment exists
if [ ! -d "$ENV_DIR" ]; then
    echo "[INFO] Virtual environment not found at $ENV_DIR. Creating one now..."
    python3.11 -m venv "$ENV_DIR"
    UPDATE_REQS=true # Force install since it's a brand new environment
else
    echo "[INFO] Found existing virtual environment at $ENV_DIR."
fi

# 4. Activate the environment
echo "[INFO] Activating environment..."
source "$ENV_DIR/bin/activate"

# 5. Install or update requirements if flagged
if [ "$UPDATE_REQS" = true ]; then
    echo "[INFO] Installing/Updating packages from $REQ_FILE..."
    pip install --upgrade pip
    
    if [ -f "$REQ_FILE" ]; then
        pip install -r "$REQ_FILE"
        echo "[INFO] Dependencies installed successfully!"
    else
        echo "[ERROR] $REQ_FILE not found! Check your project root."
    fi
else
    echo "[INFO] Skipping installation. (Run 'source scripts/setup_env.sh --update' to force package refresh)."
fi

echo "[SUCCESS] SHAN Environment is active!"
echo "Current Python: $(which python)"


# How to use it:
# Important: Because this script alters your current terminal session (by activating the environment), you must use the source command (or the . shortcut) to run it. If you just run ./setup_env.sh, the environment will activate and immediately deactivate when the script finishes.

# Scenario A: The First Time (or on a new teammate's machine)
# No matter where you are on the HPC, just point source to the script. It will see the environment is missing, create it, and install everything.

# Bash
# source /path/to/your/project/scripts/setup_env.sh
# Scenario B: Every Day After That
# When you log into the HPC tomorrow, run the exact same command. It will see the environment already exists, skip the 5-minute installation process, and just instantly activate it for you.

# Bash
# source /path/to/your/project/scripts/setup_env.sh
# Scenario C: You update requirements_hpc.txt
# If you or your teammate adds a new package to the requirements file, just pass the --update flag. It will activate the environment and run the pip installation again to grab the new packages.

# Bash
# source /path/to/your/project/scripts/setup_env.sh --update