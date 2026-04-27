#!/usr/bin/env bash

# 1. Dynamically get the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Point the environment to the massive scratch drive
ENV_DIR="${HOME}/scratch.msml640/gesture_rec_env"

# The requirements file still lives inside the code repo
REQ_FILE="$PROJECT_ROOT/requirements_hpc.txt"

# 2. Check for the --update parameter
UPDATE_REQS=false
if [[ "$1" == "--update" || "$1" == "-u" ]]; then
    UPDATE_REQS=true
fi

# 3. Check if the environment is TRULY valid by looking for the activate script
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "[INFO] Valid environment not found or corrupted. Building a fresh one..."
    mkdir -p "${HOME}/scratch.msml640"
    
    # Nuke the folder if it's a corrupted "half-built" shell
    rm -rf "$ENV_DIR"
    
    # Actually build the Python environment
    python3.11 -m venv "$ENV_DIR"
    UPDATE_REQS=true
else
    echo "[INFO] Found valid virtual environment on scratch drive."
fi

# 4. Activate the environment
source "$ENV_DIR/bin/activate"

# 5. Install or update requirements with OFFLINE FALLBACK & ERROR CHECKING
if [ "$UPDATE_REQS" = true ]; then
    echo "[INFO] Installing/Updating packages..."
    
    # Extra Safety Check: Ensure pip exists before trying to use it
    if ! command -v pip &> /dev/null; then
        echo "🚨 [ERROR] pip is missing! The virtual environment failed to build correctly."
        return 1 2>/dev/null || exit 1
    fi

    pip install --upgrade pip
    
    OFFLINE_DIR="${HOME}/manually_downloaded_pkg/torch_offline"
    INSTALL_SUCCESS=true
    
    if [ -f "$REQ_FILE" ]; then
        if [ -d "$OFFLINE_DIR" ]; then
            echo "[INFO] Found offline packages. Prioritizing local wheels..."
            pip install -f "$OFFLINE_DIR" -r "$REQ_FILE" || INSTALL_SUCCESS=false
        else
            pip install -r "$REQ_FILE" || INSTALL_SUCCESS=false
        fi
        
        # --- THE SAFETY CHECK ---
        if [ "$INSTALL_SUCCESS" = true ]; then
            echo "[INFO] Dependencies installed successfully!"
        else
            echo "--------------------------------------------------------"
            echo "🚨 [WARNING] PIP INSTALLATION FAILED!"
            echo "🚨 One or more packages could not be installed."
            echo "🚨 The environment is active, but missing dependencies."
            echo "--------------------------------------------------------"
            return 1 2>/dev/null || exit 1 
        fi
    else
        echo "🚨 [ERROR] $REQ_FILE not found in $PROJECT_ROOT."
        return 1 2>/dev/null || exit 1
    fi
else
    echo "ℹ️ [INFO] Skipping installation. Run with --update to force refresh."
fi

# This will only print if everything succeeded
echo "✅ [SUCCESS] SHAN Environment is fully built and active!"
echo "📍 [PATH] Environment installed at: $ENV_DIR"


# How to use it:
# Important: Because this script alters your current terminal session (by activating the environment), you must use the source command (or the . shortcut) to run it. If you just run ./setup_env.sh, the environment will activate and immediately deactivate when the script finishes.

# Scenario A: The First Time (or on a new teammate's machine)
# No matter where you are on the HPC, just point source to the script. It will see the environment is missing, create it, and install everything.
# Bash
# source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh


# Scenario B: Every Day After That
# When you log into the HPC tomorrow, run the exact same command. It will see the environment already exists, skip the 5-minute installation process, and just instantly activate it for you.
# Bash
# source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh


# Scenario C: You update requirements_hpc.txt
# If you or your teammate adds a new package to the requirements file, just pass the --update flag. It will activate the environment and run the pip installation again to grab the new packages.
# Bash
# source ~/dynamic-hand-gesture-recognition/scripts/setup_env.sh --update