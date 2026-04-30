import os
from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config

def create_splits(config, is_dummy=False):
    out_dir = config['paths']['splits_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    status = "Dummy CI/CD Dataset" if is_dummy else "Full 2k Subsample"
    print(f"[INFO] Generating {status} in {out_dir}...")
    # Add actual logic to parse frames_dir and write CSVs here

if __name__ == "__main__":
    cfg = load_config()
    lock_seeds(cfg['experiment']['seed'])
    create_splits(cfg, is_dummy=True)
    create_splits(cfg, is_dummy=False)