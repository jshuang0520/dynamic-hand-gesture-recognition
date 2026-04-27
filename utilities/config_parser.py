import os
import yaml

def load_config(config_path="configs/config_hpc.yaml"):
    """Loads YAML and dynamically expands environment variables like ${HOME}."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        raw_yaml = f.read()
        expanded_yaml = os.path.expandvars(raw_yaml)
        config = yaml.safe_load(expanded_yaml)
    return config