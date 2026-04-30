import os
import yaml
from dotenv import load_dotenv

def load_config(config_path="configs/config.yaml"):
    """Loads environment variables and parses the YAML configuration."""
    load_dotenv() # Injects .env variables into os.environ
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: {full_config_path}")
        
    with open(full_config_path, 'r') as f:
        raw_yaml = f.read()
        
    expanded_yaml = os.path.expandvars(raw_yaml)
    return yaml.safe_load(expanded_yaml)