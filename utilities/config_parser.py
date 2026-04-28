import os
import yaml

def load_config(config_path=None):
    """Loads YAML and dynamically expands environment variables like ${HOME}."""
    
    # 1. Dynamically find the project root
    # __file__ gets the path of this exact parser script (utilities/config_parser.py)
    # os.path.dirname gets the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 2. Safely construct the absolute default path
    if config_path is None:
        config_path = os.path.join(project_root, "configs", "config_hpc.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"🚨 Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        raw_yaml = f.read()
        
    # 3. Expand bash variables (like ${HOME}) BEFORE parsing the YAML
    expanded_yaml = os.path.expandvars(raw_yaml)
    config = yaml.safe_load(expanded_yaml)
    
    return config