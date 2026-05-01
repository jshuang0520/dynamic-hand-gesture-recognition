import os
import yaml
from dotenv import load_dotenv
from utilities.logger import get_logger

logger = get_logger("CONFIG")

def load_config():
    """Loads variables and parses the YAML based on the ENV variable."""
    load_dotenv() 
    
    # Check the environment variable. Default to 'dev' for safety.
    env_mode = os.environ.get("ENV", "dev").lower()
    if env_mode not in ["dev", "prod"]:
        logger(f"Invalid ENV '{env_mode}'. Defaulting to 'dev'.", "WARNING")
        env_mode = "dev"
        
    logger(f"Loading configuration for: {env_mode.upper()}")
    
    config_path = f"configs/{env_mode}/config.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: {full_config_path}")
        
    with open(full_config_path, 'r') as f:
        expanded_yaml = os.path.expandvars(f.read())
    return yaml.safe_load(expanded_yaml)