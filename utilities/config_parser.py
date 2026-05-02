import os
import yaml
from dotenv import load_dotenv
from utilities.logger import get_logger

logger = get_logger("CONFIG")

def load_config():
    """Loads variables and parses the YAML based on the ENV variable."""
    load_dotenv() 
    
    # 1. Determine Environment (Defaults to 'dev')
    env_mode = os.environ.get("ENV", "dev").lower()
    if env_mode not in ["dev", "prod"]:
        logger.warning(f"Invalid ENV '{env_mode}'. Defaulting to 'dev'.")
        env_mode = "dev"
        
    logger.info(f"Loading configuration for: {env_mode.upper()}")
    
    # 2. Route to the correct file
    config_path = f"configs/{env_mode}/config.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: {full_config_path}")
        
    # 3. Parse the YAML
    with open(full_config_path, 'r') as f:
        expanded_yaml = os.path.expandvars(f.read())
    cfg = yaml.safe_load(expanded_yaml)

    # 4. Validate Structural Integrity
    required_blocks = ['experiment', 'data_splits', 'paths', 'preprocessing']
    for block in required_blocks:
        if block not in cfg:
            raise KeyError(f"Missing required configuration block: '{block}' in {config_path}")

    # 5. Provide safe fallbacks for new variables
    if 'preprocess_size' not in cfg['experiment']:
        cfg['experiment']['preprocess_size'] = 224
    if 'model_input_size' not in cfg['experiment']:
        cfg['experiment']['model_input_size'] = 112

    return cfg