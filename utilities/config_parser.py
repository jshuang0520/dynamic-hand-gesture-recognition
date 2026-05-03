import os
import yaml
from dotenv import load_dotenv
from utilities.logger import get_logger

logger = get_logger("CONFIG")

def load_config():
    load_dotenv() 
    
    env_mode = os.environ.get("ENV", "dev").lower()
    if env_mode not in ["dev", "prod"]:
        env_mode = "dev"
        
    config_path = f"configs/{env_mode}/config.yaml"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: {full_config_path}")
        
    with open(full_config_path, 'r') as f:
        expanded_yaml = os.path.expandvars(f.read())
    cfg = yaml.safe_load(expanded_yaml)

    # --- DEVELOPMENT-FRIENDLY TRACKING ---
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if run_id:
        # ONLY route the logs, CSVs, and plots to the timestamped folder.
        # Leave the weights strictly in the static root folder!
        logger.info(f"📁 Tracking Logs to: {run_id} | Keeping Weights Static.")
        cfg['paths']['logs_dir'] = os.path.join(cfg['paths']['logs_dir'], run_id)
        os.makedirs(cfg['paths']['logs_dir'], exist_ok=True)

    # Validate Structural Integrity
    required_blocks = ['experiment', 'data_splits', 'paths', 'preprocessing']
    for block in required_blocks:
        if block not in cfg:
            raise KeyError(f"Missing required config block: '{block}'")

    # Provide safe fallbacks
    exp = cfg['experiment']
    exp.setdefault('preprocess_size', 224)
    exp.setdefault('model_input_size', 112)
    exp.setdefault('dropout_rate', 0.5)
    exp.setdefault('weight_decay', 1e-4)
    exp.setdefault('early_stopping_patience', 5)

    return cfg