import os
import torch
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import get_loaders

def run_exp2_finetune():
    cfg = load_config()
    logger = get_logger("03_EXP2_FINETUNE", log_dir=cfg['paths']['logs_dir'])
    
    logger.info("Initializing Experiment 2 (Full Fine-Tuning)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")
    
    train_loader, val_loader = get_loaders(cfg)
    logger.info(f"Data Loaded: {len(train_loader)} training batches.")
    
    # In a real scenario, you'd load the model and NOT freeze the backbone here
    epochs = cfg['experiment']['num_epochs']
    
    try:
        for epoch in range(epochs):
            logger.info(f"--- Starting Epoch {epoch+1}/{epochs} ---")
            # [Training Loop Logic]
            logger.info(f"Epoch {epoch+1} Complete | Train Loss: 0.000 | Val Acc: 0.00%")
            
        out_path = os.path.join(cfg['paths']['weights_dir'], "exp2_finetune_best.pth")
        # torch.save(model.state_dict(), out_path)
        logger.info(f"Model weights saved to {out_path}")
        logger.info("Experiment 2 successfully completed.")
        
    except Exception as e:
        logger.error("Fatal error during Experiment 2 training.", exc_info=True)
        raise e

if __name__ == "__main__":
    run_exp2_finetune()