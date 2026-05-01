import os
import torch
import torch.nn as nn
from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.models.hybrid import HybridResNetLSTM
from src.trainer.engine import Trainer
from src.data.dataset import build_dataloader

logger = get_logger("EXP_3_HYBRID")

if __name__ == "__main__":
    logger("Initializing Experiment 3: Hybrid ResNet50 + LSTM Pipeline")
    
    # --- SECTION 1: Configuration & Setup ---
    cfg = load_config()
    exp_cfg, paths_cfg = cfg['experiment'], cfg['paths']
    
    lock_seeds(exp_cfg['seed'])
    os.makedirs(paths_cfg['weights_dir'], exist_ok=True)
    
    # --- SECTION 2: Data Pipeline Initialization ---
    dataset_base = paths_cfg['dataset_dir'] 
    train_loader = build_dataloader(dataset_base, "train", cfg, is_train=True)
    val_loader = build_dataloader(dataset_base, "val", cfg, is_train=False)
    
    # --- SECTION 3: Model & Optimizer ---
    model = HybridResNetLSTM(num_classes=exp_cfg['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # --- SECTION 4: Training Engine Execution ---
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, exp_cfg['quick_test'])
    trained_model = trainer.run(mode=exp_cfg['execution_mode'], epochs=10)
    
    # --- SECTION 5: Save Artifacts ---
    weights_path = os.path.join(paths_cfg['weights_dir'], "exp3_hybrid_best.pth")
    torch.save(trained_model.state_dict(), weights_path)
    logger(f"Weights successfully saved to {weights_path}")