import os
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.trainer.engine import Trainer
from src.data.dataset import build_dataloader

logger = get_logger("EXP_1_FROZEN")

if __name__ == "__main__":
    logger("Initializing Experiment 1: Frozen ResNet-3D-18 Baseline")
    
    # --- SECTION 1: Configuration & Setup ---
    cfg = load_config()
    exp_cfg, paths_cfg = cfg['experiment'], cfg['paths']
    
    lock_seeds(exp_cfg['seed'])
    os.makedirs(paths_cfg['weights_dir'], exist_ok=True)
    
    # --- SECTION 2: Data Pipeline Initialization ---
    dataset_base = paths_cfg['dataset_dir'] 
    train_loader = build_dataloader(dataset_base, "train", cfg, is_train=True)
    val_loader = build_dataloader(dataset_base, "val", cfg, is_train=False)
    
    # --- SECTION 3: Model Architecture ---
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    
    logger("Freezing spatial-temporal convolution layers...", "WARNING")
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final classification head (this automatically sets requires_grad=True for this layer)
    model.fc = nn.Linear(model.fc.in_features, exp_cfg['num_classes'])
    
    # --- SECTION 4: Optimizer & Engine ---
    # Only pass the unfrozen classification head parameters to the optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=exp_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, exp_cfg['quick_test'])
    trained_model = trainer.run(mode=exp_cfg['execution_mode'], epochs=10)
    
    # --- SECTION 5: Save Artifacts ---
    weights_path = os.path.join(paths_cfg['weights_dir'], "exp1_frozen_best.pth")
    torch.save(trained_model.state_dict(), weights_path)
    logger(f"Weights successfully saved to {weights_path}")