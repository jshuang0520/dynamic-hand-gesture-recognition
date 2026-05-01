import os
import torch
import torch.nn as nn
from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.models.hybrid import HybridResNetLSTM
from src.trainer.engine import Trainer
from src.data.dataset import build_dataloader

log = get_logger("EXP_3_HYBRID")

if __name__ == "__main__":
    log("Initializing Experiment 3: Hybrid ResNet50 + LSTM Pipeline")
    
    cfg = load_config()
    exp_cfg, paths_cfg = cfg['experiment'], cfg['paths']
    
    lock_seeds(exp_cfg['seed'])
    os.makedirs(paths_cfg['weights_dir'], exist_ok=True)
    
    # Toggle this to paths_cfg['full_dataset_dir'] for the production run
    dataset_base = paths_cfg['dev_dataset_dir'] if exp_cfg['quick_test'] else paths_cfg['full_dataset_dir']
    
    train_loader = build_dataloader(dataset_base, "train", cfg, is_train=True)
    val_loader = build_dataloader(dataset_base, "val", cfg, is_train=False)
    
    model = HybridResNetLSTM(num_classes=exp_cfg['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, exp_cfg['quick_test'])
    trained_model = trainer.run(mode=exp_cfg['execution_mode'], epochs=10)
    
    weights_path = os.path.join(paths_cfg['weights_dir'], "exp3_hybrid_best.pth")
    torch.save(trained_model.state_dict(), weights_path)
    log(f"Weights successfully saved to {weights_path}")