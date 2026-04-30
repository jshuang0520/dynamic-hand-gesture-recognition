import os
import torch
import torch.nn as nn
from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config
from src.models.hybrid import HybridResNetLSTM
from src.trainer.engine import Trainer
from src.data.dataset import get_mock_dataloader

if __name__ == "__main__":
    print("\n--- Starting Experiment 3: Hybrid ResNet50 + LSTM ---")
    cfg = load_config()
    exp_cfg, paths_cfg = cfg['experiment'], cfg['paths']
    
    lock_seeds(exp_cfg['seed'])
    os.makedirs(paths_cfg['weights_dir'], exist_ok=True)
    
    train_loader = get_mock_dataloader(exp_cfg['batch_size'], exp_cfg['num_classes'])
    val_loader = get_mock_dataloader(exp_cfg['batch_size'], exp_cfg['num_classes'])
    
    model = HybridResNetLSTM(num_classes=exp_cfg['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, exp_cfg['quick_test'])
    trained_model = trainer.run(mode=exp_cfg['execution_mode'], epochs=10)
    
    weights_path = os.path.join(paths_cfg['weights_dir'], "exp3_hybrid_best.pth")
    torch.save(trained_model.state_dict(), weights_path)
    print(f"[SUCCESS] Weights saved to {weights_path}\n")