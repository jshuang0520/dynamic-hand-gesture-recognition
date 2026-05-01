import os
import torch
import torch.nn as nn
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import get_loaders
from src.trainer.engine import Trainer
from torchvision.models.video import r3d_18, R3D_18_Weights

logger = get_logger("02_EXP1")

if __name__ == "__main__":
    cfg = load_config()
    train_loader, val_loader = get_loaders(cfg)
    
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, cfg['experiment']['num_classes'])
    
    trainer = Trainer(model.cuda(), train_loader, val_loader, 
                     torch.optim.Adam(model.fc.parameters(), lr=cfg['experiment']['learning_rate']),
                     nn.CrossEntropyLoss(), cfg['experiment']['quick_test'])
    
    best_model = trainer.run(mode="gpu", epochs=cfg['experiment']['num_epochs'])
    
    save_path = cfg['evaluation']['model_weights']['exp1']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model.state_dict(), save_path)