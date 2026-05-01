import os
import torch
import torch.nn as nn
import torch.optim as optim
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import get_loaders
from src.models.hybrid import HybridResNetLSTM

def run_exp3_hybrid():
    cfg = load_config()
    logger = get_logger("04_EXP3_HYBRID", log_dir=cfg['paths']['logs_dir'])
    logger.info("Initializing Experiment 3 (Proposed Hybrid ResNet50 + LSTM)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(cfg)
    
    try:
        model = HybridResNetLSTM(num_classes=cfg['experiment']['num_classes'])
        model = model.to(device)
        logger.info("Hybrid ResNet50+LSTM loaded.")
    except Exception as e:
        logger.error("Failed to load Hybrid model. Check src/models/hybrid.py.", exc_info=True)
        raise e

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['experiment']['learning_rate'])
    
    epochs = cfg['experiment']['num_epochs']
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            # Check what your Hybrid model expects! 
            # If it uses 2D ResNet50, it likely expects (B, F, C, H, W)
            # Adjust permutation if your forward pass crashes here.
            inputs = inputs.to(device) 
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
        
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct_val / total_val
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = os.path.join(cfg['paths']['weights_dir'], "exp3_hybrid_best.pth")
            torch.save(model.state_dict(), out_path)
            logger.info(f"🌟 New best hybrid model saved to {out_path}")

if __name__ == "__main__":
    run_exp3_hybrid()