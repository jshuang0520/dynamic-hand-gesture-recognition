import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.video import r3d_18, R3D_18_Weights
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import get_loaders

def run_exp1_frozen():
    cfg = load_config()
    logger = get_logger("02_EXP1_FROZEN", log_dir=cfg['paths']['logs_dir'])
    logger.info("Initializing Baseline 1 (Frozen ResNet-3D-18)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(cfg)
    
    try:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
            
        num_ftrs = model.fc.in_features
        # --- ENHANCEMENT: Dropout added ---
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, cfg['experiment']['num_classes'])
        )
        model = model.to(device)
    except Exception as e:
        logger.error("Failed to initialize model.", exc_info=True)
        raise e

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=cfg['experiment']['learning_rate'])
    
    epochs = cfg['experiment']['num_epochs']
    best_val_loss = float('inf') # --- FIX FOR HANNAH: Val Loss Optimization ---
    metrics_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
            # --- FIX FOR HANNAH: Dynamic Downsampling ---
            target_size = cfg['experiment']['model_input_size']
            inputs = F.interpolate(inputs, size=(16, target_size, target_size), mode='trilinear', align_corners=False)
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
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
                inputs = F.interpolate(inputs, size=(16, target_size, target_size), mode='trilinear', align_corners=False)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct_val / total_val
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        metrics_history.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        
        # --- ENHANCEMENT: Checkpoint Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            out_path = os.path.join(cfg['paths']['weights_dir'], "exp1_baseline_frozen.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, out_path)
            logger.info(f"🌟 New best model (Val Loss: {best_val_loss:.4f}) saved to {out_path}")
            
    # Save Metrics CSV
    metrics_file = os.path.join(cfg['paths']['logs_dir'], "exp1_metrics.csv")
    with open(metrics_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        writer.writeheader()
        writer.writerows(metrics_history)

if __name__ == "__main__":
    run_exp1_frozen()