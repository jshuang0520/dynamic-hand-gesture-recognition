import os
import csv
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
    logger.info("Initializing Experiment 3 (Robust Hybrid ResNet50 + LSTM with Projection)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(cfg)
    
    try:
        model = HybridResNetLSTM(
            num_classes=cfg['experiment']['num_classes'],
            hidden_dim=cfg['experiment']['lstm_hidden_dim'],
            projection_dim=cfg['experiment']['projection_dim'],
            dropout_rate=cfg['experiment']['dropout_rate']
        )
        model = model.to(device)
    except Exception as e:
        logger.error("Failed to load Hybrid model.", exc_info=True)
        raise e

    criterion = nn.CrossEntropyLoss()
    
    # --- UPGRADE: Differential Learning Rates ---
    base_lr = cfg['experiment']['learning_rate'] # e.g., 1e-4
    lr_mult = cfg['experiment']['backbone_lr_multiplier'] # <-- Pull from config

    # The pre-trained ResNet gets a 10x smaller learning rate (1e-5)
    # The new, randomly initialized layers get the full learning rate (1e-4)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': base_lr * lr_mult}, # <-- Use it here
        {'params': model.projection.parameters(), 'lr': base_lr},     
        {'params': model.lstm.parameters(), 'lr': base_lr},           
        {'params': model.classifier.parameters(), 'lr': base_lr}      
    ], weight_decay=cfg['experiment']['weight_decay'])
    
    epochs = cfg['experiment']['num_epochs']
    best_val_loss = float('inf')
    metrics_history = []
    
    # --- ROBUSTNESS: Early Stopping Setup ---
    patience = cfg['experiment']['early_stopping_patience']
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (data, targets, _) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # --- ROBUSTNESS: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 2. Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, targets, _ in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        metrics_history.append({
            'epoch': epoch + 1, 'train_loss': avg_train_loss, 
            'val_loss': avg_val_loss, 'train_acc': train_acc, 'val_acc': val_acc
        })
        
        # 3. Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 
            
            os.makedirs(cfg['paths']['weights_dir'], exist_ok=True)
            out_path = os.path.join(cfg['paths']['weights_dir'], "exp3_hybrid_best.pth")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, out_path)
            logger.info(f"🌟 New best model (Val Loss: {best_val_loss:.4f}) saved to {out_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"⚠️ Validation loss did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logger.info(f"🛑 EARLY STOPPING triggered at epoch {epoch+1}! No improvement for {patience} consecutive epochs.")
                break

    # 4. Save Metrics CSV
    os.makedirs(cfg['paths']['logs_dir'], exist_ok=True)
    metrics_file = os.path.join(cfg['paths']['logs_dir'], "exp3_metrics.csv")
    with open(metrics_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        writer.writeheader()
        writer.writerows(metrics_history)
        
    logger.info("✅ Experiment 3 Training Completed.")

if __name__ == "__main__":
    run_exp3_hybrid()