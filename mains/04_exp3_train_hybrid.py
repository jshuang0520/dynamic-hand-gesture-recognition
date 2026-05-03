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
    logger.info("Initializing Experiment 3 (Robust Hybrid ResNet50 + LSTM)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(cfg)
    
    try:
        model = HybridResNetLSTM(num_classes=cfg['experiment']['num_classes'])
        model = model.to(device)
    except Exception as e:
        logger.error("Failed to load Hybrid model.", exc_info=True)
        raise e

    criterion = nn.CrossEntropyLoss()
    
    # --- ROBUSTNESS: Weight Decay (L2 Regularization) ---
    optimizer = optim.Adam(model.parameters(), lr=cfg['experiment']['learning_rate'], weight_decay=1e-4)
    
    epochs = cfg['experiment']['num_epochs']
    best_val_loss = float('inf')
    metrics_history = []
    
    # --- ROBUSTNESS: Early Stopping Setup ---
    patience = 5 
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch in train_loader:
            # --- SAFE UNPACKING: Discards the ID during training ---
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
                
            # 2D ResNet uses the 224x224 inputs directly, no interpolation/permute needed
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # --- ROBUSTNESS: Gradient Clipping for LSTM ---
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                # --- SAFE UNPACKING ---
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch
                    
                inputs = inputs.to(device)
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
        
        # --- ROBUSTNESS: Early Stopping & Loss-Based Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset counter
            
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
                break # Kills the loop gracefully

    # Save Metrics CSV regardless of early stopping
    metrics_file = os.path.join(cfg['paths']['logs_dir'], "exp3_metrics.csv")
    with open(metrics_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        writer.writeheader()
        writer.writerows(metrics_history)
        
    logger.info("✅ Experiment 3 Hybrid Training Complete.")

if __name__ == "__main__":
    run_exp3_hybrid()