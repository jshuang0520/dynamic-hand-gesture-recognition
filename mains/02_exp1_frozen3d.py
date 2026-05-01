import os
import torch
import torch.nn as nn
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
    
    # 1. Initialize Baseline Model (ResNet-3D-18 per proposal)
    try:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer for our 4 classes
        # This layer WILL require gradients
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cfg['experiment']['num_classes'])
        model = model.to(device)
        logger.info("ResNet-3D-18 loaded and frozen. Classifier head replaced.")
    except Exception as e:
        logger.error("Failed to initialize model.", exc_info=True)
        raise e

    # 2. Setup Optimizer (Only training the new classification head)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=cfg['experiment']['learning_rate'])
    
    epochs = cfg['experiment']['num_epochs']
    best_val_acc = 0.0
    
    # 3. Full Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # PyTorch 3D models expect input shape: (Batch, Channels, Frames, H, W)
            # Our preprocessor makes (Frames, Channels, H, W), so we transpose
            inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
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
        
        # 4. Validation Loop
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct_val / total_val
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # 5. Save the ONE model weight to the models directory
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = os.path.join(cfg['paths']['weights_dir'], "exp1_baseline_frozen.pth")
            torch.save(model.state_dict(), out_path)
            logger.info(f"🌟 New best model saved to {out_path}")

if __name__ == "__main__":
    run_exp1_frozen()