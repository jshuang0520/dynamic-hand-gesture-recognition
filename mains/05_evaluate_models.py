import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision.models.video import r3d_18
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import JesterTensorDataset
from torch.utils.data import DataLoader
from src.models.hybrid import HybridResNetLSTM

def run_evaluation():
    cfg = load_config()
    logger = get_logger("05_EVALUATION", log_dir=cfg['paths']['logs_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ann_path = cfg['paths']['annotations_dir']
    p_path = cfg['paths']['processed_dir']
    classes = cfg['experiment']['target_classes']
    
    test_ds = JesterTensorDataset(os.path.join(ann_path, "test.csv"), p_path, classes)
    test_loader = DataLoader(test_ds, batch_size=cfg['experiment']['batch_size'], shuffle=False)
    
    model_constructors = {
        "exp1_frozen": ("exp1_baseline_frozen.pth", lambda: r3d_18()),
        "exp2_finetune": ("exp2_finetune_best.pth", lambda: r3d_18()),
        "exp3_hybrid": ("exp3_hybrid_best.pth", lambda: HybridResNetLSTM(num_classes=cfg['experiment']['num_classes']))
    }

    for exp_name, (weight_file, get_model_func) in model_constructors.items():
        weight_path = os.path.join(cfg['paths']['weights_dir'], weight_file)
        
        if not os.path.exists(weight_path):
            logger.warning(f"⚠️ Weights missing for {exp_name} at {weight_path}")
            continue
            
        model = get_model_func()
        if "exp1" in exp_name or "exp2" in exp_name:
            # Match the Dropout structure added in training
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc[1].in_features if isinstance(model.fc, nn.Sequential) else model.fc.in_features, cfg['experiment']['num_classes'])
            )
            
        # --- FIX FOR HANNAH: Load from checkpoint dictionary ---
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        preds, labels = [], []
        target_size = cfg['experiment']['model_input_size']
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                if "exp1" in exp_name or "exp2" in exp_name:
                    inputs = inputs.permute(0, 2, 1, 3, 4) 
                    # Apply the same interpolation used in training
                    inputs = F.interpolate(inputs, size=(16, target_size, target_size), mode='trilinear', align_corners=False)
                
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
        
        acc = accuracy_score(labels, preds)
        logger.info(f"--- 📊 {exp_name.upper()} FINAL TEST ACCURACY: {acc*100:.2f}% ---")

if __name__ == "__main__":
    run_evaluation()