import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision.models.video import r3d_18

# --- CRITICAL HPC FIX: Headless backend for plotting ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import JesterTensorDataset
from torch.utils.data import DataLoader
from src.models.hybrid import HybridResNetLSTM

def generate_plots(exp_name, cfg, true_labels, pred_labels, classes, logger):
    """Generates and saves PNG plots for the report without opening GUI windows."""
    logs_dir = cfg['paths']['logs_dir']
    
    # Plot Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{exp_name.upper()} - Confusion Matrix")
    plt.xlabel("Predicted Gesture")
    plt.ylabel("Actual Gesture")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = os.path.join(logs_dir, f"{exp_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"📊 Saved Confusion Matrix Plot: {cm_path}")

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

    # --- HANNAH'S FIX: Dedicated directory for test predictions ---
    results_dir = os.path.join(cfg['paths']['logs_dir'], "test_predictions")
    os.makedirs(results_dir, exist_ok=True)

    for exp_name, (weight_file, get_model_func) in model_constructors.items():
        weight_path = os.path.join(cfg['paths']['weights_dir'], weight_file)
        
        if not os.path.exists(weight_path):
            logger.warning(f"⚠️ Weights missing for {exp_name} at {weight_path}")
            continue
            
        model = get_model_func()
        if "exp1" in exp_name or "exp2" in exp_name:
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc[1].in_features if isinstance(model.fc, nn.Sequential) else model.fc.in_features, cfg['experiment']['num_classes'])
            )
            
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # --- HANNAH'S FIX: Tracking Video IDs ---
        preds, labels, video_ids = [], [], []
        target_size = cfg['experiment']['model_input_size']
        
        with torch.no_grad():
            for batch in test_loader:
                # --- SAFETY NET: Handles both 2-item and 3-item dataset returns ---
                if len(batch) == 3:
                    inputs, targets, ids = batch
                else:
                    inputs, targets = batch
                    ids = ["unknown"] * inputs.size(0)

                if "exp1" in exp_name or "exp2" in exp_name:
                    inputs = inputs.permute(0, 2, 1, 3, 4) 
                    inputs = F.interpolate(inputs, size=(16, target_size, target_size), mode='trilinear', align_corners=False)
                
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(targets.cpu().numpy())
                video_ids.extend(ids)
        
        acc = accuracy_score(labels, preds)
        logger.info(f"--- 📊 {exp_name.upper()} FINAL TEST ACCURACY: {acc*100:.2f}% ---")

        # Map to string labels for readable CSVs and Classification Reports
        true_labels_str = [classes[i] for i in labels]
        pred_labels_str = [classes[i] for i in preds]
        
        report = classification_report(true_labels_str, pred_labels_str, target_names=classes)
        logger.info(f"\nClassification Report for {exp_name}:\n{report}")

        # --- HANNAH'S FIX: Pandas DataFrame Export ---
        df = pd.DataFrame({
            "video_id": video_ids, 
            "true_label": true_labels_str,  # Saved as readable strings
            "pred_label": pred_labels_str
        })
        csv_path = os.path.join(results_dir, f"{exp_name}_test_predictions.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Saved predictions to {csv_path}")
        
        # Generate the Auto-Plots
        generate_plots(exp_name, cfg, true_labels_str, pred_labels_str, classes, logger)

if __name__ == "__main__":
    run_evaluation()