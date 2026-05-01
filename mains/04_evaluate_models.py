import os
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.models.hybrid import HybridResNetLSTM
from src.data.dataset import build_dataloader

logger = get_logger("EVALUATION")

def load_architecture(model_filename, num_classes):
    """Dynamically initializes the correct PyTorch architecture based on the weight filename."""
    if "exp1" in model_filename or "exp2" in model_filename:
        # Both Baseline 1 and 2 utilize the 3D ResNet-18 architecture
        model = r3d_18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif "exp3" in model_filename:
        # The Proposed Architecture
        return HybridResNetLSTM(num_classes=num_classes)
    else:
        raise ValueError(f"Unrecognized model architecture mapped to: {model_filename}")

def evaluate_loop(model, dataloader, device):
    """Executes a strict inference loop and calculates core classification metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            outputs = model(data)
            
            # Extract the highest probability class
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    # Calculate weighted metrics across the 4 distinct gesture classes
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )
    
    return acc, precision, recall, f1

if __name__ == "__main__":
    logger("Initializing Model Evaluation Phase across all Experiments")
    
    # --- SECTION 1: Configuration & Setup ---
    cfg = load_config()
    exp_cfg, paths_cfg = cfg['experiment'], cfg['paths']
    
    lock_seeds(exp_cfg['seed'])
    
    # Graceful hardware fallback for evaluation
    device = torch.device("cuda:0" if exp_cfg['execution_mode'] == "gpu" and torch.cuda.is_available() else "cpu")
    logger(f"Inference hardware selected: {device}")
    
    # --- SECTION 2: Load Test Dataset ---
    dataset_base = paths_cfg['dataset_dir'] 
    logger("Mounting Test DataLoader (Augmentations Disabled)")
    # We load 'test.csv' and ensure is_train=False so JesterAugmenter is bypassed
    test_loader = build_dataloader(dataset_base, "test", cfg, is_train=False)
    
    # --- SECTION 3: Model Iteration & Benchmarking ---
    weights_dir = paths_cfg['weights_dir']
    logger(f"Scanning for trained weights in {weights_dir}...")
    
    # The expected output artifacts from Experiments 1, 2, and 3
    target_models = [
        "exp1_frozen_best.pth", 
        "exp2_finetune_best.pth", 
        "exp3_hybrid_best.pth"
    ]
    
    for weight_file in target_models:
        full_path = os.path.join(weights_dir, weight_file)
        
        if not os.path.exists(full_path):
            logger(f"Skipping {weight_file} - Weights not found. Run the training script first.", "WARNING")
            continue
            
        logger(f"--- Benchmarking {weight_file} ---")
        
        try:
            # 1. Instantiate the correct skeleton
            model = load_architecture(weight_file, exp_cfg['num_classes'])
            
            # 2. Load the trained parameters securely onto the target device
            model.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
            model = model.to(device)
            
            # 3. Execute inference
            logger(f"Running inference over {len(test_loader.dataset)} test samples...")
            acc, prec, rec, f1 = evaluate_loop(model, test_loader, device)
            
            # 4. Report Results
            logger(f"RESULTS for {weight_file}:")
            logger(f"  -> Accuracy : {acc * 100:.2f}%")
            logger(f"  -> Precision: {prec:.4f}")
            logger(f"  -> Recall   : {rec:.4f}")
            logger(f"  -> F1-Score : {f1:.4f}")
            
        except Exception as e:
            logger(f"Failed to evaluate {weight_file}. Error: {str(e)}", "ERROR")
            
    logger("Evaluation phase completely finalized.")