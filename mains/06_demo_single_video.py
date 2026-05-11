import os
import argparse
import sys

# 1. Provide immediate feedback before heavy imports begin
print("\n⏳ Starting Demo... Importing PyTorch and Deep Learning libraries (This takes a few seconds)...\n")

# --- BULLETPROOF CONFIGURATION FIX ---
if "ENV" not in os.environ:
    os.environ["ENV"] = "prod"

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import numpy as np

from utilities.config_parser import load_config
from src.data.transforms import OfflineVideoProcessor

def run_demo(video_folder_path, model_choice="exp3"):
    print("="*60)
    print("🚀 JESTER DYNAMIC HAND GESTURE RECOGNITION DEMO")
    print("="*60)
    
    # ==========================================================
    # STEP 1: INITIALIZATION
    # ==========================================================
    print("\n[STEP 1] ENVIRONMENT INITIALIZATION")
    print("Loading configuration variables...")
    cfg = load_config()
    
    device = torch.device("cpu")
    print("🖥️  Hardware: Forced CPU (Optimized for interactive demo wait times)")
    classes = cfg['experiment']['target_classes']

    # ==========================================================
    # STEP 2: VIDEO SAMPLING & PREPROCESSING
    # ==========================================================
    print(f"\n[STEP 2] VIDEO SAMPLING & PREPROCESSING")
    print(f"📂 Target Video Directory: \n   -> {video_folder_path}")
    
    all_f = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.jpg')])
    if len(all_f) == 0:
        print("❌ ERROR: No JPG frames found in folder.")
        return
        
    print(f"🎞️  Total available raw frames: {len(all_f)}")

    # Calculate and show exactly which 16 frames are pulled
    indices = np.linspace(0, len(all_f) - 1, 16).astype(int)
    print(f"⏱️  Temporal Subsampling (Selected Indices): \n   -> {indices.tolist()}")
    
    to_tensor = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("🪄  Applying Spatial Transforms (Resize, Tensor Conversion, Normalization)...")
    frames = [to_tensor(Image.open(os.path.join(video_folder_path, all_f[i])).convert('RGB')) for i in indices]
    processor = OfflineVideoProcessor(target_h=224, target_w=224)
    video_tensor = processor(torch.stack(frames)).unsqueeze(0) 
    print(f"✅ Preprocessing Complete! Final Tensor Shape: {list(video_tensor.shape)}")

    # ==========================================================
    # STEP 3: MODEL ARCHITECTURE & WEIGHT LOADING
    # ==========================================================
    print(f"\n[STEP 3] MODEL LOADING ({model_choice.upper()})")
    
    if model_choice == "exp3":
        from src.models.hybrid import HybridResNetLSTM
        model = HybridResNetLSTM(
            num_classes=cfg['experiment']['num_classes'],
            hidden_dim=cfg['experiment'].get('lstm_hidden_dim', 256),
            projection_dim=cfg['experiment'].get('projection_dim', 512)
        )
        weight_path = os.path.join(cfg['paths']['weights_dir'], "exp3_hybrid_best.pth")
    else:
        print("Please choose exp3.")
        return
    
    print(f"🔗 Target Weights Path: \n   -> {weight_path}")
    
    if not os.path.exists(weight_path):
        print(f"❌ ERROR: Weights not found at {weight_path}")
        return

    print("⏳ Loading weights into architecture...")
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Architecture built and weights loaded successfully.")

    # ==========================================================
    # STEP 4: INFERENCING
    # ==========================================================
    print(f"\n[STEP 4] INFERENCING")
    print("🧠 Forward-passing the 16-frame tensor through the network...")
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probabilities, 1)
        
    predicted_gesture = classes[top_class_idx.item()]
    confidence = top_prob.item() * 100
    
    print("\n" + "="*60)
    print(f"🎯 FINAL PREDICTION :  {predicted_gesture.upper()}")
    print(f"📊 MODEL CONFIDENCE :  {confidence:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single Jester video.")
    parser.add_argument("--video", type=str, required=True, 
                        help="Relative path to the video folder (e.g., prod/raw_data/videos/test/137745)")
    parser.add_argument("--model", type=str, default="exp3", choices=["exp2", "exp3"], 
                        help="Choose the model to evaluate (exp2 or exp3)")
    
    args = parser.parse_args()

    # Strictly enforce the .env file
    output_root = os.environ.get("OUTPUT_ROOT")
    if not output_root:
        print("❌ ERROR: OUTPUT_ROOT environment variable is missing!")
        print("Please ensure your .env file is loaded (e.g., run 'set -a; source .env; set +a' first).")
        sys.exit(1)
        
    output_root = output_root.strip().replace('\r', '')
    TARGET_VIDEO = os.path.join(output_root, args.video)
    
    if not os.path.exists(TARGET_VIDEO):
        print(f"❌ ERROR: Cannot find the directory: {TARGET_VIDEO}")
    else:
        run_demo(TARGET_VIDEO, model_choice=args.model)