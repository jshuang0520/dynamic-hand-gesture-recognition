import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.transforms import OfflineVideoProcessor

logger = get_logger("01_PREPROCESS")

def get_sampled_indices(total_frames, target_count):
    """Uniformly samples indices across the video duration."""
    if total_frames >= target_count:
        return np.linspace(0, total_frames - 1, target_count).astype(int)
    # Pad by repeating last frame if video is too short
    return np.concatenate([np.arange(total_frames), [total_frames - 1] * (target_count - total_frames)])

def run_preprocessing():
    cfg = load_config()
    raw_in = cfg['paths']['raw_data_dir']
    proc_out = cfg['paths']['processed_dir']
    target_f = cfg['experiment']['frames_per_video']
    os.makedirs(proc_out, exist_ok=True)
    
    processor = OfflineVideoProcessor()
    resize_to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    for vid_id in os.listdir(raw_in):
        vid_path = os.path.join(raw_in, vid_id)
        if not os.path.isdir(vid_path): continue
        
        all_f = sorted([f for f in os.listdir(vid_path) if f.endswith('.jpg')])
        indices = get_sampled_indices(len(all_f), target_f)
        
        frames = []
        for i in indices:
            img = Image.open(os.path.join(vid_path, all_f[i])).convert('RGB')
            frames.append(resize_to_tensor(img))
        
        # Apply Pad, Crop, Contrast, and Noise in one stack
        video_tensor = processor(torch.stack(frames))
        torch.save(video_tensor, os.path.join(proc_out, f"{vid_id}.pt"))
        logger(f"Saved Preprocessed Tensor: {vid_id}.pt")

if __name__ == "__main__":
    run_preprocessing()