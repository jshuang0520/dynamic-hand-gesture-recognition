import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.transforms import OfflineVideoProcessor

def get_sampled_indices(total_frames, target_count):
    if total_frames == 0: return None
    if total_frames >= target_count:
        indices = np.linspace(0, total_frames - 1, target_count).astype(int)
    else:
        indices = np.arange(total_frames)
        padding = np.full(target_count - total_frames, total_frames - 1)
        indices = np.concatenate([indices, padding])
    return np.clip(indices, 0, total_frames - 1)

def run_preprocessing():
    cfg = load_config()
    logger = get_logger("01_PREPROCESS", log_dir=cfg['paths']['logs_dir'])
    
    videos_in = cfg['paths']['videos_dir']
    proc_out = cfg['paths']['processed_dir']
    target_f = cfg['experiment']['frames_per_video']
    
    os.makedirs(proc_out, exist_ok=True)
    processor = OfflineVideoProcessor() 
    
    # --- FIX FOR HANNAH: Pulling ImageNet stats & Size dynamically ---
    norm_mean = cfg['preprocessing']['normalize_mean']
    norm_std = cfg['preprocessing']['normalize_std']
    prep_size = cfg['experiment']['preprocess_size']
    
    to_tensor = T.Compose([
        T.Resize((prep_size, prep_size)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std) 
    ])

    splits = ['train', 'val', 'test']
    total_processed = 0
    
    for split in splits:
        split_dir = os.path.join(videos_in, split)
        if not os.path.exists(split_dir): 
            continue
            
        video_ids = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        logger.info(f"Processing {len(video_ids)} videos in {split} split...")
        
        for vid_id in video_ids:
            vid_path = os.path.join(split_dir, vid_id)
            all_f = sorted([f for f in os.listdir(vid_path) if f.endswith('.jpg')])
            
            if len(all_f) == 0: continue
            
            try:
                indices = get_sampled_indices(len(all_f), target_f)
                frames = [to_tensor(Image.open(os.path.join(vid_path, all_f[i])).convert('RGB')) for i in indices]
                
                video_tensor = processor(torch.stack(frames))
                torch.save(video_tensor, os.path.join(proc_out, f"{vid_id}.pt"))
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Failed processing video {vid_id}: {e}")
    logger.info(f"Successfully preprocessed {total_processed} videos.")

if __name__ == "__main__":
    run_preprocessing()