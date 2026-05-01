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
    videos_in = cfg['paths']['videos_dir']
    proc_out = cfg['paths']['processed_dir']
    target_f = cfg['experiment']['frames_per_video']
    
    os.makedirs(proc_out, exist_ok=True)
    processor = OfflineVideoProcessor()
    to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    # Look through the new subdirectories
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(videos_in, split)
        if not os.path.exists(split_dir): 
            continue
            
        video_ids = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for vid_id in video_ids:
            vid_path = os.path.join(split_dir, vid_id)
            all_f = sorted([f for f in os.listdir(vid_path) if f.endswith('.jpg')])
            
            if len(all_f) == 0:
                logger(f"⚠️ Warning: Skipping empty folder {vid_id} in {split}")
                continue
            
            indices = get_sampled_indices(len(all_f), target_f)
            
            try:
                frames = []
                for i in indices:
                    img_path = os.path.join(vid_path, all_f[i])
                    img = Image.open(img_path).convert('RGB')
                    frames.append(to_tensor(img))
                
                video_tensor = processor(torch.stack(frames))
                out_path = os.path.join(proc_out, f"{vid_id}.pt")
                torch.save(video_tensor, out_path)
                logger(f"✅ Processed {split}/{vid_id}: {len(all_f)} -> {target_f} frames")
                
            except Exception as e:
                logger(f"❌ Error processing {vid_id}: {str(e)}")

if __name__ == "__main__":
    run_preprocessing()