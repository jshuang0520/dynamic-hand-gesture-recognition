import os
import pandas as pd
import torch
from tqdm import tqdm
import concurrent.futures
from utilities.config_parser import load_config
from src.data.mediapipe_etl import SkeletonExtractor

def process_video_worker(args):
    """Worker function for multiprocessing. Needs to instantiate its own extractor."""
    video_id, frames_dir, output_dir, num_frames = args
    
    video_path = os.path.join(frames_dir, str(video_id))
    out_path = os.path.join(output_dir, f"{video_id}.pt")
    
    # Skip if already processed or missing
    if os.path.exists(out_path): return True
    if not os.path.exists(video_path): return False
        
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
    if not frame_files: return False

    indices = torch.linspace(0, len(frame_files) - 1, steps=num_frames).long()
    selected = [frame_files[i] for i in indices]

    extractor = SkeletonExtractor()
    video_skeleton = []
    last_valid = [0.0] * 63 

    for f_name in selected:
        coords = extractor.extract_from_frame(os.path.join(video_path, f_name))
        if coords is not None:
            last_valid = coords.flatten().tolist()
        video_skeleton.append(last_valid)

    extractor.close()
    
    # Save tensor [16, 63]
    torch.save(torch.tensor(video_skeleton, dtype=torch.float32), out_path)
    return True

if __name__ == "__main__":
    print("[INFO] Initializing Parallel Keypoint Extraction...")
    config = load_config()
    
    frames_dir = config['data']['frames_dir']
    out_dir = config['data']['keypoints_dir']
    num_workers = config['extraction']['num_workers']
    num_frames = config['data']['frames_per_video']
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Video IDs
    dfs = [pd.read_csv(config['data'][k]) for k in ['train_csv', 'val_csv', 'test_csv']]
    video_ids = pd.concat(dfs)['video_id'].unique()
    
    # Prepare arguments for multiprocessing
    task_args = [(vid, frames_dir, out_dir, num_frames) for vid in video_ids]
    
    print(f"[INFO] Processing {len(video_ids)} videos using {num_workers} CPU cores.")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_video_worker, task_args), total=len(task_args)))
        success_count = sum(1 for r in results if r)
            
    print(f"[SUCCESS] Extracted {success_count}/{len(video_ids)} videos to {out_dir}")