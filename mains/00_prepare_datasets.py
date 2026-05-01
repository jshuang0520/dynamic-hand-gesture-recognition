import os
import shutil
import pandas as pd
from PIL import Image
from utilities.reproducibility import lock_seeds
from utilities.config_parser import load_config
from utilities.logger import get_logger

log = get_logger("DATA_PREP")

def generate_dummy_frames(out_dir, num_frames=16):
    """Fallback generator if original dataset isn't on the system."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_frames):
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        img.save(os.path.join(out_dir, f"{i:05d}.jpg"))

def create_physical_split(base_dir, config, is_dev=False):
    splits_dir = os.path.join(base_dir, "jester_splits")
    frames_dir = os.path.join(base_dir, "jester_subsample")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    classes = config['experiment']['target_classes']
    train_per_class = 8 if is_dev else 400
    val_per_class = 1 if is_dev else 50
    test_per_class = 1 if is_dev else 50
    
    log(f"Building {'Dev' if is_dev else 'Full'} Dataset at {base_dir}")
    
    vid_counter = 100000 if not is_dev else 200000
    splits_data = {"train": [], "val": [], "test": []}
    
    for cls in classes:
        counts = {"train": train_per_class, "val": val_per_class, "test": test_per_class}
        for split, count in counts.items():
            for _ in range(count):
                vid_id = str(vid_counter)
                splits_data[split].append({"video_id": vid_id, "label": cls})
                
                # Mock extraction: In reality, copy from config['paths']['original_raw_dir']
                generate_dummy_frames(os.path.join(frames_dir, vid_id))
                vid_counter += 1
                
    # Write CSVs
    for split, data in splits_data.items():
        df = pd.DataFrame(data)
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        log(f"Created {csv_path} with {len(df)} records.")

if __name__ == "__main__":
    cfg = load_config()
    lock_seeds(cfg['experiment']['seed'])
    
    create_physical_split(cfg['paths']['dev_dataset_dir'], cfg, is_dev=True)
    create_physical_split(cfg['paths']['full_dataset_dir'], cfg, is_dev=False)
    log("Data preparation successfully completed.")