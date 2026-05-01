import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from src.data.transforms import JesterAugmenter
from utilities.logger import get_logger

logger = get_logger("DATALOADER")

class JesterDataset(Dataset):
    """PyTorch Dataset for loading sequential video frames."""
    def __init__(self, csv_path, frames_dir, target_classes, num_frames=16, is_train=True):
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.data = pd.read_csv(csv_path)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(target_classes)}
        
        self.resize = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augmenter = JesterAugmenter(severity=0.5) if is_train else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vid_id = str(row['video_id'])
        label_idx = self.class_to_idx[row['label']]
        
        vid_dir = os.path.join(self.frames_dir, vid_id)
        
        # --- Handle Missing Data Gracefully ---
        if not os.path.exists(vid_dir):
            return torch.zeros((self.num_frames, 3, 224, 224)), label_idx
            
        frame_files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
        if len(frame_files) == 0:
            return torch.zeros((self.num_frames, 3, 224, 224)), label_idx

        # --- Uniform Frame Sampling ---
        indices = torch.linspace(0, len(frame_files) - 1, self.num_frames).long()
        frames = [self.resize(Image.open(os.path.join(vid_dir, frame_files[i])).convert('RGB')) for i in indices]
        
        video_tensor = torch.stack(frames)
        if self.augmenter:
            video_tensor = self.augmenter(video_tensor)
            
        return video_tensor, label_idx

def build_dataloader(base_dir, split, config, is_train=True):
    """Helper factory for clean dataloader initialization."""
    csv_path = os.path.join(base_dir, "jester_splits", f"{split}.csv")
    frames_dir = os.path.join(base_dir, "jester_subsample")
    
    logger(f"Mounting {split} dataset from {csv_path}")
    dataset = JesterDataset(
        csv_path=csv_path,
        frames_dir=frames_dir,
        target_classes=config['experiment']['target_classes'],
        num_frames=config['experiment']['frames_per_video'],
        is_train=is_train
    )
    return DataLoader(dataset, batch_size=config['experiment']['batch_size'], shuffle=is_train, num_workers=4)