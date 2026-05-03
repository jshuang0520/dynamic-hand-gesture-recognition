import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utilities.logger import get_logger

logger = get_logger("DATALOADER")

class JesterTensorDataset(Dataset):
    def __init__(self, csv_path, processed_dir, target_classes):
        self.data = pd.read_csv(csv_path)
        
        # --- FIX: Drop empty rows to completely prevent the KeyError: 'nan' ---
        self.data.dropna(subset=['video_id', 'gesture'], inplace=True)
        
        self.processed_dir = processed_dir
        
        # Keep both the list and the dictionary for safety
        self.classes = target_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(target_classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Safely convert ID to string (prevents float issues like 1037.0)
        vid = row['video_id']
        vid_id = str(int(vid)) if isinstance(vid, float) else str(vid)
        
        label_str = str(row['gesture']).strip()
        label_idx = self.class_to_idx[label_str]

        path = os.path.join(self.processed_dir, f"{vid_id}.pt")
        
        # --- YOUR RESTORED LOGIC: Safety fallback for missing tensors ---
        if not os.path.exists(path):
            return torch.zeros((16, 3, 224, 224)), label_idx, vid_id

        # --- HANNAH'S FIX: Return 3 items (Tensor, Label, ID) ---
        return torch.load(path), label_idx, vid_id

def get_loaders(config):
    p_path = config['paths']['processed_dir']
    ann_path = config['paths']['annotations_dir'] 
    
    bz = config['experiment']['batch_size']
    classes = config['experiment']['target_classes']
    
    # Check for 'validation.csv' vs 'val.csv' dynamically
    val_csv_name = "validation.csv" if os.path.exists(os.path.join(ann_path, "validation.csv")) else "val.csv"
    
    train_ds = JesterTensorDataset(os.path.join(ann_path, "train.csv"), p_path, classes)
    val_ds = JesterTensorDataset(os.path.join(ann_path, val_csv_name), p_path, classes)
    
    return (
        DataLoader(train_ds, batch_size=bz, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=bz, shuffle=False, num_workers=4, pin_memory=True)
    )