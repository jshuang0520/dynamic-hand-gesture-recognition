import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class JesterTensorDataset(Dataset):
    def __init__(self, csv_path, processed_dir, target_classes):
        self.data = pd.read_csv(csv_path)
        self.processed_dir = processed_dir
        self.class_to_idx = {cls: i for i, cls in enumerate(target_classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Depending on your CSV structure, grab the ID and Label
        video_id = str(row['video_id']) 
        label_str = row['gesture']
        label_idx = self.classes.index(label_str)
        
        # Load the pre-processed tensor
        tensor_path = os.path.join(self.processed_dir, f"{video_id}.pt")
        video_tensor = torch.load(tensor_path)
        
        # --- HANNAH'S CRITICAL FIX: Return 3 items instead of 2 ---
        # Old: return video_tensor, label_idx
        return video_tensor, label_idx, video_id

def get_loaders(config):
    p_path = config['paths']['processed_dir']
    
    # Use the newly defined annotations directory
    ann_path = config['paths']['annotations_dir'] 
    
    bz = config['experiment']['batch_size']
    classes = config['experiment']['target_classes']
    
    train_ds = JesterTensorDataset(os.path.join(ann_path, "train.csv"), p_path, classes)
    val_ds = JesterTensorDataset(os.path.join(ann_path, "val.csv"), p_path, classes)
    
    return (
        DataLoader(train_ds, batch_size=bz, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=bz, shuffle=False, num_workers=4)
    )