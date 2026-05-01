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
        vid_id = str(row['video_id'])
        label_idx = self.class_to_idx[row['gesture']]
        
        path = os.path.join(self.processed_dir, f"{vid_id}.pt")
        if not os.path.exists(path):
            return torch.zeros((16, 3, 224, 224)), label_idx
            
        return torch.load(path), label_idx

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