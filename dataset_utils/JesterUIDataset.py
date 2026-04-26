import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class JesterUIDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, label_map=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map

        if self.label_map is None:
            raise ValueError("label_map must be provided to prevent class mismatch.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        video_id = str(self.dataframe.iloc[idx]['video_id'])
        label_str = self.dataframe.iloc[idx]['gesture']
        label_idx = self.label_map[label_str]
        
        video_path = os.path.join(self.root_dir, video_id)
        
        if not os.path.exists(video_path):
            return torch.zeros((3, 16, 112, 112)), label_idx

        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(frame_files) == 0:
            return torch.zeros((3, 16, 112, 112)), label_idx

        # Uniform frame sampling
        if len(frame_files) >= 16:
            indices = torch.linspace(0, len(frame_files) - 1, steps=16).long()
            selected_files = [frame_files[i] for i in indices]
        else:
            selected_files = frame_files.copy()
            while len(selected_files) < 16:
                selected_files.append(frame_files[-1])

        frames = []
        for file_name in selected_files:
            img_path = os.path.join(video_path, file_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames.append(img)
            except Exception:
                frames.append(torch.zeros((3, 112, 112)))

        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        return video_tensor, label_idx
    
