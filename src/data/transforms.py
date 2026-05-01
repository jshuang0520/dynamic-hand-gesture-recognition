import torch
import torchvision.transforms.functional as F

class OfflineVideoProcessor:
    """Handles spatial transformations and noise offline."""
    def __init__(self, target_h=224, target_w=224, pad_val=10, contrast=1.1, noise_std=0.05):
        self.target_size = (target_h, target_w)
        self.pad_val = pad_val
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, tensor_frames):
        # 1. Spatial: Pad and then Center Crop to target
        frames = F.pad(tensor_frames, padding=self.pad_val)
        frames = F.center_crop(frames, output_size=self.target_size)
        
        # 2. Appearance: Adjust Contrast
        frames = F.adjust_contrast(frames, self.contrast)
        
        # 3. Noise: Gaussian Noise for robustness
        frames += torch.randn_like(frames) * self.noise_std
        
        return torch.clamp(frames, 0.0, 1.0)