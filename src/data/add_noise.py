import random
import torch
import torchvision.transforms.functional as F

class JesterAugmenter:
    """Online data augmentation pipeline (noise, contrast, cropping) to act as a regularizer."""
    def __init__(self, severity=0.5):
        self.severity = severity

    def __call__(self, tensor_frames):
        if random.random() > self.severity: 
            return tensor_frames
            
        # 1. Contrast and Gaussian Noise
        tensor_frames = F.adjust_contrast(tensor_frames, random.uniform(0.8, 1.2))
        tensor_frames += torch.randn_like(tensor_frames) * 0.05
        
        # 2. Random Cropping (Simulated via padding)
        _, _, h, w = tensor_frames.shape
        tensor_frames = F.pad(tensor_frames, padding=10)
        tensor_frames = F.center_crop(tensor_frames, output_size=[h, w])
        
        return torch.clamp(tensor_frames, 0.0, 1.0)