import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3Baseline(nn.Module):
    def __init__(self, num_classes=14):
        super(InceptionV3Baseline, self).__init__()
        
        # 1. Load Pre-trained InceptionV3 (let it load the aux branch initially)
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        
        # 2. Manually disable the auxiliary branch
        # This prevents Inception from returning a tuple of (output, aux_output) during training
        self.inception.aux_logits = False
        self.inception.AuxLogits = None
        
        # 3. We want to extract features, not classify ImageNet categories.
        # InceptionV3's final layer is named 'fc'. We replace it with an Identity layer.
        self.inception.fc = nn.Identity()
        
        # 4. Final Classification Head
        # Maps the temporally pooled 2048-dim vector to our 14 dynamic gesture classes
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Expected input 'x' shape: [batch_size, 16, 3, 299, 299]
        batch_size, num_frames, C, H, W = x.shape
        
        # Collapse Batch and Time dimensions
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Spatial Extraction -> [batch_size * 16, 2048]
        spatial_features = self.inception(x)
        
        # Unflatten back to separate temporal dimension -> [batch_size, 16, 2048]
        spatial_features = spatial_features.view(batch_size, num_frames, 2048)
        
        # Temporal Pooling (Global Average Pooling across the 16 frames) -> [batch_size, 2048]
        pooled_features = spatial_features.mean(dim=1)
        
        # Final Classification -> [batch_size, 14]
        logits = self.classifier(pooled_features)
        
        return logits

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # Simulate a tiny batch: 2 videos, 16 frames, 3 channels, 299x299
    dummy_input = torch.randn(2, 16, 3, 299, 299)
    model = InceptionV3Baseline(num_classes=14)
    
    print("Testing InceptionV3 Baseline...")
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [2, 14])")
    print("Test passed! Ready for training.")