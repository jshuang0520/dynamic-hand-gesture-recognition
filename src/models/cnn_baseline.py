import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3Baseline(nn.Module):
    def __init__(self, num_classes=14):
        super(InceptionV3Baseline, self).__init__()
        
        # Load Pre-trained InceptionV3
        # aux_logits=False prevents Inception from returning a tuple during training
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
        
        # We want to extract features, not classify ImageNet categories.
        # InceptionV3's final layer is named 'fc'. We replace it with an Identity layer.
        # Passing an image through self.inception now outputs a [2048] feature vector.
        self.inception.fc = nn.Identity()
        
        # Final Classification Head
        # Maps the temporally pooled 2048-dim vector to our 14 dynamic gesture classes
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Expected input 'x' shape: [batch_size, 16, 3, 299, 299]
        batch_size, num_frames, C, H, W = x.shape
        
        # 1. Collapse Batch and Time dimensions
        # InceptionV3 only processes 2D images, so we pretend we have a huge batch of individual images
        x = x.view(batch_size * num_frames, C, H, W)
        
        # 2. Spatial Extraction
        # Shape becomes: [batch_size * 16, 2048]
        spatial_features = self.inception(x)
        
        # 3. Unflatten back to separate temporal dimension
        # Shape becomes: [batch_size, 16, 2048]
        spatial_features = spatial_features.view(batch_size, num_frames, 2048)
        
        # 4. Temporal Pooling (Global Average Pooling across the 16 frames)
        # We average along dimension 1 (the num_frames dimension)
        # Shape becomes: [batch_size, 2048]
        pooled_features = spatial_features.mean(dim=1)
        
        # 5. Final Classification
        # Shape becomes: [batch_size, 14]
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