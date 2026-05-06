import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from utilities.logger import get_logger

logger = get_logger("MODEL_HYBRID")

class HybridResNetLSTM(nn.Module):
    """Upgraded Architecture: ResNet50 + Projection Head + Bi-Directional Deep LSTM"""
    def __init__(self, num_classes=4, hidden_dim=256, projection_dim=512, dropout_rate=0.5):
        super(HybridResNetLSTM, self).__init__()
        logger.info("Initializing Upgraded BI-DIRECTIONAL Hybrid ResNet-50 + LSTM + Projection")
        
        # 1. Spatial Feature Extractor (Frozen or gently fine-tuned)
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() 
        
        # 2. NEW: The Projection Head 
        # Compresses 2048-dim ImageNet features down to 512-dim gesture features
        # LayerNorm acts as a "volume control" before hitting the LSTM
        # Use the variable instead of hardcoded numbers
        self.projection = nn.Sequential(
            nn.Linear(2048, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim)
        )
        
        # 3. Upgraded Temporal Modeler (Now takes 512 instead of 2048)
        self.lstm = nn.LSTM(
            input_size=projection_dim, # <-- Matches projection output
            hidden_size=hidden_dim, 
            num_layers=2,        # Deep temporal modeling
            batch_first=True, 
            dropout=0.3,         # Internal dropout between LSTM layers
            bidirectional=True   # Reads trajectory forwards AND backwards
        )
        
        # 4. Robust Classifier
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Bi-directional LSTM outputs TWO hidden states concatenated together
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        b, f, c, h, w = x.shape
        # Flatten batch and frames for the 2D CNN
        x = x.view(b * f, c, h, w)
        
        # Extract features: Shape becomes (b*f, 2048)
        spatial_feats = self.backbone(x)
        
        # Compress and normalize: Shape becomes (b*f, 512)
        spatial_feats = self.projection(spatial_feats)
        
        # Reshape back to sequence: (Batch, Frames, 512)
        spatial_feats = spatial_feats.view(b, f, -1)
        
        # Pass through the Bi-Directional LSTM
        lstm_out, (hn, cn) = self.lstm(spatial_feats)
        
        # Grab the final forward hidden state and final backward hidden state
        final_forward = hn[-2, :, :]
        final_backward = hn[-1, :, :]
        
        # Concatenate them together
        final_state = torch.cat((final_forward, final_backward), dim=1)
        
        # Classify
        out = self.dropout(final_state)
        return self.classifier(out)