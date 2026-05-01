import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from utilities.logger import get_logger

logger = get_logger("MODEL_HYBRID")

class HybridResNetLSTM(nn.Module):
    """Proposed architecture: Pre-trained ResNet50 spatial extractor + Temporal LSTM."""
    def __init__(self, num_classes=4, hidden_dim=256):
        super(HybridResNetLSTM, self).__init__()
        logger.info("Initializing Hybrid ResNet-50 + LSTM Architecture")
        
        # Spatial Feature Extractor
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() 
        
        # Temporal Modeler & Classifier
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Flatten time into batch for CNN extraction
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        spatial_feats = self.backbone(x)
        
        # Reshape to sequence for LSTM
        spatial_feats = spatial_feats.view(b, f, -1)
        lstm_out, (hn, cn) = self.lstm(spatial_feats)
        
        return self.classifier(hn[-1])