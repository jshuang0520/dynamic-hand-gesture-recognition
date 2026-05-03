import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from utilities.logger import get_logger

logger = get_logger("MODEL_HYBRID")

class HybridResNetLSTM(nn.Module):
    """Upgraded Architecture: ResNet50 + LayerNorm + Bi-Directional Deep LSTM"""
    def __init__(self, num_classes=4, hidden_dim=256, dropout_rate=0.5):
        super(HybridResNetLSTM, self).__init__()
        logger.info("Initializing Upgraded BI-DIRECTIONAL Hybrid ResNet-50 + LSTM")
        
        # 1. Spatial Feature Extractor
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() 
        
        # 2. Feature Stabilizer (Prevents ResNet features from blowing up the LSTM)
        self.layer_norm = nn.LayerNorm(2048)
        
        # 3. Upgraded Temporal Modeler
        # - num_layers=2: Deeper temporal understanding
        # - bidirectional=True: Reads the gesture forwards and backwards
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3,         # Internal dropout between LSTM layers
            bidirectional=True
        )
        
        # 4. Robust Classifier
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Because it's bidirectional, the LSTM outputs TWO hidden states concatenated together
        # So the input to the classifier must be hidden_dim * 2
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        
        # Extract features
        spatial_feats = self.backbone(x)
        
        # Reshape to sequence: (Batch, Frames, 2048)
        spatial_feats = spatial_feats.view(b, f, -1)
        
        # Normalize the features to stabilize the LSTM
        spatial_feats = self.layer_norm(spatial_feats)
        
        # Pass through the Bi-Directional LSTM
        lstm_out, (hn, cn) = self.lstm(spatial_feats)
        
        # hn shape for bidirectional is (num_layers * 2, batch, hidden_dim)
        # We grab the final forward hidden state and final backward hidden state
        final_forward = hn[-2, :, :]
        final_backward = hn[-1, :, :]
        
        # Concatenate them together
        final_state = torch.cat((final_forward, final_backward), dim=1)
        
        out = self.dropout(final_state)
        return self.classifier(out)