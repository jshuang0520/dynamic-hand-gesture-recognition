import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HybridResNetLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=256):
        super(HybridResNetLSTM, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() 
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        spatial_feats = self.backbone(x)
        spatial_feats = spatial_feats.view(b, f, -1)
        
        lstm_out, (hn, cn) = self.lstm(spatial_feats)
        return self.classifier(hn[-1])