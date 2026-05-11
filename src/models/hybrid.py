import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from utilities.logger import get_logger

logger = get_logger("MODEL_HYBRID")

class SpatioTemporalAttention(nn.Module):
    """Early 3D Attention inside the ResNet backbone"""
    def __init__(self, in_channels):
        super(SpatioTemporalAttention, self).__init__()
        reduced_channels = in_channels // 8
        self.query = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, T, H, W = x.size()
        N = T * H * W  
        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  
        k = self.key(x).view(B, -1, N)                     
        v = self.value(x).view(B, -1, N).permute(0, 2, 1)  
        
        energy = torch.bmm(q, k)
        attention = self.softmax(energy)
        
        out = torch.bmm(attention, v)
        out = out.permute(0, 2, 1).view(B, C, T, H, W)
        return x + self.gamma * out

class TemporalAttention(nn.Module):
    """Late 1D Attention after the LSTM sequence"""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_fc = nn.Linear(hidden_dim * 2, hidden_dim)  
        self.score_fc = nn.Linear(hidden_dim, 1, bias=False)  
        self.dropout = nn.Dropout(0.1)

    def forward(self, lstm_outputs, mask=None):
        proj = torch.tanh(self.attn_fc(lstm_outputs))   
        proj = self.dropout(proj)
        scores = self.score_fc(proj).squeeze(-1)        
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=1)         

        if mask is not None:
            attn_weights = attn_weights * mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_outputs, dim=1)  
        return context, attn_weights

class HybridResNetLSTM(nn.Module):
    """Dual-Attention Hybrid: ResNet-50 + Early 3D Attention + BiLSTM + Late Temporal Attention"""
    def __init__(self, num_classes=4, hidden_dim=256, projection_dim=512, dropout_rate=0.5):
        super(HybridResNetLSTM, self).__init__()
        logger.info("Initializing DUAL-ATTENTION Hybrid: ResNet-50 + Early Spatiotemporal + BiLSTM + Late Temporal")
        
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  
        self.layer2 = backbone.layer2  
        self.layer3 = backbone.layer3  
        
        self.early_attention = SpatioTemporalAttention(in_channels=1024)
        
        self.layer4 = backbone.layer4  
        self.avgpool = backbone.avgpool 
        
        self.projection = nn.Sequential(
            nn.Linear(2048, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=projection_dim, 
            hidden_size=hidden_dim, 
            num_layers=2,        
            batch_first=True, 
            dropout=0.3,         
            bidirectional=True   
        )
        
        self.late_attention = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, seq_lengths=None):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  
        
        _, C3, H3, W3 = x.shape
        x = x.view(B, F, C3, H3, W3).permute(0, 2, 1, 3, 4) 
        x = self.early_attention(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * F, C3, H3, W3)
        
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        
        x = self.projection(x)
        x = x.view(B, F, -1) 
        
        lstm_out, _ = self.lstm(x) 
        
        mask = None
        if seq_lengths is not None:
            device = lstm_out.device
            idx = torch.arange(F, device=device).unsqueeze(0)  
            mask = idx < seq_lengths.unsqueeze(1)                    

        context, attn_weights = self.late_attention(lstm_out, mask=mask)
        
        out = self.dropout(context)
        logits = self.classifier(out)
        
        return logits