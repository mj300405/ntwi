import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Adapter(nn.Module):
    """
    Adapter module for fine-tuning pre-trained models
    """
    def __init__(self, in_features, ffn_num=32, adapter_ffn_scalar=1):
        super().__init__()
        self.down = nn.Linear(in_features, ffn_num)
        self.up = nn.Linear(ffn_num, in_features)
        self.scalar = adapter_ffn_scalar
        
    def forward(self, x):
        return x + self.scalar * self.up(F.gelu(self.down(x)))

class Snuffy(nn.Module):
    """
    Snuffy model combining self-supervised learning with multiple instance learning
    """
    def __init__(self, 
                 backbone='resnet18',
                 num_classes=2,
                 use_adapter=False,
                 adapter_type='dino',
                 ffn_num=32,
                 adapter_ffn_scalar=1,
                 num_heads=4,
                 dropout=0.1,
                 max_tiles=100):
        super().__init__()
        
        # Initialize backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif backbone == 'vit_small':
            # Placeholder for ViT-Small
            self.backbone = nn.Sequential(
                nn.Linear(224*224*3, 384),
                nn.LayerNorm(384),
                nn.GELU()
            )
            self.feature_dim = 384
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Adapter configuration
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = Adapter(
                in_features=self.feature_dim,
                ffn_num=ffn_num,
                adapter_ffn_scalar=adapter_ffn_scalar
            )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.max_tiles = max_tiles
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, max_tiles, channels, height, width]
        Returns:
            logits: Classification logits
            attention_weights: Attention weights for visualization
        """
        batch_size = x.size(0)
        
        # Process each tile through backbone
        features = []
        for i in range(x.size(1)):  # Iterate over tiles
            tile = x[:, i]  # [batch_size, channels, height, width]
            if isinstance(self.backbone, models.ResNet):
                # ResNet processing
                feat = self.backbone.conv1(tile)
                feat = self.backbone.bn1(feat)
                feat = self.backbone.relu(feat)
                feat = self.backbone.maxpool(feat)
                feat = self.backbone.layer1(feat)
                feat = self.backbone.layer2(feat)
                feat = self.backbone.layer3(feat)
                feat = self.backbone.layer4(feat)
                feat = self.backbone.avgpool(feat)
                feat = torch.flatten(feat, 1)
            else:
                # ViT processing
                feat = tile.view(batch_size, -1)
                feat = self.backbone(feat)
            
            if self.use_adapter:
                feat = self.adapter(feat)
            
            features.append(feat)
        
        # Stack features
        features = torch.stack(features, dim=1)  # [batch_size, max_tiles, feature_dim]
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            features, features, features
        )
        
        # Global pooling (mean of attended features)
        pooled = attn_output.mean(dim=1)  # [batch_size, feature_dim]
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, attn_weights
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization
        """
        _, attn_weights = self.forward(x)
        return attn_weights
