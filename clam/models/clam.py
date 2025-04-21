import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 2048  # ResNet50's output dimension
        
    def forward(self, x):
        # x shape: [B*n, C, H, W]
        x = self.features(x)  # [B*n, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B*n, 2048]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, L, D, K, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.L = L  # input dimension
        self.D = D  # hidden dimension
        self.K = K  # number of attention heads
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        
        self.query = nn.Linear(L, D)
        self.key = nn.Linear(L, D)
        self.value = nn.Linear(L, D)
        self.proj = nn.Linear(D, D)
        
        self.layer_norm = nn.LayerNorm(L)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [B, n, L]
        B, n, L = x.size()
        
        # Linear projections
        Q = self.query(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        A = F.softmax(scores, dim=-1)
        A = self.dropout(A)
        
        # Apply attention to values
        M = torch.matmul(A, V)  # [B, num_heads, n, head_dim]
        M = M.transpose(1, 2).contiguous().view(B, n, self.D)
        M = self.proj(M)
        
        # Residual connection and layer normalization
        M = self.layer_norm(x + M)
        
        return M, A

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, L, D, K, num_heads=8):
        super(GatedMultiHeadAttention, self).__init__()
        self.L = L  # input dimension
        self.D = D  # hidden dimension
        self.K = K  # number of attention heads
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        
        self.query = nn.Linear(L, D)
        self.key = nn.Linear(L, D)
        self.value = nn.Linear(L, D)
        self.gate = nn.Linear(L, D)
        self.proj = nn.Linear(D, D)
        
        self.layer_norm = nn.LayerNorm(L)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [B, n, L]
        B, n, L = x.size()
        
        # Linear projections
        Q = self.query(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        G = torch.sigmoid(self.gate(x)).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with gating
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        A = F.softmax(scores, dim=-1)
        A = self.dropout(A)
        
        # Apply attention to values with gating
        M = torch.matmul(A, V * G)  # [B, num_heads, n, head_dim]
        M = M.transpose(1, 2).contiguous().view(B, n, self.D)
        M = self.proj(M)
        
        # Residual connection and layer normalization
        M = self.layer_norm(x + M)
        
        return M, A

class CLAM(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2):
        """
        Enhanced CLAM model with multi-head attention and residual connections
        Args:
            gate: Whether to use gated attention
            size_arg: Size of the model ('small' or 'big')
            dropout: Dropout rate
            k_sample: Number of attention heads
            n_classes: Number of classes (2 for EGFR pos/neg)
        """
        super(CLAM, self).__init__()
        self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        size = self.size_dict[size_arg]
        
        # Initialize backbone
        self.backbone = FeatureExtractor()
        
        # Enable fine-tuning of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # Feature transformation layers with residual connection
        self.feature_transform = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.LayerNorm(size[1]),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        
        # Attention mechanism
        if gate:
            self.attention = GatedMultiHeadAttention(L=size[1], D=size[1], K=k_sample)
        else:
            self.attention = MultiHeadAttention(L=size[1], D=size[1], K=k_sample)
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(size[1], size[1]),
            nn.LayerNorm(size[1]),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(size[1], size[1] // 2),
            nn.LayerNorm(size[1] // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(size[1] // 2, n_classes)
        )
        
        self.k_sample = k_sample
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.size_arg = size_arg
        
    def forward(self, x):
        # x comes as [B, n, C, H, W] from the dataset
        B, n, C, H, W = x.size()
        x = x.view(B * n, C, H, W)  # [B*n, C, H, W]
        
        # Extract features
        features = self.backbone(x)  # [B*n, 2048]
        features = features.view(B, n, -1)  # [B, n, 2048]
        
        # Transform features
        features = self.feature_transform(features)  # [B, n, 512]
        
        # Apply attention mechanism
        M, A = self.attention(features)  # M: [B, n, 512], A: [B, num_heads, n, n]
        
        # Global average pooling across instances
        M = M.mean(dim=1)  # [B, 512]
        
        # Classification
        logits = self.classifier(M)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)
        
        return logits, Y_prob, Y_hat, A
    
    def calculate_loss(self, logits, labels):
        # Calculate instance-level loss
        instance_loss = self.instance_loss_fn(logits, labels)
        
        # Add attention regularization to encourage more uniform attention
        # This helps prevent the model from focusing too much on a few instances
        attention_entropy = -torch.mean(torch.sum(A.mean(dim=1) * torch.log(A.mean(dim=1) + 1e-10), dim=1))
        attention_loss = -attention_entropy  # Maximize entropy = more uniform attention
        
        # Combine losses with a weight for the attention regularization
        total_loss = instance_loss + 0.1 * attention_loss
        
        return total_loss 