import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 2048  # ResNet50's output dimension
        
    def forward(self, x):
        # x shape: [B*n, C, H, W]
        x = self.features(x)  # [B*n, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B*n, 2048]
        return x

class Attention(nn.Module):
    def __init__(self, L, D, K):
        super(Attention, self).__init__()
        self.L = L  # input dimension
        self.D = D  # hidden dimension
        self.K = K  # number of attention heads
        
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K)
        )
        
    def forward(self, x):
        # x shape: [B, n, L]
        B, n, L = x.size()
        
        # Compute attention scores
        A = self.attention(x)  # [B, n, K]
        A = F.softmax(A, dim=1)  # [B, n, K]
        
        # Apply attention to features
        M = torch.bmm(A.transpose(1, 2), x)  # [B, K, L]
        
        return M, A

class GatedAttention(nn.Module):
    def __init__(self, L, D, K):
        super(GatedAttention, self).__init__()
        self.L = L  # input dimension
        self.D = D  # hidden dimension
        self.K = K  # number of attention heads
        
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(D, K)
        
    def forward(self, x):
        # x shape: [B, n, L]
        B, n, L = x.size()
        
        # Compute attention scores with gating
        A_V = self.attention_V(x)  # [B, n, D]
        A_U = self.attention_U(x)  # [B, n, D]
        A = self.attention_weights(A_V * A_U)  # [B, n, K]
        A = F.softmax(A, dim=1)  # [B, n, K]
        
        # Apply attention to features
        M = torch.bmm(A.transpose(1, 2), x)  # [B, K, L]
        
        return M, A

class CLAM(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2):
        """
        CLAM model (Single Branch version) with feature extractor
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
        
        self.backbone = FeatureExtractor()
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = GatedAttention(L=size[1], D=size[2], K=k_sample)
        else:
            attention_net = Attention(L=size[1], D=size[2], K=k_sample)
        fc.extend([attention_net])
        self.attention_net = nn.Sequential(*fc)
        
        self.classifier = nn.Linear(size[1], n_classes)
        
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
        
        # Apply attention mechanism
        features = self.attention_net[:-1](features)  # Apply FC layers before attention
        M, A = self.attention_net[-1](features)  # M: [B, K, 512], A: [B, K, n]
        
        # Use the first attention head only
        M = M[:, 0, :]  # [B, 512]
        A = A[:, 0, :]  # [B, n]
        
        # Classification
        logits = self.classifier(M)  # [B, n_classes]
        Y_prob = torch.nn.functional.softmax(logits, dim=1)  # Use torch.nn.functional.softmax
        Y_hat = torch.argmax(Y_prob, dim=1)
        
        return logits, Y_prob, Y_hat, A
    
    def calculate_loss(self, logits, labels):
        return self.instance_loss_fn(logits, labels) 