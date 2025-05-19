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
        
        # Add feature compression layer with LayerNorm
        self.compression = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        # x shape: [B*n, C, H, W]
        x = self.features(x)  # [B*n, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B*n, 2048]
        x = self.compression(x)  # [B*n, 512]
        return x

class SimplifiedAttention(nn.Module):
    def __init__(self, L, D, K, num_heads=8):
        super(SimplifiedAttention, self).__init__()
        self.L = L  # input dimension
        self.D = D  # hidden dimension
        self.K = K  # number of attention heads
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        
        # Simplified linear projections
        self.query = nn.Linear(L, D)
        self.key = nn.Linear(L, D)
        self.value = nn.Linear(L, D)
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(L)
        self.layer_norm2 = nn.LayerNorm(D)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [B, n, L]
        B, n, L = x.size()
        
        # First layer normalization
        x = self.layer_norm1(x)
        
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
        
        # Second layer normalization and residual connection
        M = self.layer_norm2(M)
        
        return M, A

class ResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ResidualClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # First residual block
        identity = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second residual block
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc3(x)
        return x

class CLAM(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2):
        """
        Enhanced CLAM model with improved feature extraction, simplified attention, and residual classifier
        Args:
            gate: Whether to use gated attention (ignored in new implementation)
            size_arg: Size of the model ('small' or 'big')
            dropout: Dropout rate
            k_sample: Number of attention heads
            n_classes: Number of classes (2 for EGFR pos/neg)
        """
        super(CLAM, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [512, 384, 256]}
        size = self.size_dict[size_arg]
        
        # Initialize backbone with feature compression
        self.backbone = FeatureExtractor()
        
        # Enable fine-tuning of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # Simplified attention mechanism
        self.attention = SimplifiedAttention(L=size[0], D=size[1], K=k_sample)
        
        # Residual classifier
        self.classifier = ResidualClassifier(size[1], size[2], n_classes)
        
        self.k_sample = k_sample
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.size_arg = size_arg
        
    def forward(self, x):
        # x comes as [B, n, C, H, W] from the dataset
        B, n, C, H, W = x.size()
        x = x.view(B * n, C, H, W)  # [B*n, C, H, W]
        
        # Extract and compress features
        features = self.backbone(x)  # [B*n, 512]
        features = features.view(B, n, -1)  # [B, n, 512]
        
        # Apply attention
        M, A = self.attention(features)  # [B, n, 256/384], [B, num_heads, n, n]
        
        # Global average pooling
        M = M.mean(dim=1)  # [B, 256/384]
        
        # Classification
        logits = self.classifier(M)  # [B, n_classes]
        
        # Calculate probabilities and predictions
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)
        
        return logits, Y_prob, Y_hat, A
    
    def calculate_loss(self, logits, labels, A=None):
        """
        Calculate the loss for the model
        Args:
            logits: Model output logits [B, n_classes]
            labels: Ground truth labels [B]
            A: Attention weights (optional)
        """
        # Instance-level loss
        instance_loss = self.instance_loss_fn(logits, labels)
        
        # Add attention regularization if attention weights are provided
        if A is not None:
            # Encourage attention weights to be sparse
            attention_reg = torch.mean(torch.abs(A))
            return instance_loss + 0.01 * attention_reg
        
        return instance_loss 