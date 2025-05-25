import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image

class DINOTransform:
    """
    DINO-specific transforms for self-supervised learning
    """
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), is_train=True):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.is_train = is_train
        
        # Global transforms
        self.global_transform = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.ToTensor(),  # Convert PIL Image to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Local transforms
        self.local_transform = T.Compose([
            T.RandomResizedCrop(96, scale=local_crops_scale),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.ToTensor(),  # Convert PIL Image to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
            
        if self.is_train:
            # Generate two global views
            global_view1 = self.global_transform(image)
            global_view2 = self.global_transform(image)
            
            # Generate local views
            local_views = [self.local_transform(image) for _ in range(8)]
            
            return global_view1, global_view2, local_views
        else:
            # For inference, just return a single transformed view
            return self.global_transform(image)

class MAETransform:
    """
    MAE-specific transforms for self-supervised learning
    """
    def __init__(self, mask_ratio=0.75, is_train=True):
        self.mask_ratio = mask_ratio
        self.is_train = is_train
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # Convert PIL Image to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
            
        # Apply basic transforms
        x = self.transform(image)
        
        if self.is_train:
            # Reshape to patches
            B, C, H, W = x.shape
            x = x.reshape(B, C, H//16, 16, W//16, 16)
            x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H//16)*(W//16), C*16*16)
            
            # Apply masking
            x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
            
            return x_masked, mask, ids_restore
        else:
            # For inference, just return the transformed image
            return x

class SnuffyTransform:
    """
    Combined transforms for Snuffy model
    """
    def __init__(self, adapter_type='dino', is_train=True):
        self.adapter_type = adapter_type
        self.is_train = is_train
        
        if adapter_type == 'dino':
            self.transform = DINOTransform(is_train=is_train)
        elif adapter_type == 'mae':
            self.transform = MAETransform(is_train=is_train)
        else:
            # Default transforms for ResNet
            self.transform = T.Compose([
                T.Resize(256) if not is_train else T.RandomResizedCrop(224),
                T.RandomHorizontalFlip() if is_train else T.Lambda(lambda x: x),
                T.ToTensor(),  # Convert PIL Image to tensor
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        return self.transform(image)
