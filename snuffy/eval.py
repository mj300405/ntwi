import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from models.snuffy import Snuffy
from datasets.snuffy_dataset import SnuffyBagDataset
from utils.transforms import SnuffyTransform

def visualize_attention(model, dataset, save_dir, device, num_samples=5):
    """
    Visualize attention weights for sample slides
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        tiles, label = dataset[idx]
        tiles = tiles.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get attention weights
        with torch.no_grad():
            _, attn_weights = model(tiles)
        
        # Process attention weights
        attn_weights = attn_weights[0].cpu().numpy()  # Remove batch dimension
        attn_weights = attn_weights.mean(axis=0)  # Average over heads
        
        # Create attention heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_weights, cmap='jet')
        plt.colorbar()
        plt.title(f'Attention Weights (Label: {label})')
        plt.savefig(os.path.join(save_dir, f'attention_{idx}.png'))
        plt.close()
        
        # Create tile attention visualization
        tiles_np = tiles[0].cpu().numpy()
        n_tiles = min(tiles_np.shape[0], 16)  # Show up to 16 tiles
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()
        
        for i in range(n_tiles):
            tile = tiles_np[i].transpose(1, 2, 0)
            tile = (tile * 255).astype(np.uint8)
            
            # Overlay attention
            attention = attn_weights[i]
            attention = cv2.resize(attention, (224, 224))
            attention = (attention - attention.min()) / (attention.max() - attention.min())
            attention = (attention * 255).astype(np.uint8)
            attention = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
            
            # Blend
            overlay = cv2.addWeighted(tile, 0.7, attention, 0.3, 0)
            
            axes[i].imshow(overlay)
            axes[i].set_title(f'Tile {i} (Attn: {attention.mean():.3f})')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'tiles_{idx}.png'))
        plt.close()

def evaluate(args):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = Snuffy(
        backbone=args.backbone,
        num_classes=2,
        use_adapter=args.use_adapter,
        adapter_type=args.adapter_type,
        ffn_num=args.ffn_num,
        adapter_ffn_scalar=args.adapter_ffn_scalar,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_tiles=args.max_tiles
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create transform
    transform = SnuffyTransform(
        adapter_type=args.adapter_type,
        is_train=False
    )
    
    # Create dataset
    dataset = SnuffyBagDataset(
        data_dir=args.data_dir,
        transform=transform,
        max_tiles=args.max_tiles,
        include_augmented=False,
        val_split=args.val_split,
        is_validation=True,
        use_adapter=args.use_adapter,
        adapter_type=args.adapter_type
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for tiles, labels in pbar:
            tiles, labels = tiles.to(device), labels.to(device)
            
            logits, _ = model(tiles)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    # Calculate metrics
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Visualize attention if requested
    if args.visualize:
        print("\nGenerating attention visualizations...")
        visualize_attention(
            model=model,
            dataset=dataset,
            save_dir=args.vis_dir,
            device=device,
            num_samples=args.num_vis_samples
        )
        print(f"Visualizations saved to {args.vis_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Snuffy model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--max_tiles', type=int, default=100, help='Maximum number of tiles per slide')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'vit_small'], help='Backbone architecture')
    parser.add_argument('--use_adapter', action='store_true', help='Use adapter-based fine-tuning')
    parser.add_argument('--adapter_type', type=str, default='dino', choices=['dino', 'mae'], help='Type of adapter')
    parser.add_argument('--ffn_num', type=int, default=32, help='Number of features in adapter FFN')
    parser.add_argument('--adapter_ffn_scalar', type=float, default=1.0, help='Adapter FFN scalar')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create visualization directory if needed
    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)
    
    # Evaluate model
    evaluate(args)

if __name__ == '__main__':
    main()
