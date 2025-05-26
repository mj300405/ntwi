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
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns

from models.snuffy import Snuffy
from datasets.snuffy_dataset import SnuffyBagDataset, TestSnuffyBagDataset
from utils.transforms import SnuffyTransform

def visualize_attention(model, dataset, save_dir, device, num_samples=5):
    """
    Visualize attention weights for sample slides
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Get random samples, but ensure we don't try to sample more than available
    num_samples = min(num_samples, len(dataset))
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
        
        # Create attention bar plot
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(attn_weights)), attn_weights)
        plt.title(f'Attention Weights (Label: {label})')
        plt.xlabel('Tile Index')
        plt.ylabel('Attention Weight')
        plt.savefig(os.path.join(save_dir, f'attention_{idx}.png'))
        plt.close()
        
        # Create tile attention visualization
        tiles_np = tiles[0].cpu().numpy()
        n_tiles = min(tiles_np.shape[0], 16)  # Show up to 16 tiles
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()
        
        for i in range(n_tiles):
            tile = tiles_np[i].transpose(1, 2, 0)  # CHW to HWC
            axes[i].imshow(tile)
            axes[i].set_title(f'Tile {i} (Attn: {attn_weights[i]:.3f})')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_tiles, 16):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'tiles_{idx}.png'))
        plt.close()

def validate_data_paths(data_dir):
    """
    Validate that the data directory contains the expected structure
    Args:
        data_dir: Path to the data directory
    Returns:
        bool: True if paths are valid, False otherwise
    """
    data_dir = Path(data_dir)
    
    # Define expected directories
    required_dirs = ['C-S-EGFR_positive', 'C-S-EGFR_negative']
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return False
    
    # Check for required subdirectories
    missing_dirs = []
    for dir_name in required_dirs:
        if not (data_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Error: Required directories not found in {data_dir}:")
        for dir_name in missing_dirs:
            print(f"  - {dir_name}")
        return False
    
    return True

def load_best_model(model_path, device):
    """
    Load the best model from the specified path
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    Returns:
        model: Loaded Snuffy model
        args: Model configuration
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint['args']
    
    # Initialize model with same configuration
    model = Snuffy(
        backbone=args['backbone'],
        num_classes=2,
        use_adapter=args['use_adapter'],
        adapter_type=args['adapter_type'],
        ffn_num=args['ffn_num'],
        adapter_ffn_scalar=args['adapter_ffn_scalar'],
        num_heads=args['num_heads'],
        dropout=args['dropout'],
        max_tiles=args['max_tiles']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, args

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset
    Args:
        model: Snuffy model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(data)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            # Collect predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_metrics(metrics, save_dir):
    """
    Plot and save evaluation metrics
    Args:
        metrics: Dictionary containing evaluation metrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'], metrics['tpr'], label=f'AUC = {metrics["auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # Save metrics to JSON
    metrics_to_save = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Snuffy model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='test', help='Path to test data directory')
    parser.add_argument('--max_tiles', type=int, default=100, help='Maximum number of tiles per slide')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'vit_small'], help='Backbone architecture')
    parser.add_argument('--use_adapter', action='store_true', help='Use adapter-based fine-tuning')
    parser.add_argument('--adapter_type', type=str, default='dino', choices=['dino', 'mae'], help='Type of adapter')
    parser.add_argument('--ffn_num', type=int, default=32, help='Number of features in adapter FFN')
    parser.add_argument('--adapter_ffn_scalar', type=float, default=1.0, help='Adapter FFN scalar')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--checkpoint_path', type=str, default='snuffy/checkpoints/best_models/latest_best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--visualization_dir', type=str, default='output/visualizations', help='Directory to save visualizations')
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory for faster data transfer to GPU')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate data paths
    if not validate_data_paths(args.data_dir):
        print("\nExpected directory structure:")
        print("data_dir/")
        print("├── C-S-EGFR_positive/")
        print("└── C-S-EGFR_negative/")
        return
    
    # Load best model
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Best model not found at {args.checkpoint_path}")
        return
    
    print(f"Loading best model from {args.checkpoint_path}")
    model, model_args = load_best_model(args.checkpoint_path, device)
    
    # Create transform
    transform = SnuffyTransform(
        adapter_type=model_args['adapter_type'],
        is_train=False
    )
    
    # Initialize test dataset and dataloader
    test_dataset = TestSnuffyBagDataset(
        data_dir=args.data_dir,
        transform=transform,
        max_tiles=model_args['max_tiles']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and not torch.backends.mps.is_available()
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Plot and save metrics
    print("\nSaving evaluation metrics and plots...")
    plot_metrics(metrics, args.visualization_dir)
    print(f"Results saved to {args.visualization_dir}")

if __name__ == '__main__':
    main()

# Custom dataset class for test data with different directory names
class TestSnuffyBagDataset(SnuffyBagDataset):
    def __init__(self, data_dir=None, transform=None, max_tiles=100):
        """
        Custom dataset class for test data with different directory names
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_tiles = max_tiles
        
        # Load real data from directory
        self.slide_paths = []
        self.labels = []
        
        # Load positive samples
        positive_dir = self.data_dir / "C-S-EGFR_positive"
        if positive_dir.exists():
            for slide_dir in positive_dir.iterdir():
                if slide_dir.is_dir():
                    tile_paths = list(slide_dir.glob("*.png"))
                    if tile_paths:  # Only add if we found valid tiles
                        self.slide_paths.append(tile_paths)
                        self.labels.append(1)  # 1 for positive
        
        # Load negative samples
        negative_dir = self.data_dir / "C-S-EGFR_negative"
        if negative_dir.exists():
            for slide_dir in negative_dir.iterdir():
                if slide_dir.is_dir():
                    tile_paths = list(slide_dir.glob("*.png"))
                    if tile_paths:  # Only add if we found valid tiles
                        self.slide_paths.append(tile_paths)
                        self.labels.append(0)  # 0 for negative
        
        if not self.slide_paths:
            raise ValueError(f"No valid slides found in {data_dir}")
        
        print(f"Found {len(self.slide_paths)} slides ({sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative)")
