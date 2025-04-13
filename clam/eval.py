import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns

from clam.models.clam import CLAM
from clam.datasets.egfr_dataset import EGFRBagDataset

# Custom dataset class for test data with different directory names
class TestEGFRBagDataset(EGFRBagDataset):
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

def load_best_model(model_path, device):
    """
    Load the best model from the specified path
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    Returns:
        model: Loaded CLAM model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint['args']
    
    # Initialize model with same configuration
    model = CLAM(
        gate=True,
        size_arg=args['model_size'],
        dropout=args['dropout'],
        k_sample=args['k_sample'],
        n_classes=args['n_classes']
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
        model: CLAM model
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
            logits, Y_prob, Y_hat, A = model(data)
            
            # Collect predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(Y_hat.cpu().numpy())
            all_probs.extend(Y_prob[:, 1].cpu().numpy())  # Probability of positive class
    
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

def main(args):
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) for acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    # Validate data paths
    if not validate_data_paths(args.data_dir):
        print("\nExpected directory structure:")
        print("data_dir/")
        print("├── C-S-EGFR_positive/")
        print("└── C-S-EGFR_negative/")
        return
    
    # Load best model
    best_model_path = os.path.join(args.checkpoint_dir, "best_models", "latest_best_model.pt")
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        return
    
    print(f"Loading best model from {best_model_path}")
    model, model_args = load_best_model(best_model_path, device)
    
    # Initialize test dataset and dataloader with the custom dataset class
    test_dataset = TestEGFRBagDataset(
        data_dir=args.data_dir,
        max_tiles=model_args['max_tiles']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Save metrics and plots
    save_dir = os.path.join(args.checkpoint_dir, "evaluation_results")
    plot_metrics(metrics, save_dir)
    print(f"\nResults saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLAM model for EGFR mutation prediction")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test"),
                       help="Path to the test data directory")
    parser.add_argument("--checkpoint_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "clam/checkpoints"),
                       help="Path to the checkpoints directory")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    
    args = parser.parse_args()
    main(args) 