import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import torch.nn.functional as F
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from .models.clam import CLAM
from .datasets.egfr_dataset import EGFRBagDataset

def validate_data_paths(data_dir, split='train'):
    """
    Validate that the data directory contains the expected structure
    Args:
        data_dir: Path to the data directory
        split: Either 'train' or 'test'
    Returns:
        bool: True if paths are valid, False otherwise
    """
    data_dir = Path(data_dir)
    
    # Define expected directories based on split
    if split == 'train':
        required_dirs = ['EGFR_positive', 'EGFR_negative']
        optional_dirs = ['EGFR_positive_aug', 'EGFR_negative_aug', 'EGFR_positive_cnn', 'EGFR_negative_cnn']
    else:  # test
        required_dirs = ['C-S-EGFR_positive', 'C-S-EGFR_negative']
        optional_dirs = []
    
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
    
    # Check for optional directories
    found_optional = []
    for dir_name in optional_dirs:
        if (data_dir / dir_name).exists():
            found_optional.append(dir_name)
    
    if found_optional:
        print(f"Found optional directories in {data_dir}:")
        for dir_name in found_optional:
            print(f"  - {dir_name}")
    
    return True

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.current_metrics = None  # Store current metrics

    def __call__(self, val_loss, model, optimizer, epoch, args, checkpoint_dir=None, metrics=None):
        if metrics is not None:
            self.current_metrics = metrics
            
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, args, True, checkpoint_dir)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch, args, True, checkpoint_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, args, is_best=False, checkpoint_dir=None):
        """Save model checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        
        # Create timestamped directory for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Create best_models directory if it doesn't exist
        best_models_dir = os.path.join(checkpoint_dir, "best_models")
        os.makedirs(best_models_dir, exist_ok=True)
        
        # Save training configuration
        config = {
            'epoch': epoch,
            'loss': val_loss,
            'args': vars(args)
        }
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'args': vars(args)
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if needed
        if is_best and self.current_metrics is not None:
            # Save in run directory
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved to {best_model_path}")
            
            # Create filename with metrics
            metrics_str = f"f1_{self.current_metrics['f1']:.3f}_acc_{self.current_metrics['accuracy']:.3f}_auc_{self.current_metrics['auc']:.3f}"
            best_model_name = f"best_model_{timestamp}_{metrics_str}.pt"
            
            # Save in best_models directory with timestamp and metrics
            best_model_path = os.path.join(best_models_dir, best_model_name)
            torch.save(checkpoint, best_model_path)
            print(f"New best model also saved to {best_model_path}")
            
            # Create/update symlink to latest best model
            latest_best_path = os.path.join(best_models_dir, "latest_best_model.pt")
            if os.path.exists(latest_best_path):
                os.remove(latest_best_path)
            os.symlink(best_model_name, latest_best_path)
            print(f"Updated symlink to latest best model at {latest_best_path}")
        
        return run_dir

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics"""
    # Define labels to ensure confusion matrix has correct shape
    labels = [0, 1]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_score
    }

def plot_metrics(metrics, epoch, run_dir):
    """Plot and save metrics"""
    # Create metrics directory
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(metrics_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'], metrics['tpr'], label=f'AUC = {metrics["auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, f'roc_curve_epoch_{epoch}.png'))
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
    
    with open(os.path.join(metrics_dir, f'metrics_epoch_{epoch}.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    return metrics_dir

def train_model(args):
    """Train the CLAM model with given parameters"""
    # Set up device with automatic MPS detection
    if torch.backends.mps.is_available():
        try:
            # Use a more conservative memory fraction
            torch.mps.set_per_process_memory_fraction(1.0)
            device = torch.device("mps")
            print("Using MPS device with 100% memory limit")
        except RuntimeError as e:
            print(f"Warning: Could not set MPS memory fraction: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"train_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training parameters
    with open(output_dir / "params.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize model
    model = CLAM(
        size_arg=args.model_size,
        dropout=args.dropout,
        k_sample=args.k_sample,
        n_classes=2
    ).to(device)
    
    # Enable gradient checkpointing if using MPS
    if device.type == 'mps':
        model.use_checkpointing = True
    
    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Initialize dataset with memory-efficient settings
    train_dataset = EGFRBagDataset(
        data_dir=args.data_dir,
        max_tiles=args.max_tiles,
        val_split=args.val_split,
        include_augmented=args.include_augmented,
        is_validation=False
    )
    
    val_dataset = EGFRBagDataset(
        data_dir=args.data_dir,
        max_tiles=args.max_tiles,
        val_split=args.val_split,
        include_augmented=args.include_augmented,
        is_validation=True
    )
    
    # Create data loaders with reduced prefetch factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=None  # Set to None since we're not using multiprocessing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=None  # Set to None since we're not using multiprocessing
    )
    
    # Training loop with memory cleanup
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = []
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = defaultdict(list)
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]', leave=False)
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, Y_prob, Y_hat, A = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Use detach() to avoid the warning
            train_loss += loss.detach().item()
            metrics = calculate_metrics(
                target.cpu().detach().numpy(),
                Y_hat.cpu().detach().numpy(),
                Y_prob[:, 1].cpu().detach().numpy()
            )
            for k, v in metrics.items():
                train_metrics[k].append(v)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.detach().item():.4f}',
                'acc': f'{metrics["accuracy"]:.4f}'
            })
            
            # Clear cache more frequently
            if batch_idx % 5 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Free up memory
            del loss
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(list)
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]', leave=False)
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                logits, Y_prob, Y_hat, A = model(data)
                val_loss += F.cross_entropy(logits, target).item()
                metrics = calculate_metrics(
                    target.cpu().numpy(),
                    Y_hat.cpu().numpy(),
                    Y_prob[:, 1].cpu().numpy()
                )
                for k, v in metrics.items():
                    val_metrics[k].append(v)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{F.cross_entropy(logits, target).item():.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}'
                })
                
                # Free up memory
                if device.type == 'mps':
                    torch.mps.empty_cache()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': {k: np.mean(v) for k, v in train_metrics.items()},
            'val_metrics': {k: np.mean(v) for k, v in val_metrics.items()}
        }
        metrics_history.append(epoch_metrics)
        
        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_history, f, indent=4)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Train Metrics:", {k: f"{v:.4f}" for k, v in epoch_metrics['train_metrics'].items()})
        print("Val Metrics:", {k: f"{v:.4f}" for k, v in epoch_metrics['val_metrics'].items()})
        
        # Clear cache after each epoch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLAM model for EGFR mutation prediction")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "train"),
                       help="Path to the data directory (default: project_root/train)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                       help="Data split to use (default: train)")
    parser.add_argument("--max_tiles", type=int, default=100, help="Maximum number of tiles per bag")
    parser.add_argument("--include_augmented", action="store_true", default=True,
                       help="Include augmented data folders in training")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Fraction of data to use for validation (default: 0.2)")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="big", choices=["small", "big"], help="Model size")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--k_sample", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    
    # Early stopping arguments
    parser.add_argument("--patience", type=int, default=7, help="Number of epochs to wait before early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change in loss to qualify as an improvement")
    
    args = parser.parse_args()
    train_model(args) 