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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from models.snuffy import Snuffy
from datasets.snuffy_dataset import SnuffyBagDataset
from utils.transforms import SnuffyTransform

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
        optional_dirs = ['EGFR_positive_aug', 'EGFR_negative_aug']
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
        self.current_metrics = None

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
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
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
    """Train the Snuffy model with given parameters"""
    # Set up device with automatic MPS detection
    if args.use_cpu:
        device = torch.device("cpu")
        print("Using CPU as requested")
    elif torch.backends.mps.is_available():
        try:
            # Set memory limit to 90% of available memory
            torch.mps.set_per_process_memory_fraction(0.9)
            device = torch.device("mps")
            print("Using MPS device with 90% memory limit")
        except RuntimeError as e:
            print(f"Warning: Could not set MPS memory fraction: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validate data paths
    if not validate_data_paths(args.data_dir, split='train'):
        print("\nExpected directory structure:")
        print("data_dir/")
        print("├── EGFR_positive/")
        print("├── EGFR_negative/")
        print("├── EGFR_positive_aug/ (optional)")
        print("└── EGFR_negative_aug/ (optional)")
        return
    
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
    
    # Create transform
    transform = SnuffyTransform(
        adapter_type=args.adapter_type,
        is_train=True
    )
    
    # Create datasets
    train_dataset = SnuffyBagDataset(
        data_dir=args.data_dir,
        transform=transform,
        max_tiles=args.max_tiles,
        include_augmented=True,
        val_split=args.val_split,
        is_validation=False,
        use_adapter=args.use_adapter,
        adapter_type=args.adapter_type
    )
    
    val_dataset = SnuffyBagDataset(
        data_dir=args.data_dir,
        transform=transform,
        max_tiles=args.max_tiles,
        include_augmented=False,
        val_split=args.val_split,
        is_validation=True,
        use_adapter=args.use_adapter,
        adapter_type=args.adapter_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and not torch.backends.mps.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and not torch.backends.mps.is_available()
    )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_labels = []
        all_train_preds = []
        all_train_probs = []
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for i, (tiles, labels) in enumerate(pbar):
            tiles, labels = tiles.to(device), labels.to(device)
            
            # Forward pass
            logits, _ = model(tiles)
            loss = criterion(logits, labels)
            
            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item() * args.gradient_accumulation_steps
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Handle remaining gradients
        if len(train_loader) % args.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate training metrics
        train_metrics = calculate_metrics(
            np.array(all_train_labels),
            np.array(all_train_preds),
            np.array(all_train_probs)
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for tiles, labels in pbar:
                tiles, labels = tiles.to(device), labels.to(device)
                
                logits, _ = model(tiles)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs[:, 1].detach().cpu().numpy())
                
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(
            np.array(all_val_labels),
            np.array(all_val_preds),
            np.array(all_val_probs)
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*val_correct/val_total:.2f}%")
        
        # Plot and save metrics
        run_dir = plot_metrics(train_metrics, epoch, os.path.join(args.checkpoint_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        plot_metrics(val_metrics, epoch, run_dir)
        
        # Early stopping
        early_stopping(val_loss, model, optimizer, epoch, args, args.checkpoint_dir, val_metrics)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

def main():
    parser = argparse.ArgumentParser(description='Train Snuffy model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='train', help='Path to data directory')
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
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='Early stopping patience')
    
    # Memory management
    parser.add_argument('--use_cpu', action='store_true', help='Force using CPU instead of MPS')
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory for faster data transfer')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train model
    train_model(args)

if __name__ == '__main__':
    main()
