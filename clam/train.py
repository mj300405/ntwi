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

from models.clam import CLAM
from datasets.egfr_dataset import EGFRBagDataset

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
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
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
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) for acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    # Validate data paths
    if not validate_data_paths(args.data_dir, split=args.split):
        print("\nExpected directory structure:")
        if args.split == 'train':
            print("data_dir/")
            print("├── EGFR_positive/")
            print("├── EGFR_negative/")
            print("├── EGFR_positive_aug/ (optional)")
            print("├── EGFR_negative_aug/ (optional)")
            print("├── EGFR_positive_cnn/ (optional)")
            print("└── EGFR_negative_cnn/ (optional)")
        else:
            print("data_dir/")
            print("├── C-S-EGFR_positive/")
            print("└── C-S-EGFR_negative/")
        return
    
    # Initialize model
    model = CLAM(
        gate=True,
        size_arg=args.model_size,
        dropout=args.dropout,
        k_sample=args.k_sample,
        n_classes=args.n_classes
    )
    model = model.to(device)
    
    # Initialize datasets and dataloaders
    train_dataset = EGFRBagDataset(
        data_dir=args.data_dir,
        max_tiles=args.max_tiles,
        include_augmented=args.include_augmented,
        val_split=args.val_split,
        is_validation=False
    )
    
    val_dataset = EGFRBagDataset(
        data_dir=args.data_dir,
        max_tiles=args.max_tiles,
        include_augmented=args.include_augmented,
        val_split=args.val_split,
        is_validation=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )
    
    # Initialize metrics tracking
    all_metrics = []
    best_metrics = None
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_labels = []
        train_preds = []
        train_probs = []
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for batch_idx, (data, labels) in enumerate(train_pbar):
            # Move data to device
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, Y_prob, Y_hat, A = model(data)
            
            # Calculate loss
            loss = model.calculate_loss(logits, labels, A)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.detach().item()
            train_pbar.set_postfix({"loss": loss.detach().item()})
            
            # Collect predictions for metrics
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(Y_hat.detach().cpu().numpy())
            train_probs.extend(Y_prob[:, 1].detach().cpu().numpy())
            
            # Clear memory
            del data, labels, logits, Y_prob, Y_hat, A, loss
            if device.type == 'mps':
                gc.collect()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []
        val_probs = []
        
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            
            for batch_idx, (data, labels) in enumerate(val_pbar):
                # Move data to device
                data = data.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits, Y_prob, Y_hat, A = model(data)
                
                # Calculate loss
                loss = model.calculate_loss(logits, labels, A)
                
                # Update progress bar
                val_loss += loss.detach().item()
                val_pbar.set_postfix({"loss": loss.detach().item()})
                
                # Collect predictions for metrics
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(Y_hat.detach().cpu().numpy())
                val_probs.extend(Y_prob[:, 1].detach().cpu().numpy())
                
                # Clear memory
                del data, labels, logits, Y_prob, Y_hat, A, loss
                if device.type == 'mps':
                    gc.collect()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics for both training and validation
        train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
        val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("Training:")
        print(f"Loss: {avg_train_loss:.4f}")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Precision: {train_metrics['precision']:.4f}")
        print(f"Recall: {train_metrics['recall']:.4f}")
        print(f"F1 Score: {train_metrics['f1']:.4f}")
        print(f"AUC: {train_metrics['auc']:.4f}")
        
        print("\nValidation:")
        print(f"Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}")
        print(f"F1 Score: {val_metrics['f1']:.4f}")
        print(f"AUC: {val_metrics['auc']:.4f}")
        
        # Save metrics plots
        run_dir = early_stopping.save_checkpoint(avg_val_loss, model, optimizer, epoch + 1, args)
        metrics_dir = plot_metrics(val_metrics, epoch + 1, run_dir)
        
        # Update best metrics if needed
        if best_metrics is None or val_metrics['f1'] > best_metrics['f1']:
            best_metrics = val_metrics
            best_epoch = epoch + 1
        
        # Early stopping check using validation loss
        early_stopping(avg_val_loss, model, optimizer, epoch + 1, args)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Clear memory after each epoch
        gc.collect()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"Best validation F1 score: {best_metrics['f1']:.4f} at epoch {best_epoch}")
    print(f"Best validation accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best validation precision: {best_metrics['precision']:.4f}")
    print(f"Best validation recall: {best_metrics['recall']:.4f}")
    print(f"Best validation AUC: {best_metrics['auc']:.4f}")
    
    # Save final metrics summary
    final_metrics = {
        'best_epoch': best_epoch,
        'best_metrics': {
            'accuracy': float(best_metrics['accuracy']),
            'precision': float(best_metrics['precision']),
            'recall': float(best_metrics['recall']),
            'f1': float(best_metrics['f1']),
            'auc': float(best_metrics['auc']),
            'confusion_matrix': best_metrics['confusion_matrix'].tolist()
        },
        'all_metrics': [{
            'epoch': i+1,
            'train_metrics': {
                'accuracy': float(m['accuracy']),
                'precision': float(m['precision']),
                'recall': float(m['recall']),
                'f1': float(m['f1']),
                'auc': float(m['auc'])
            },
            'val_metrics': {
                'accuracy': float(m['accuracy']),
                'precision': float(m['precision']),
                'recall': float(m['recall']),
                'f1': float(m['f1']),
                'auc': float(m['auc'])
            }
        } for i, m in enumerate(all_metrics)]
    }
    
    with open(os.path.join(run_dir, 'metrics', 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLAM model for EGFR mutation prediction")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "train"),
                       help="Path to the data directory (default: project_root/train)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                       help="Data split to use (default: train)")
    parser.add_argument("--max_tiles", type=int, default=100, help="Maximum number of tiles per bag")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
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