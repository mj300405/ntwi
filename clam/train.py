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

from clam.models.clam import CLAM
from clam.datasets.egfr_dataset import EGFRBagDataset

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
        optional_dirs = ['EGFR_positive_cnn', 'EGFR_negative_cnn']
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

def save_checkpoint(model, optimizer, epoch, loss, args, is_best=False, checkpoint_dir=None):
    """
    Save model checkpoint
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        args: Training arguments
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    
    # Create timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'epoch': epoch,
        'loss': loss,
        'args': vars(args)
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model if needed
    if is_best:
        best_model_path = os.path.join(run_dir, "best_model.pt")
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved to {best_model_path}")
    
    return run_dir

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the checkpoint into
        optimizer: Optimizer to load the checkpoint into (optional)
        device: Device to load the model to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

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
    
    # Initialize dataset and dataloader
    train_dataset = EGFRBagDataset(
        data_dir=args.data_dir,
        max_tiles=args.max_tiles
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'mps' else False
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            # Move data to device
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, Y_prob, Y_hat, A = model(data)
            
            # Calculate loss
            loss = model.calculate_loss(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.detach().item()
            pbar.set_postfix({"loss": loss.detach().item()})
            
            # Clear memory
            del data, labels, logits, Y_prob, Y_hat, A, loss
            if device.type == 'mps':
                # Force garbage collection for MPS
                gc.collect()
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        run_dir = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_loss,
            args=args,
            is_best=is_best
        )
        
        # Clear memory after each epoch
        gc.collect()
    
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"All checkpoints and best model saved to {run_dir}")

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
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "big"], help="Model size")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--k_sample", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    
    args = parser.parse_args()
    train_model(args) 