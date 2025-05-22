import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.snuffy import Snuffy
from datasets.snuffy_dataset import SnuffyBagDataset
from utils.transforms import SnuffyTransform

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(args):
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
    
    # Create transforms
    train_transform = SnuffyTransform(
        adapter_type=args.adapter_type,
        is_train=True
    )
    
    val_transform = SnuffyTransform(
        adapter_type=args.adapter_type,
        is_train=False
    )
    
    # Create datasets
    train_dataset = SnuffyBagDataset(
        data_dir=args.data_dir,
        transform=train_transform,
        max_tiles=args.max_tiles,
        include_augmented=True,
        val_split=args.val_split,
        is_validation=False,
        use_adapter=args.use_adapter,
        adapter_type=args.adapter_type
    )
    
    val_dataset = SnuffyBagDataset(
        data_dir=args.data_dir,
        transform=val_transform,
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for tiles, labels in pbar:
            tiles, labels = tiles.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(tiles)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for tiles, labels in pbar:
                tiles, labels = tiles.to(device), labels.to(device)
                
                logits, _ = model(tiles)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(args.checkpoint_dir) / f'snuffy_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = Path(args.checkpoint_dir) / f'snuffy_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

def main():
    parser = argparse.ArgumentParser(description='Train Snuffy model')
    
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
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Early stopping minimum delta')
    
    # Directory parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train model
    train(args)

if __name__ == '__main__':
    main()
