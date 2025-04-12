import torch
import sys
import os

# Add the parent directory to the path so we can import the dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import EGFRBagDataset
from torch.utils.data import DataLoader

def test_dataset():
    print("Testing EGFRBagDataset...")
    
    # Create dummy dataset
    dataset = EGFRBagDataset(max_tiles=50)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get a batch of data
    tiles, labels = next(iter(dataloader))
    
    print("\nDataset info:")
    print(f"Number of slides: {len(dataset)}")
    print(f"Batch shapes:")
    print(f"Tiles: {tiles.shape}")
    print(f"Labels: {labels.shape}")
    
    # Check data types
    print("\nData types:")
    print(f"Tiles: {tiles.dtype}")
    print(f"Labels: {labels.dtype}")
    
    # Check value ranges
    print("\nValue ranges:")
    print(f"Tiles min: {tiles.min().item():.4f}")
    print(f"Tiles max: {tiles.max().item():.4f}")
    print(f"Labels: {labels.unique().tolist()}")
    
    print("\nDataset test completed successfully!")

if __name__ == "__main__":
    test_dataset() 