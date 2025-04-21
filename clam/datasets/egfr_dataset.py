import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from pathlib import Path
import random

class EGFRBagDataset(Dataset):
    """
    Dataset class for EGFR classification using CLAM model.
    Each sample is a directory containing multiple PNG files (tiles).
    Optimized for Apple Metal (MPS) acceleration.
    """
    def __init__(self, data_dir=None, transform=None, max_tiles=100, include_augmented=True, val_split=0.2, is_validation=False):
        """
        Args:
            data_dir: Directory containing the data (if None, will use dummy data)
            transform: Optional transform to be applied on a sample (ignored)
            max_tiles: Maximum number of tiles to use per slide
            include_augmented: Whether to include augmented data folders
            val_split: Fraction of data to use for validation (default: 0.2)
            is_validation: Whether this is a validation dataset (default: False)
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_tiles = max_tiles
        self.val_split = val_split
        self.is_validation = is_validation
        
        # If no data directory is provided, create dummy data
        if data_dir is None:
            self.slide_paths = []
            self.labels = []
            # Create 10 dummy slides with random labels
            for i in range(10):
                # Create a dummy slide with random tiles
                n_tiles = np.random.randint(50, 150)
                tiles = []
                for j in range(n_tiles):
                    # Create a random RGB image
                    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    img = Image.fromarray(img)
                    tiles.append(img)
                self.slide_paths.append(tiles)
                self.labels.append(np.random.randint(0, 2))  # Binary labels
        else:
            # Load real data from directory
            self.slide_paths = []
            self.labels = []
            
            # Load positive samples
            positive_dirs = ["EGFR_positive"]
            if include_augmented:
                positive_dirs.append("EGFR_positive_aug")
                
            positive_samples = []
            for dir_name in positive_dirs:
                positive_dir = self.data_dir / dir_name
                if positive_dir.exists():
                    print(f"Loading positive samples from {dir_name}")
                    for slide_dir in positive_dir.iterdir():
                        if slide_dir.is_dir():
                            tile_paths = list(slide_dir.glob("*.png"))
                            if tile_paths:  # Only add if we found valid tiles
                                positive_samples.append((tile_paths, 1))  # 1 for positive
            
            # Load negative samples
            negative_dirs = ["EGFR_negative"]
            if include_augmented:
                negative_dirs.append("EGFR_negative_aug")
                
            negative_samples = []
            for dir_name in negative_dirs:
                negative_dir = self.data_dir / dir_name
                if negative_dir.exists():
                    print(f"Loading negative samples from {dir_name}")
                    for slide_dir in negative_dir.iterdir():
                        if slide_dir.is_dir():
                            tile_paths = list(slide_dir.glob("*.png"))
                            if tile_paths:  # Only add if we found valid tiles
                                negative_samples.append((tile_paths, 0))  # 0 for negative
            
            if not positive_samples and not negative_samples:
                raise ValueError(f"No valid slides found in {data_dir}")
            
            # Balance the dataset by using the same number of samples from each class
            n_samples = min(len(positive_samples), len(negative_samples))
            if n_samples == 0:
                raise ValueError("No samples found in one or both classes")
            
            # Randomly sample equal number of positive and negative samples
            positive_samples = random.sample(positive_samples, n_samples)
            negative_samples = random.sample(negative_samples, n_samples)
            
            # Split into train/val sets
            all_samples = positive_samples + negative_samples
            random.shuffle(all_samples)
            
            split_idx = int(len(all_samples) * (1 - val_split))
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]
            
            # Select appropriate samples based on is_validation flag
            selected_samples = val_samples if is_validation else train_samples
            
            # Unpack the selected samples
            for tile_paths, label in selected_samples:
                self.slide_paths.append(tile_paths)
                self.labels.append(label)
            
            print(f"Found {len(self.slide_paths)} slides ({sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative)")
            print(f"{'Validation' if is_validation else 'Training'} set size: {len(self.slide_paths)}")
    
    def __len__(self):
        return len(self.slide_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            tiles: A tensor of shape [n, C, H, W] where n is the number of tiles
            label: The label of the slide
        """
        tile_paths = self.slide_paths[idx]
        label = self.labels[idx]
        
        # Limit the number of tiles if necessary
        if len(tile_paths) > self.max_tiles:
            # Randomly sample max_tiles tiles
            tile_paths = random.sample(tile_paths, self.max_tiles)
        
        # Load tiles on-demand
        tiles = []
        for tile_path in tile_paths:
            try:
                # Load image and convert to tensor directly
                img = Image.open(tile_path).convert('RGB')
                img_array = np.array(img)
                
                # Convert to tensor with minimal memory usage
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                
                # Free memory
                del img, img_array
                
                tiles.append(img_tensor)
            except Exception as e:
                print(f"Error loading {tile_path}: {e}")
                # If we can't load an image, create a blank one
                blank_img = torch.zeros((3, 224, 224))
                tiles.append(blank_img)
        
        # Stack tiles into a single tensor
        tiles_tensor = torch.stack(tiles)
        
        return tiles_tensor, torch.tensor(label, dtype=torch.long) 