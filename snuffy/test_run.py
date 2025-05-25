import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import Namespace
from train import train
from eval import evaluate

def create_dummy_image(path, size=(224, 224)):
    """Create a dummy image for testing."""
    # Create a random image
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)

def main():
    # Create a small test dataset directory
    data_dir = Path("test_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create test directories and dummy images
    for class_name in ["EGFR_positive", "EGFR_negative"]:
        class_dir = data_dir / class_name
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 2 slides with 5 tiles each
        for slide_idx in range(2):
            slide_dir = class_dir / f"slide_{slide_idx}"
            os.makedirs(slide_dir, exist_ok=True)
            
            # Create 5 tiles per slide
            for tile_idx in range(5):
                tile_path = slide_dir / f"tile_{tile_idx}.png"
                create_dummy_image(tile_path)
    
    # Training arguments
    train_args = Namespace(
        data_dir=str(data_dir),
        max_tiles=50,  # Smaller for testing
        val_split=0.2,
        backbone='resnet18',  # Start with ResNet for testing
        use_adapter=False,  # Start without adapter
        adapter_type='dino',
        ffn_num=32,
        adapter_ffn_scalar=1.0,
        num_heads=4,
        dropout=0.1,
        batch_size=4,  # Small batch size for testing
        num_workers=2,
        epochs=2,  # Just a few epochs for testing
        lr=0.001,
        weight_decay=0.0001,
        save_freq=1,
        grad_clip=1.0,
        patience=3,
        min_delta=0.0,
        checkpoint_dir='test_checkpoints'
    )
    
    # Create checkpoint directory
    os.makedirs(train_args.checkpoint_dir, exist_ok=True)
    
    print("Starting training...")
    train(train_args)
    
    # Evaluation arguments
    eval_args = Namespace(
        data_dir=str(data_dir),
        max_tiles=50,
        val_split=0.2,
        backbone='resnet18',
        use_adapter=False,
        adapter_type='dino',
        ffn_num=32,
        adapter_ffn_scalar=1.0,
        num_heads=4,
        dropout=0.1,
        checkpoint_path=str(Path(train_args.checkpoint_dir) / 'snuffy_best.pth'),
        batch_size=4,
        num_workers=2,
        visualize=True,
        vis_dir='test_visualizations',
        num_vis_samples=2
    )
    
    print("\nStarting evaluation...")
    evaluate(eval_args)

if __name__ == '__main__':
    main() 