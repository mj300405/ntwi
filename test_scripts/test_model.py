import torch
import sys
import os

# Add the parent directory to the path so we can import the model and dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from dataset import EGFRBagDataset
from torch.utils.data import DataLoader

def test_model():
    print("Testing CLAM model with ResNet50 backbone...")
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(device)
    
    # Create dummy dataset
    dataset = EGFRBagDataset(max_tiles=50)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get a batch of data
    tiles, labels = next(iter(dataloader))
    tiles = tiles.to(device)
    labels = labels.to(device)
    
    print("\nInput shapes:")
    print(f"Tiles: {tiles.shape}")
    print(f"Labels: {labels.shape}")
    
    # Forward pass
    try:
        logits, Y_prob, Y_hat, A = model(tiles)
        
        print("\nOutput shapes:")
        print(f"Logits: {logits.shape}")
        print(f"Probabilities: {Y_prob.shape}")
        print(f"Predictions: {Y_hat.shape}")
        print(f"Attention weights: {A.shape}")
        
        # Calculate loss
        loss = model.calculate_loss(logits, labels)
        print(f"\nLoss: {loss.detach().item():.4f}")
        
        print("\nPrediction details:")
        print(f"Predicted class: {Y_hat.item()}")
        print(f"Class probabilities: {Y_prob.detach().cpu().numpy()[0]}")
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model() 