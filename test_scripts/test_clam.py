import torch
from clam.models.clam import CLAM
from clam.datasets.egfr_dataset_fixed import EGFRBagDataset
from clam.utils.transforms import get_basic_transform

def test_model_and_dataset():
    # Initialize model
    model = CLAM(
        gate=True,
        size_arg='small',
        dropout=0.25,
        k_sample=8,
        n_classes=2
    )
    
    # Initialize dataset
    dataset = EGFRBagDataset(
        data_dir=None,  # Set to your data directory
        transform=get_basic_transform(),
        max_tiles=100
    )
    
    # Get a batch of data
    data, label = dataset[0]
    print(f"Input shape: {data.shape}")
    print(f"Label: {label}")
    
    # Forward pass
    logits, Y_prob, Y_hat, A = model(data.unsqueeze(0))
    
    print(f"Logits shape: {logits.shape}")
    print(f"Y_prob shape: {Y_prob.shape}")
    print(f"Y_hat: {Y_hat}")
    print(f"Attention weights shape: {A.shape}")
    
    # Calculate loss
    loss = model.calculate_loss(logits, torch.tensor([label]))
    print(f"Loss: {loss.detach().item()}")

if __name__ == "__main__":
    test_model_and_dataset() 