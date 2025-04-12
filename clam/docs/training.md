# Training and Checkpoints

This document explains the training process and checkpoint system for the CLAM model.

## Training Process

The training process is handled by `clam/train.py`. The script supports various command-line arguments to customize the training:

### Data Arguments
- `--data_dir`: Path to the data directory (default: project_root/train)
  - The data directory should have one of the following structures:
    ```
    # For training data (--split train):
    data_dir/
    ├── EGFR_positive/
    ├── EGFR_negative/
    ├── EGFR_positive_cnn/ (optional)
    └── EGFR_negative_cnn/ (optional)

    # For test data (--split test):
    data_dir/
    ├── C-S-EGFR_positive/
    └── C-S-EGFR_negative/
    ```
  - If not specified, the script will look for data in the `train` directory at the project root
- `--split`: Data split to use, either "train" or "test" (default: "train")
- `--max_tiles`: Maximum number of tiles per bag (default: 100)
- `--num_workers`: Number of workers for data loading (default: 4)

### Model Arguments
- `--model_size`: Model size, either "small" or "big" (default: "small")
- `--dropout`: Dropout rate (default: 0.25)
- `--k_sample`: Number of attention heads (default: 8)
- `--n_classes`: Number of classes (default: 2)

### Training Arguments
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 1)
- `--learning_rate`: Learning rate (default: 0.0001)

## Checkpoint System

The checkpoint system automatically saves model checkpoints during training in the `clam/checkpoints` directory. This directory is created automatically and contains all training runs.

### Directory Structure

Each training run creates a new timestamped directory with the following structure:

```
clam/checkpoints/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json           # Training configuration
    ├── checkpoint_epoch_N.pt # Checkpoint for epoch N
    └── best_model.pt         # Best model checkpoint
```

Where:
- Each `run_YYYYMMDD_HHMMSS` directory represents a single training run
- `config.json` contains the training configuration and results
- `checkpoint_epoch_N.pt` files are checkpoints saved after each epoch
- `best_model.pt` is the model checkpoint with the lowest loss

### Checkpoint Contents

Each checkpoint file (`.pt`) contains:
- Model state dictionary
- Optimizer state dictionary
- Current epoch number
- Current loss value
- Training arguments

The `config.json` file contains:
- Current epoch
- Current loss
- Training arguments

### Loading Checkpoints

To load a checkpoint and resume training, use the `load_checkpoint` function:

```python
from clam.train import load_checkpoint

# Load checkpoint
model, optimizer, epoch, loss = load_checkpoint(
    checkpoint_path="path/to/checkpoint.pt",
    model=model,
    optimizer=optimizer,
    device="cpu"  # or "cuda" or "mps"
)
```

### Best Model

The training script automatically saves the best model (based on loss) as `best_model.pt` in each run directory. This is the model you should use for inference.

## Example Usage

```bash
# Train the model using default data directory (project_root/train)
python -m clam.train

# Train with custom data directory
python -m clam.train --data_dir /path/to/data

# Train on test data
python -m clam.train --data_dir /path/to/data --split test

# Train with custom parameters
python -m clam.train \
    --data_dir /path/to/data \
    --split train \
    --model_size big \
    --dropout 0.5 \
    --k_sample 16 \
    --num_epochs 50 \
    --learning_rate 0.00001
``` 