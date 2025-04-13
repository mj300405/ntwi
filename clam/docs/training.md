# Training the CLAM Model

This document provides instructions for training the CLAM (Clustering-constrained Attention Multiple Instance Learning) model for EGFR mutation prediction.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Apple Metal (MPS) support for acceleration
- Required Python packages (see `requirements.txt`)

## Data Preparation

The model expects data organized in the following structure:

```
project_root/
├── train/
│   ├── EGFR_positive/
│   │   └── [patient folders with image tiles]
│   ├── EGFR_negative/
│   │   └── [patient folders with image tiles]
│   ├── EGFR_positive_cnn/ (optional)
│   │   └── [patient folders with image tiles]
│   └── EGFR_negative_cnn/ (optional)
│       └── [patient folders with image tiles]
└── test/
    ├── C-S-EGFR_positive/
    │   └── [patient folders with image tiles]
    └── C-S-EGFR_negative/
        └── [patient folders with image tiles]
```

Each patient folder should contain image tiles extracted from whole slide images.

## Training Process

### Basic Training

To train the model with default settings:

```bash
python -m clam.train
```

This will:
- Use the "big" model architecture by default
- Train for up to 50 epochs
- Use early stopping with patience of 7 epochs
- Save checkpoints and the best model

### Advanced Training Options

You can customize the training with various command-line arguments:

```bash
python -m clam.train \
  --data_dir /path/to/data \
  --split train \
  --model_size big \
  --num_epochs 100 \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --patience 10 \
  --min_delta 0.0005 \
  --max_tiles 100 \
  --num_workers 4
```

#### Key Parameters

- **Data Parameters**:
  - `--data_dir`: Path to the data directory (default: project_root/train)
  - `--split`: Data split to use (train or test)
  - `--max_tiles`: Maximum number of tiles per bag (default: 100)
  - `--num_workers`: Number of workers for data loading (default: 4)

- **Model Parameters**:
  - `--model_size`: Model size (small or big, default: big)
  - `--dropout`: Dropout rate (default: 0.25)
  - `--k_sample`: Number of attention heads (default: 8)
  - `--n_classes`: Number of classes (default: 2)

- **Training Parameters**:
  - `--num_epochs`: Number of training epochs (default: 50)
  - `--batch_size`: Batch size (default: 1)
  - `--learning_rate`: Learning rate (default: 0.0001)

- **Early Stopping Parameters**:
  - `--patience`: Number of epochs to wait before early stopping (default: 7)
  - `--min_delta`: Minimum change in loss to qualify as an improvement (default: 0.001)

## Early Stopping

The training process includes early stopping to prevent overfitting:

1. The model tracks the best loss achieved during training
2. If the loss doesn't improve by at least `min_delta` for `patience` epochs, training stops
3. The best model (based on lowest loss) is saved
4. Progress information is displayed during training

## Checkpoints and Model Saving

- Checkpoints are saved after each epoch
- The best model (based on lowest loss) is saved separately
- All checkpoints and models are saved in timestamped directories under `clam/checkpoints/`
- Each run directory contains:
  - `config.json`: Training configuration
  - `checkpoint_epoch_X.pt`: Checkpoints for each epoch
  - `best_model.pt`: The best model based on loss

## Monitoring Training

During training, you'll see:
- Progress bar for each epoch
- Current loss values
- Early stopping counter (if no improvement)
- Checkpoint saving information

## Troubleshooting

### Memory Issues

If you encounter memory issues:
- Reduce `--max_tiles` to process fewer tiles per bag
- Reduce `--batch_size` (already set to 1 by default)
- Reduce `--num_workers` if using CPU

### Performance Optimization

For optimal performance on Apple Metal:
- The script automatically uses MPS acceleration when available
- Garbage collection is performed after each batch and epoch

### Data Directory Structure

If you see errors about missing directories, ensure your data follows the expected structure:
- For training: `EGFR_positive/` and `EGFR_negative/` are required
- For testing: `C-S-EGFR_positive/` and `C-S-EGFR_negative/` are required
- Optional CNN directories can be included for additional training data 