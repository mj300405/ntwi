# Training Documentation

## Overview

This document describes the training process for the CLAM (Clustering-constrained Attention Multiple Instance Learning) model for EGFR mutation prediction.

## Data Structure

The training data should be organized in the following directory structure:

```
data_dir/
├── EGFR_positive/           # Required: Contains positive samples
├── EGFR_negative/           # Required: Contains negative samples
├── EGFR_positive_aug/       # Optional: Contains augmented positive samples
├── EGFR_negative_aug/       # Optional: Contains augmented negative samples
├── EGFR_positive_cnn/       # Optional: Contains CNN-processed positive samples
└── EGFR_negative_cnn/       # Optional: Contains CNN-processed negative samples
```

Each sample directory should contain multiple PNG files (tiles) extracted from the whole slide image.

## Training Process

The training process uses the following components:

1. **Data Loading**: The `EGFRBagDataset` class loads data from the specified directories. By default, it includes both regular and augmented data folders.

2. **Training Loop**: The model is trained for a specified number of epochs with early stopping to prevent overfitting.

3. **Checkpointing**: The best model is saved in the `checkpoints/best_models` directory with a filename that includes performance metrics.

## Command Line Arguments

The training script accepts the following arguments:

### Data Arguments
- `--data_dir`: Path to the data directory (default: project_root/train)
- `--split`: Data split to use (default: train)
- `--max_tiles`: Maximum number of tiles per bag (default: 100)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--include_augmented`: Include augmented data folders in training (default: True)

### Model Arguments
- `--model_size`: Model size (default: big)
- `--dropout`: Dropout rate (default: 0.25)
- `--k_sample`: Number of attention heads (default: 8)
- `--n_classes`: Number of classes (default: 2)

### Training Arguments
- `--num_epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 1)
- `--learning_rate`: Learning rate (default: 0.0005)

### Early Stopping Arguments
- `--patience`: Number of epochs to wait before early stopping (default: 7)
- `--min_delta`: Minimum change in loss to qualify as an improvement (default: 0.001)

## Output and Metrics

The training process generates the following outputs:

1. **Checkpoints**: Regular checkpoints are saved in `checkpoints/run_TIMESTAMP/`.
2. **Best Model**: The best model is saved in `checkpoints/best_models/` with a filename that includes performance metrics (F1, accuracy, AUC).
3. **Metrics**: Training metrics are saved in `checkpoints/run_TIMESTAMP/metrics/`:
   - Confusion matrices (PNG)
   - ROC curves (PNG)
   - Detailed metrics (JSON)
4. **Final Summary**: A comprehensive `final_metrics.json` containing:
   - Best epoch
   - Best metrics (accuracy, precision, recall, F1, AUC)
   - Metrics for all epochs

## Example Usage

```bash
# Train with default settings (includes augmented data)
python -m clam.train

# Train without augmented data
python -m clam.train --no-include_augmented

# Train with custom settings
python -m clam.train --data_dir /path/to/data --num_epochs 100 --learning_rate 0.0005
```

## Augmented Data

The training process can optionally include augmented data from the `EGFR_positive_aug` and `EGFR_negative_aug` directories. This augmented data can help improve model performance by providing more diverse training samples. By default, augmented data is included in the training process.

To disable the use of augmented data, use the `--no-include_augmented` flag when running the training script.

## Troubleshooting

### Memory Issues

If you encounter memory issues:
- Reduce `--max_tiles` to process fewer tiles per bag
- Reduce `--batch_size` (already set to 1 by default)
- Reduce `--num_workers` if using CPU
- The script includes automatic garbage collection for MPS memory management

### Performance Optimization

For optimal performance:
- The script automatically uses MPS acceleration when available
- Garbage collection is performed after each batch and epoch
- Memory is automatically managed between CPU and GPU
- If you experience slowdown, try reducing the batch size or number of tiles 