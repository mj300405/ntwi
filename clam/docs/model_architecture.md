# CLAM Model Architecture

This document describes the architecture of the CLAM (Clustering-constrained Attention Multiple Instance Learning) model used for EGFR mutation prediction.

## Overview

CLAM is a deep learning model designed for whole slide image (WSI) analysis. It uses attention mechanisms to identify important regions in the slide and make predictions based on these regions.

## Model Components

### 1. Feature Extractor

The model uses a pre-trained ResNet-50 as the feature extractor:
- Input: Image tiles (224x224 pixels)
- Output: 1024-dimensional feature vectors
- Pre-trained on ImageNet

### 2. Attention Mechanism

The attention mechanism identifies important regions in the slide:
- Input: Feature vectors from all tiles
- Output: Attention scores for each tile
- Uses a gating mechanism to focus on relevant regions

### 3. Bag Classifier

The bag classifier makes the final prediction:
- Input: Weighted sum of features based on attention scores
- Output: Probability of EGFR mutation

## Model Sizes

The model is available in two sizes:

### Small Model (Default in previous versions)
- Fewer parameters
- Faster training
- Suitable for smaller datasets or limited computational resources

### Big Model (Default in current version)
- More parameters
- Better performance on complex datasets
- Recommended for most use cases
- Improved feature extraction and attention mechanisms

## Training Process

### Early Stopping

The model uses early stopping to prevent overfitting:
- Monitors validation loss during training
- Stops training if loss doesn't improve for a specified number of epochs
- Saves the best model based on lowest loss
- Default patience: 7 epochs
- Default minimum improvement: 0.001

### Checkpointing

The model saves checkpoints during training:
- Regular checkpoints after each epoch
- Best model checkpoint based on lowest loss
- All checkpoints include model state, optimizer state, and training configuration

## Memory Optimization

The model is optimized for Apple Metal (MPS) acceleration:
- Efficient memory management
- Garbage collection after each batch and epoch
- Pin memory enabled for faster data transfer
- Automatic device selection (MPS if available, CPU otherwise)

## Hyperparameters

Key hyperparameters for the model:

### Feature Extraction
- Input size: 224x224 pixels
- Feature dimension: 1024

### Attention Mechanism
- Number of attention heads: 8 (default)
- Gating mechanism: Enabled by default

### Training
- Learning rate: 0.0001 (default)
- Batch size: 1 (default)
- Dropout rate: 0.25 (default)
- Number of epochs: 50 (default)
- Early stopping patience: 7 (default)
- Early stopping minimum delta: 0.001 (default)

## Usage

The model is designed to be used with the training script:

```bash
python -m clam.train
```

This will use the big model by default with early stopping enabled. For more details on training options, see the [Training Documentation](training.md). 