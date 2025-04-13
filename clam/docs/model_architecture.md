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

### Small Model
- Fewer parameters
- Faster training
- Suitable for smaller datasets or limited computational resources

### Big Model (Default)
- More parameters
- Better performance on complex datasets
- Recommended for most use cases
- Improved feature extraction and attention mechanisms

## Architecture Details

### Feature Extraction
- Input size: 224x224 pixels
- Feature dimension: 1024
- Backbone: ResNet-50 pre-trained on ImageNet

### Attention Mechanism
- Number of attention heads: 8
- Gating mechanism: Enabled by default
- Attention scores: Softmax-normalized weights for each tile

### Classification Head
- Input: Weighted sum of features
- Hidden layers: Fully connected layers with dropout
- Output: Binary classification (EGFR mutation status)
- Activation: Sigmoid for probability output

## Memory Considerations

The model architecture is designed with memory efficiency in mind:
- Feature extraction is performed on individual tiles
- Attention mechanism processes features in batches
- Memory usage scales with the number of tiles and batch size

For implementation details and training parameters, see the [Training Documentation](training.md). 