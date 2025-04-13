# CLAM for EGFR Mutation Prediction

A deep learning model for predicting EGFR mutation status from whole slide images (WSIs) of lung adenocarcinoma using Clustering-constrained Attention Multiple Instance Learning (CLAM).

## Overview

This project implements a CLAM-based model for analyzing whole slide images to predict EGFR mutation status. The model uses attention mechanisms to identify important regions in the slide images and make predictions based on these regions.

## Features

- **Multiple Instance Learning**: Processes whole slide images by dividing them into tiles
- **Attention Mechanism**: Identifies important regions in the slide images
- **Data Augmentation**: Supports augmented data for improved model performance
- **Apple Metal Acceleration**: Optimized for Apple Silicon Macs using MPS
- **Comprehensive Metrics**: Tracks and visualizes model performance during training

## Documentation

For detailed information, please refer to the following documentation:

- [Installation Guide](clam/docs/installation.md) - How to set up the environment
- [Model Architecture](clam/docs/model_architecture.md) - Details about the CLAM model
- [Training Documentation](clam/docs/training.md) - How to train the model

## Affiliation

This project is developed at the [Silesian University of Technology](https://www.polsl.pl/), Poland.

## License

[MIT License](LICENSE)
