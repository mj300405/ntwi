# CLAM for EGFR Mutation Prediction

This repository implements a Clustering-constrained Attention Multiple Instance Learning (CLAM) model for predicting EGFR mutation status from whole slide images (WSIs) of lung cancer tissue.

## Features

- **Multiple Instance Learning**: Processes whole slide images by dividing them into tiles
- **Attention Mechanism**: Identifies important regions in the slide
- **Apple Metal Acceleration**: Optimized for M1/M2 Macs using Metal Performance Shaders (MPS)
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Checkpoint System**: Saves model checkpoints and best model during training

## Project Structure

```
.
├── clam/
│   ├── models/
│   │   ├── __init__.py
│   │   └── clam.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── egfr_dataset.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── transforms.py
│   ├── docs/
│   │   ├── model_architecture.md
│   │   ├── training.md
│   │   └── installation.md
│   └── checkpoints/
├── dataset_csv/
├── requirements.txt
└── README.md
```

## Installation

We recommend using `uv`, a fast Python package installer and resolver, for setting up the environment. For detailed installation instructions, see the [Installation Guide](clam/docs/installation.md).

Quick start:

```bash
# Install uv
pip install uv

# Clone the repository
git clone https://github.com/yourusername/clam-egfr.git
cd clam-egfr

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## Data Preparation

Organize your data in the following structure:

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

## Usage

### Training

To train the model with default settings:

```bash
python -m clam.train
```

This will:
- Use the "big" model architecture by default
- Train for up to 50 epochs
- Use early stopping with patience of 7 epochs
- Save checkpoints and the best model

#### Advanced Training Options

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

### Testing

To test the model on a test dataset:

```bash
python -m clam.train --data_dir /path/to/test/data --split test
```

## Documentation

- [Installation Guide](clam/docs/installation.md): Detailed setup instructions using `uv`
- [Model Architecture](clam/docs/model_architecture.md): Details about the CLAM model architecture
- [Training Documentation](clam/docs/training.md): Comprehensive guide to training the model

## Apple Metal Acceleration

This implementation is optimized for Apple Silicon (M1/M2) Macs using Metal Performance Shaders (MPS):

- Automatic detection of MPS availability
- Memory optimizations for Metal
- Garbage collection after each batch and epoch
- Pin memory enabled for faster data transfer

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CLAM model architecture based on the paper "Clustering-constrained Attention Multiple Instance Learning"
- ResNet-50 backbone pre-trained on ImageNet

## Dataset

The dataset should be organized as follows:
```
data_dir/
├── patient1/
│   ├── tile1.png
│   ├── tile2.png
│   └── ...
├── patient2/
│   ├── tile1.png
│   ├── tile2.png
│   └── ...
└── ...
```

Each patient folder should contain multiple image tiles, and the label information should be provided in a separate metadata file.

## Model Variants

The model supports different size configurations:
- small: 256 features
- medium: 512 features
- large: 1024 features

## Citation

If you use this code, please cite:
```
@article{lu2020clam,
  title={Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images},
  author={Lu, Ming Y and Williamson, Drew and Wang, Andrew J and Chen, Richard J and Li, Ivy and Shady, Mohammed and Williams, Matthew and Zhang, Tiffany and Oldenburg, Caroline E and Schalper, Kurt A and others},
  journal={Nature Machine Intelligence},
  volume={2},
  number={8},
  pages={464--474},
  year={2020},
  publisher={Nature Publishing Group}
}
```