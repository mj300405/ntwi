# CLAM: Clustering-constrained Attention Multiple Instance Learning

This repository contains a PyTorch implementation of the CLAM (Clustering-constrained Attention Multiple Instance Learning) model for EGFR mutation prediction in histopathology images.

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

### Using uv (Recommended)

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows
```

3. Install dependencies with PyTorch support:
```bash
uv pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu --index-strategy unsafe-best-match -r requirements.txt
```

### Using pip (Alternative)

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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

To train the model:

```bash
python clam/train.py --data_dir /path/to/data --model_size big
```

### Evaluation

To evaluate the model:

```bash
python clam/evaluate.py --model_path /path/to/model.pt --data_dir /path/to/test_data
```

## Documentation

Detailed documentation is available in the `docs` directory:

- [Model Architecture](docs/model_architecture.md)
- [Training Process](docs/training.md)
- [Installation Guide](docs/installation.md)

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