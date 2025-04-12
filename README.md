# CLAM for EGFR Mutation Prediction

This project implements a CLAM (Clustering-constrained Attention Multiple Instance Learning) model for predicting EGFR mutation status from histopathology images.

## Project Structure

```
.
├── clam/
│   ├── models/
│   │   ├── __init__.py
│   │   └── clam.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── egfr_dataset_fixed.py
│   └── utils/
│       ├── __init__.py
│       └── transforms.py
├── requirements.txt
├── test_clam.py
├── train.py
└── README.md
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Testing the Model

To test if the model and dataset are working correctly:

```bash
python test_clam.py
```

### Training the Model

To train the model:

```bash
python train.py --data_dir /path/to/data --num_epochs 10 --batch_size 1 --learning_rate 0.0001
```

Arguments:
- `--data_dir`: Path to the data directory (required)
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 1)
- `--learning_rate`: Learning rate (default: 0.0001)

## Model Architecture

The CLAM model consists of:
1. Feature Extractor: ResNet-based backbone
2. Attention Mechanism: Gated attention for instance-level importance
3. Bag-level Classification: Final prediction layer

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