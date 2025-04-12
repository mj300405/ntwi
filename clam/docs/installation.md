# Installation Guide

This document provides detailed instructions for setting up the CLAM model environment using `uv`, a fast Python package installer and resolver.

## Prerequisites

- Python 3.8 or higher
- `uv` package installer
- Apple Silicon Mac (M1/M2) for Metal acceleration

## Installing uv

If you don't have `uv` installed, you can install it using one of the following methods:

### Using pip

```bash
pip install uv
```

### Using Homebrew (macOS)

```bash
brew install uv
```

### Using curl (Unix-like systems)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setting Up the Environment

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/clam-egfr.git
cd clam-egfr
```

### 2. Create a Virtual Environment with uv

```bash
uv venv
```

This creates a virtual environment in the `.venv` directory.

### 3. Activate the Virtual Environment

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 4. Install Dependencies with uv

```bash
uv pip install -r requirements.txt
```

## Verifying the Installation

To verify that everything is installed correctly, run:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

You should see output indicating the PyTorch version and whether MPS (Metal Performance Shaders) is available.

## Troubleshooting

### MPS Not Available

If MPS is not available, ensure:
- You're using PyTorch 2.0 or higher
- You're on macOS 12.3 or higher
- You're using an Apple Silicon Mac (M1/M2)

### Package Installation Issues

If you encounter issues with package installation:

```bash
# Try installing with verbose output
uv pip install -r requirements.txt -v

# Try installing packages one by one
uv pip install torch
uv pip install torchvision
# ... and so on
```

### Memory Issues During Training

If you encounter memory issues during training:
- Reduce the `--max_tiles` parameter
- Reduce the `--batch_size` parameter (already set to 1 by default)
- Close other memory-intensive applications

## Alternative Installation Methods

### Using conda

If you prefer using conda:

```bash
# Create a conda environment
conda create -n clam python=3.10
conda activate clam

# Install PyTorch with MPS support
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Using pip

If you prefer using pip directly:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Next Steps

After installation, you can:

1. Prepare your data according to the [data preparation guide](training.md#data-preparation)
2. Start training the model using the [training instructions](training.md#training-process)
3. Explore the [model architecture](model_architecture.md) to understand how CLAM works 