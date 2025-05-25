# Snuffy Implementation

This is an implementation of the Snuffy model for whole slide image classification.

## Setup

1. Install dependencies from the root directory:
```bash
pip install -r ../requirements.txt
```

2. Prepare your dataset:
   - Create a directory structure:
     ```
     data/
     ├── EGFR_positive/
     │   └── slide_1/
     │       ├── tile_1.png
     │       ├── tile_2.png
     │       └── ...
     └── EGFR_negative/
         └── slide_1/
             ├── tile_1.png
             ├── tile_2.png
             └── ...
     ```

## Running the Code

### Test Run
To test the implementation with a small dataset:
```bash
python test_run.py
```

### Training
To train the model:
```bash
python train.py --data_dir /path/to/data \
                --backbone resnet18 \
                --use_adapter \
                --adapter_type dino \
                --batch_size 32 \
                --epochs 100
```

### Evaluation
To evaluate a trained model:
```bash
python eval.py --data_dir /path/to/data \
               --checkpoint_path /path/to/checkpoint.pth \
               --visualize
```

## Model Options

- Backbone: `resnet18` or `vit_small`
- Adapter types: `dino` or `mae`
- Use adapter-based fine-tuning with `--use_adapter`

## Visualization

The evaluation script can generate attention visualizations:
- Attention heatmaps
- Tile-level attention overlays
- Combined visualizations

## Requirements

All dependencies are managed through the root-level `requirements.txt` file. 