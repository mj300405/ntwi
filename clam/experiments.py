import os
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import psutil
import torch
import time

def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    # Get system information
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    gpu_cores = 20  # M4 MacBook Pro has 20 GPU cores
    
    # Calculate optimal workers
    # For MPS, we want to be more conservative with parallelization
    # to avoid GPU memory contention
    workers = min(
        int(cpu_count * 0.5),  # Reduced from 0.75 to 0.5 for MPS
        int(memory_gb / 4),     # Increased memory per worker for MPS
        gpu_cores // 2          # Use half of GPU cores to avoid contention
    )
    
    return max(1, workers)  # Ensure at least 1 worker

def run_experiment(args):
    """Run a single experiment with given parameters"""
    # Set up environment variables for MPS if available
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_ALLOCATOR'] = 'native'
        print("MPS is available, using Metal Performance Shaders")
    else:
        print("MPS is not available, falling back to CPU")
    
    # Construct the command
    cmd = [
        "python", "-m", "clam.train",
        "--data_dir", args.data_dir,
        "--max_tiles", str(args.max_tiles),
        "--model_size", args.model_size,
        "--dropout", str(args.dropout),
        "--k_sample", str(args.k_sample),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--patience", str(args.patience)
    ]
    
    if args.include_augmented:
        cmd.append("--include_augmented")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False

def run_grid_search(args):
    """Run grid search over hyperparameters"""
    # Define parameter grid
    param_grid = {
        'model_size': ['small', 'big'],
        'dropout': [0.1, 0.25, 0.5],
        'k_sample': [4, 8, 16],
        'batch_size': [2, 4, 8],  # Conservative batch sizes for MPS
        'learning_rate': [0.0001, 0.0005, 0.001]
    }
    
    # Create all combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    print(f"Grid search will test {len(param_combinations)} combinations")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments") / f"grid_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grid configuration
    with open(output_dir / "grid_config.json", 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    # Run experiments sequentially
    for i, params in enumerate(param_combinations):
        print(f"\nRunning experiment {i+1}/{len(param_combinations)}")
        exp_args = argparse.Namespace(
            data_dir=args.data_dir,
            max_tiles=args.max_tiles,
            include_augmented=args.include_augmented,
            model_size=params[0],
            dropout=params[1],
            k_sample=params[2],
            batch_size=params[3],
            learning_rate=params[4],
            num_epochs=args.num_epochs,
            patience=args.patience
        )
        
        try:
            success = run_experiment(exp_args)
            print(f"Experiment {i+1}/{len(param_combinations)} completed: {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"Experiment {i+1}/{len(param_combinations)} failed with error: {e}")
        
        # Add a small delay between experiments to allow system to stabilize
        time.sleep(1)

def run_ablation_study(args):
    """Run ablation study to evaluate the impact of different components"""
    # Define baseline configuration
    baseline_config = {
        'model_size': 'big',
        'dropout': 0.25,
        'k_sample': 8,
        'batch_size': 4,  # Conservative batch size for MPS
        'learning_rate': 0.0005
    }
    
    # Define components to ablate
    components = {
        'attention': {'k_sample': 0},  # No attention
        'dropout': {'dropout': 0.0},   # No dropout
        'small_model': {'model_size': 'small'}  # Smaller model
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments") / f"ablation_study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study configuration
    with open(output_dir / "study_config.json", 'w') as f:
        json.dump({
            'baseline': baseline_config,
            'components': components
        }, f, indent=4)
    
    # Run baseline experiment
    print("Running baseline experiment...")
    baseline_args = argparse.Namespace(
        data_dir=args.data_dir,
        max_tiles=args.max_tiles,
        include_augmented=args.include_augmented,
        **baseline_config,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    run_experiment(baseline_args)
    
    # Run ablation experiments
    for component, config in components.items():
        print(f"\nRunning ablation experiment for {component}...")
        ablation_config = baseline_config.copy()
        ablation_config.update(config)
        
        ablation_args = argparse.Namespace(
            data_dir=args.data_dir,
            max_tiles=args.max_tiles,
            include_augmented=args.include_augmented,
            **ablation_config,
            num_epochs=args.num_epochs,
            patience=args.patience
        )
        run_experiment(ablation_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLAM experiments")
    parser.add_argument("--data_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "train"),
                       help="Path to the data directory (default: project_root/train)")
    parser.add_argument("--max_tiles", type=int, default=100, help="Maximum number of tiles per bag")
    parser.add_argument("--include_augmented", action="store_true", help="Include augmented data in training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    
    subparsers = parser.add_subparsers(dest="command", help="Experiment type")
    
    # Grid search command
    grid_parser = subparsers.add_parser("grid", help="Run grid search")
    
    # Ablation study command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    
    args = parser.parse_args()
    
    if args.command == "grid":
        run_grid_search(args)
    elif args.command == "ablation":
        run_ablation_study(args)
    else:
        parser.print_help() 