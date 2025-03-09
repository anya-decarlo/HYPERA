# HYPERA Experiments

This directory contains scripts for running experiments to evaluate the performance of the HYPERA hyperparameter optimization system.

## Comparison Experiment: SAC vs Baseline

The `run_comparison.py` script runs a comparison experiment between the SAC-based hyperparameter optimization approach and a baseline model with fixed hyperparameters.

### Usage

```bash
python run_comparison.py --data_dir /path/to/dataset --output_dir /path/to/output --n_runs 3 --epochs 100
```

### Arguments

- `--data_dir`: Path to the dataset directory (required)
- `--output_dir`: Path to the output directory (default: "experiment_results")
- `--n_runs`: Number of runs for each method (default: 3)
- `--epochs`: Number of training epochs per run (default: 100)
- `--batch_size`: Batch size for training (default: 4)
- `--model_type`: Type of segmentation model (default: "unet")
- `--in_channels`: Number of input channels (default: 1)
- `--out_channels`: Number of output channels/classes (default: 3)
- `--device`: Device to use for training (default: "cuda" if available, otherwise "cpu")

### Output

The experiment will create a directory structure like:

```
experiment_results/
└── experiment_YYYYMMDD_HHMMSS/
    ├── experiment.log
    ├── results.json
    ├── results_comparison.png
    ├── baseline/
    │   ├── run_0/
    │   ├── run_1/
    │   └── run_2/
    └── sac/
        ├── run_0/
        ├── run_1/
        └── run_2/
```

Each run directory contains:
- Model checkpoints (best and final)
- Training logs
- Performance metrics
- Training curves

### Results Analysis

The experiment automatically analyzes the results and generates:

1. A JSON file (`results.json`) with detailed metrics
2. A visualization (`results_comparison.png`) comparing:
   - Best Dice scores for each method
   - Convergence speed (epoch of best performance)

## Example

```bash
# Run a comparison with 3 runs per method, 100 epochs each
python run_comparison.py --data_dir /data/medical_images --output_dir experiments/results --n_runs 3 --epochs 100
```

## Requirements

- PyTorch
- MONAI
- NumPy
- Matplotlib
- tqdm
