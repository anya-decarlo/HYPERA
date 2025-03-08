#!/usr/bin/env python
# Experiment Script to Compare SAC-based Hyperparameter Optimization with Baseline

import os
import sys
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_with_agents import train as train_with_agents
from training.train_baseline import train as train_baseline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comparison experiment between SAC-based hyperparameter optimization and baseline")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="experiment_results", help="Path to output directory")
    
    # Experiment arguments
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs for each method")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs per run")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet"], help="Type of segmentation model")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels (classes)")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    
    return parser.parse_args()

def setup_experiment_directories(args):
    """Set up experiment directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    
    # Create main experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories for each method
    baseline_dir = os.path.join(experiment_dir, "baseline")
    sac_dir = os.path.join(experiment_dir, "sac")
    
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(sac_dir, exist_ok=True)
    
    return experiment_dir, baseline_dir, sac_dir

def setup_logging(experiment_dir):
    """Set up logging for the experiment."""
    log_file = os.path.join(experiment_dir, "experiment.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_baseline_experiments(args, baseline_dir, n_runs):
    """Run baseline experiments."""
    results = []
    
    for run in range(n_runs):
        run_dir = os.path.join(baseline_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        
        logging.info(f"Starting baseline run {run+1}/{n_runs}")
        
        # Create arguments for baseline training
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_interval=5,
            device=args.device,
            model_type=args.model_type,
            in_channels=args.in_channels,
            out_channels=args.out_channels
        )
        
        # Run baseline training
        best_dice, best_epoch = train_baseline(train_args)
        
        results.append({
            "run": run,
            "best_dice": best_dice,
            "best_epoch": best_epoch
        })
        
        logging.info(f"Completed baseline run {run+1}/{n_runs}: Best Dice = {best_dice:.4f} at epoch {best_epoch}")
    
    return results

def run_sac_experiments(args, sac_dir, n_runs):
    """Run SAC-based hyperparameter optimization experiments."""
    results = []
    
    for run in range(n_runs):
        run_dir = os.path.join(sac_dir, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        
        logging.info(f"Starting SAC run {run+1}/{n_runs}")
        
        # Create arguments for SAC training
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_interval=5,
            device=args.device,
            model_type=args.model_type,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            use_agents=True,
            agent_update_freq=1,
            # Specify which agents to use
            agent_types=["learning_rate", "weight_decay", "class_weights", "normalization", "loss_function"],
            # Agent configuration
            agent_config={
                "learning_rate": {
                    "initial_lr": 1e-3,
                    "min_lr": 1e-6,
                    "max_lr": 1e-1,
                    "update_frequency": 1,  # Update every epoch
                    "patience": 3,
                    "cooldown": 5
                },
                "weight_decay": {
                    "initial_weight_decay": 1e-5,
                    "min_weight_decay": 1e-8,
                    "max_weight_decay": 1e-2,
                    "update_frequency": 5,  # Update every 5 epochs
                    "patience": 5,
                    "cooldown": 10
                },
                "class_weights": {
                    "num_classes": args.out_channels,
                    "initial_weights": [1.0] * args.out_channels,
                    "update_frequency": 10,  # Update every 10 epochs
                    "patience": 5,
                    "cooldown": 15
                },
                "normalization": {
                    "initial_norm_type": "instance",
                    "update_frequency": 20,  # Update every 20 epochs
                    "patience": 10,
                    "cooldown": 30
                },
                "loss_function": {
                    "loss_type": "dicece",
                    "initial_lambda_ce": 0.5,
                    "initial_lambda_dice": 1.5,
                    "update_frequency": 15,  # Update every 15 epochs
                    "patience": 5,
                    "cooldown": 20
                }
            },
            # Conflict resolution strategy
            conflict_resolution_strategy="priority"
        )
        
        # Run SAC training
        best_dice, best_epoch = train_with_agents(train_args)
        
        results.append({
            "run": run,
            "best_dice": best_dice,
            "best_epoch": best_epoch
        })
        
        logging.info(f"Completed SAC run {run+1}/{n_runs}: Best Dice = {best_dice:.4f} at epoch {best_epoch}")
    
    return results

def analyze_results(baseline_results, sac_results, experiment_dir):
    """Analyze and visualize experiment results."""
    # Extract metrics
    baseline_dice = [r["best_dice"] for r in baseline_results]
    sac_dice = [r["best_dice"] for r in sac_results]
    
    baseline_epochs = [r["best_epoch"] for r in baseline_results]
    sac_epochs = [r["best_epoch"] for r in sac_results]
    
    # Calculate statistics
    baseline_mean_dice = np.mean(baseline_dice)
    sac_mean_dice = np.mean(sac_dice)
    
    baseline_std_dice = np.std(baseline_dice)
    sac_std_dice = np.std(sac_dice)
    
    baseline_mean_epoch = np.mean(baseline_epochs)
    sac_mean_epoch = np.mean(sac_epochs)
    
    # Log results
    logging.info("\n" + "="*50)
    logging.info("EXPERIMENT RESULTS")
    logging.info("="*50)
    logging.info(f"Baseline: Mean Dice = {baseline_mean_dice:.4f} ± {baseline_std_dice:.4f}, Mean Best Epoch = {baseline_mean_epoch:.1f}")
    logging.info(f"SAC: Mean Dice = {sac_mean_dice:.4f} ± {sac_std_dice:.4f}, Mean Best Epoch = {sac_mean_epoch:.1f}")
    logging.info(f"Improvement: {(sac_mean_dice - baseline_mean_dice) / baseline_mean_dice * 100:.2f}%")
    logging.info("="*50)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    
    # Bar chart for Dice scores
    plt.subplot(1, 2, 1)
    methods = ["Baseline", "SAC"]
    means = [baseline_mean_dice, sac_mean_dice]
    stds = [baseline_std_dice, sac_std_dice]
    
    plt.bar(methods, means, yerr=stds, capsize=10, color=["blue", "green"])
    plt.ylabel("Best Dice Score")
    plt.title("Comparison of Best Dice Scores")
    plt.ylim(0.5, 1.0)  # Adjust as needed
    
    # Bar chart for best epochs
    plt.subplot(1, 2, 2)
    means = [baseline_mean_epoch, sac_mean_epoch]
    
    plt.bar(methods, means, color=["blue", "green"])
    plt.ylabel("Epoch of Best Performance")
    plt.title("Comparison of Convergence Speed")
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "results_comparison.png"))
    
    # Save results to file
    results = {
        "baseline": {
            "runs": baseline_results,
            "mean_dice": baseline_mean_dice,
            "std_dice": baseline_std_dice,
            "mean_epoch": baseline_mean_epoch
        },
        "sac": {
            "runs": sac_results,
            "mean_dice": sac_mean_dice,
            "std_dice": sac_std_dice,
            "mean_epoch": sac_mean_epoch
        },
        "improvement_percentage": (sac_mean_dice - baseline_mean_dice) / baseline_mean_dice * 100
    }
    
    import json
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Main function to run the experiment."""
    args = parse_args()
    
    # Set up experiment directories
    experiment_dir, baseline_dir, sac_dir = setup_experiment_directories(args)
    
    # Set up logging
    logger = setup_logging(experiment_dir)
    
    logger.info("Starting comparison experiment")
    logger.info(f"Arguments: {args}")
    
    # Run baseline experiments
    logger.info("Running baseline experiments")
    baseline_results = run_baseline_experiments(args, baseline_dir, args.n_runs)
    
    # Run SAC experiments
    logger.info("Running SAC experiments")
    sac_results = run_sac_experiments(args, sac_dir, args.n_runs)
    
    # Analyze results
    logger.info("Analyzing results")
    results = analyze_results(baseline_results, sac_results, experiment_dir)
    
    logger.info(f"Experiment completed. Results saved to {experiment_dir}")
    
    return results

if __name__ == "__main__":
    main()
