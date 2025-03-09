#!/usr/bin/env python
# Visualization Script for Experiment Results

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    
    parser.add_argument("--results_dir", type=str, required=True, help="Path to experiment results directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory for visualizations (default: same as results_dir)")
    
    return parser.parse_args()

def load_experiment_data(results_dir):
    """Load experiment data from results directory."""
    # Load main results
    results_file = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Load training logs for each run
    baseline_metrics = []
    sac_metrics = []
    
    # Load baseline metrics
    baseline_dir = os.path.join(results_dir, "baseline")
    for run_dir in sorted(os.listdir(baseline_dir)):
        if not run_dir.startswith("run_"):
            continue
        
        run_path = os.path.join(baseline_dir, run_dir)
        metrics_file = os.path.join(run_path, "metrics.json")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                baseline_metrics.append(metrics)
    
    # Load SAC metrics
    sac_dir = os.path.join(results_dir, "sac")
    for run_dir in sorted(os.listdir(sac_dir)):
        if not run_dir.startswith("run_"):
            continue
        
        run_path = os.path.join(sac_dir, run_dir)
        metrics_file = os.path.join(run_path, "metrics.json")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                sac_metrics.append(metrics)
    
    return results, baseline_metrics, sac_metrics

def visualize_dice_comparison(results, output_dir):
    """Visualize comparison of Dice scores."""
    baseline_dice = [r["best_dice"] for r in results["baseline"]["runs"]]
    sac_dice = [r["best_dice"] for r in results["sac"]["runs"]]
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        "Method": ["Baseline"] * len(baseline_dice) + ["SAC"] * len(sac_dice),
        "Dice Score": baseline_dice + sac_dice
    })
    
    plt.figure(figsize=(10, 6))
    
    # Box plot
    ax = sns.boxplot(x="Method", y="Dice Score", data=df, palette=["blue", "green"])
    
    # Add individual points
    sns.stripplot(x="Method", y="Dice Score", data=df, color="black", alpha=0.5)
    
    # Add mean line
    means = df.groupby("Method")["Dice Score"].mean()
    for i, method in enumerate(means.index):
        plt.hlines(means[method], i-0.3, i+0.3, colors="red", linestyles="dashed", linewidth=2)
    
    # Add improvement percentage
    improvement = results["improvement_percentage"]
    plt.text(0.5, 0.95, f"Improvement: {improvement:.2f}%", 
             horizontalalignment="center", verticalalignment="center", 
             transform=plt.gca().transAxes, fontsize=12, fontweight="bold",
             bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"))
    
    plt.title("Comparison of Best Dice Scores", fontsize=14, fontweight="bold")
    plt.ylabel("Dice Score", fontsize=12)
    plt.ylim(0.5, 1.0)  # Adjust as needed
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice_comparison.png"), dpi=300)
    plt.close()

def visualize_convergence_comparison(results, output_dir):
    """Visualize comparison of convergence speed."""
    baseline_epochs = [r["best_epoch"] for r in results["baseline"]["runs"]]
    sac_epochs = [r["best_epoch"] for r in results["sac"]["runs"]]
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        "Method": ["Baseline"] * len(baseline_epochs) + ["SAC"] * len(sac_epochs),
        "Best Epoch": baseline_epochs + sac_epochs
    })
    
    plt.figure(figsize=(10, 6))
    
    # Box plot
    ax = sns.boxplot(x="Method", y="Best Epoch", data=df, palette=["blue", "green"])
    
    # Add individual points
    sns.stripplot(x="Method", y="Best Epoch", data=df, color="black", alpha=0.5)
    
    # Add mean line
    means = df.groupby("Method")["Best Epoch"].mean()
    for i, method in enumerate(means.index):
        plt.hlines(means[method], i-0.3, i+0.3, colors="red", linestyles="dashed", linewidth=2)
    
    # Add speedup percentage
    if results["baseline"]["mean_epoch"] > 0:
        speedup = (results["baseline"]["mean_epoch"] - results["sac"]["mean_epoch"]) / results["baseline"]["mean_epoch"] * 100
        plt.text(0.5, 0.95, f"Convergence Speedup: {speedup:.2f}%", 
                horizontalalignment="center", verticalalignment="center", 
                transform=plt.gca().transAxes, fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"))
    
    plt.title("Comparison of Convergence Speed", fontsize=14, fontweight="bold")
    plt.ylabel("Epoch of Best Performance", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Ensure y-axis uses integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=300)
    plt.close()

def visualize_hyperparameter_trajectories(sac_metrics, output_dir):
    """Visualize hyperparameter trajectories for SAC runs."""
    # Check if we have hyperparameter data
    if not sac_metrics or "hyperparameters" not in sac_metrics[0]:
        return
    
    # Get list of hyperparameters
    hyperparams = list(sac_metrics[0]["hyperparameters"][0].keys())
    
    for param in hyperparams:
        plt.figure(figsize=(12, 6))
        
        for i, metrics in enumerate(sac_metrics):
            epochs = list(range(1, len(metrics["hyperparameters"]) + 1))
            values = [hp[param] for hp in metrics["hyperparameters"]]
            
            plt.plot(epochs, values, marker="o", markersize=4, linewidth=2, label=f"Run {i+1}")
        
        plt.title(f"Trajectory of {param}", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(param, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trajectory_{param}.png"), dpi=300)
        plt.close()

def visualize_metrics_over_time(baseline_metrics, sac_metrics, output_dir):
    """Visualize metrics over time for both methods."""
    # Check if we have metrics data
    if not baseline_metrics or not sac_metrics:
        return
    
    # Get list of metrics
    metrics_keys = [key for key in baseline_metrics[0].keys() if key != "hyperparameters"]
    
    for metric in metrics_keys:
        plt.figure(figsize=(12, 6))
        
        # Plot baseline metrics
        for i, metrics in enumerate(baseline_metrics):
            epochs = list(range(1, len(metrics[metric]) + 1))
            values = metrics[metric]
            
            plt.plot(epochs, values, "b-", alpha=0.3)
        
        # Plot mean baseline
        mean_values = np.mean([metrics[metric] for metrics in baseline_metrics], axis=0)
        epochs = list(range(1, len(mean_values) + 1))
        plt.plot(epochs, mean_values, "b-", linewidth=3, label="Baseline (mean)")
        
        # Plot SAC metrics
        for i, metrics in enumerate(sac_metrics):
            epochs = list(range(1, len(metrics[metric]) + 1))
            values = metrics[metric]
            
            plt.plot(epochs, values, "g-", alpha=0.3)
        
        # Plot mean SAC
        mean_values = np.mean([metrics[metric] for metrics in sac_metrics], axis=0)
        epochs = list(range(1, len(mean_values) + 1))
        plt.plot(epochs, mean_values, "g-", linewidth=3, label="SAC (mean)")
        
        plt.title(f"{metric} Over Time", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metric_{metric}.png"), dpi=300)
        plt.close()

def main():
    """Main function to visualize experiment results."""
    args = parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment data
    results, baseline_metrics, sac_metrics = load_experiment_data(args.results_dir)
    
    # Visualize results
    visualize_dice_comparison(results, output_dir)
    visualize_convergence_comparison(results, output_dir)
    visualize_hyperparameter_trajectories(sac_metrics, output_dir)
    visualize_metrics_over_time(baseline_metrics, sac_metrics, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
