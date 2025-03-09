#!/usr/bin/env python
"""
Script to compare results between standard MONAI UNet and agent-based training.
"""

import os
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_latest_training_dir(base_dir):
    """Find the most recent training directory in the given base directory."""
    training_dirs = glob.glob(os.path.join(base_dir, "training_*"))
    if not training_dirs:
        return None
    return max(training_dirs, key=os.path.getctime)

def load_metrics(training_dir):
    """Load metrics from the training directory."""
    metrics_file = os.path.join(training_dir, "logs", "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return None
    
    try:
        return pd.read_csv(metrics_file)
    except Exception as e:
        print(f"Error loading metrics from {metrics_file}: {e}")
        return None

def plot_comparison(standard_metrics, agent_metrics, output_dir):
    """Plot comparison of metrics between standard and agent-based training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    if standard_metrics is not None:
        plt.plot(standard_metrics['epoch'], standard_metrics['train_loss'], label='Standard UNet')
    if agent_metrics is not None:
        plt.plot(agent_metrics['epoch'], agent_metrics['train_loss'], label='Agent-based UNet')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss_comparison.png'))
    
    # Plot validation loss
    plt.figure(figsize=(12, 6))
    if standard_metrics is not None:
        plt.plot(standard_metrics['epoch'], standard_metrics['val_loss'], label='Standard UNet')
    if agent_metrics is not None:
        plt.plot(agent_metrics['epoch'], agent_metrics['val_loss'], label='Agent-based UNet')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'validation_loss_comparison.png'))
    
    # Plot Dice score
    plt.figure(figsize=(12, 6))
    if standard_metrics is not None:
        plt.plot(standard_metrics['epoch'], standard_metrics['dice_score'], label='Standard UNet')
    if agent_metrics is not None:
        plt.plot(agent_metrics['epoch'], agent_metrics['dice_score'], label='Agent-based UNet')
    plt.title('Dice Score Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dice_score_comparison.png'))
    
    # Plot learning rate (only for agent-based, since standard has fixed LR)
    plt.figure(figsize=(12, 6))
    if standard_metrics is not None and 'learning_rate' in standard_metrics.columns:
        plt.axhline(y=standard_metrics['learning_rate'].iloc[0], linestyle='--', color='blue', label='Standard UNet (fixed)')
    if agent_metrics is not None and 'lr' in agent_metrics.columns:
        plt.plot(agent_metrics['epoch'], agent_metrics['lr'], label='Agent-based UNet (dynamic)')
    plt.title('Learning Rate Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'))
    
    # If agent metrics have weight decay, plot it
    if agent_metrics is not None and 'weight_decay' in agent_metrics.columns:
        plt.figure(figsize=(12, 6))
        if standard_metrics is not None:
            # Standard UNet has fixed weight decay
            plt.axhline(y=standard_metrics['weight_decay'].iloc[0] if 'weight_decay' in standard_metrics.columns 
                        else 1e-5, linestyle='--', color='blue', label='Standard UNet (fixed)')
        plt.plot(agent_metrics['epoch'], agent_metrics['weight_decay'], label='Agent-based UNet (dynamic)')
        plt.title('Weight Decay Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Decay')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(os.path.join(output_dir, 'weight_decay_comparison.png'))
    
    print(f"Comparison plots saved to {output_dir}")

def compare_final_metrics(standard_metrics, agent_metrics):
    """Compare final metrics between standard and agent-based training."""
    if standard_metrics is None or agent_metrics is None:
        print("Cannot compare final metrics: missing data")
        return
    
    # Get the best Dice score for each method
    standard_best_dice = standard_metrics['dice_score'].max()
    agent_best_dice = agent_metrics['dice_score'].max()
    
    # Get the epoch at which the best Dice score was achieved
    standard_best_epoch = standard_metrics.loc[standard_metrics['dice_score'].idxmax(), 'epoch']
    agent_best_epoch = agent_metrics.loc[agent_metrics['dice_score'].idxmax(), 'epoch']
    
    print("\n=== Final Metrics Comparison ===")
    print(f"Standard UNet - Best Dice Score: {standard_best_dice:.4f} (Epoch {standard_best_epoch})")
    print(f"Agent-based UNet - Best Dice Score: {agent_best_dice:.4f} (Epoch {agent_best_epoch})")
    
    # Calculate improvement
    improvement = (agent_best_dice - standard_best_dice) / standard_best_dice * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Compare convergence speed
    if standard_best_dice > 0 and agent_best_dice > 0:
        threshold = 0.95 * max(standard_best_dice, agent_best_dice)
        
        # Find the first epoch where Dice score exceeds the threshold
        standard_convergence = None
        agent_convergence = None
        
        for i, dice in enumerate(standard_metrics['dice_score']):
            if dice >= threshold:
                standard_convergence = standard_metrics.iloc[i]['epoch']
                break
        
        for i, dice in enumerate(agent_metrics['dice_score']):
            if dice >= threshold:
                agent_convergence = agent_metrics.iloc[i]['epoch']
                break
        
        if standard_convergence is not None and agent_convergence is not None:
            print(f"Standard UNet reached {threshold:.4f} Dice score at epoch {standard_convergence}")
            print(f"Agent-based UNet reached {threshold:.4f} Dice score at epoch {agent_convergence}")
            
            if standard_convergence > agent_convergence:
                print(f"Agent-based UNet converged {standard_convergence - agent_convergence} epochs faster")
            elif standard_convergence < agent_convergence:
                print(f"Standard UNet converged {agent_convergence - standard_convergence} epochs faster")
            else:
                print("Both methods converged at the same rate")

def main():
    parser = argparse.ArgumentParser(description="Compare results between standard and agent-based training")
    parser.add_argument("--standard_dir", type=str, required=True, help="Directory containing standard UNet results")
    parser.add_argument("--agent_dir", type=str, required=True, help="Directory containing agent-based UNet results")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Find the latest training directories
    standard_training_dir = find_latest_training_dir(args.standard_dir)
    agent_training_dir = find_latest_training_dir(args.agent_dir)
    
    if standard_training_dir is None:
        print(f"No training directories found in {args.standard_dir}")
    else:
        print(f"Using standard UNet results from: {standard_training_dir}")
    
    if agent_training_dir is None:
        print(f"No training directories found in {args.agent_dir}")
    else:
        print(f"Using agent-based UNet results from: {agent_training_dir}")
    
    # Load metrics
    standard_metrics = load_metrics(standard_training_dir) if standard_training_dir else None
    agent_metrics = load_metrics(agent_training_dir) if agent_training_dir else None
    
    if standard_metrics is None and agent_metrics is None:
        print("No metrics found. Make sure you've run both training scripts.")
        return
    
    # Plot comparison
    plot_comparison(standard_metrics, agent_metrics, args.output_dir)
    
    # Compare final metrics
    compare_final_metrics(standard_metrics, agent_metrics)

if __name__ == "__main__":
    main()
