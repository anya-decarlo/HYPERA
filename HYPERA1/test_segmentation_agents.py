#!/usr/bin/env python
# Test Segmentation Agents - Evaluates trained segmentation agents on test data

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, 
    RandRotate90d, RandFlipd, ToTensord
)
from monai.data import CacheDataset
import logging
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to the path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import segmentation modules
from HYPERA1.segmentation.segmentation_state_manager import SegmentationStateManager
from HYPERA1.segmentation.agents.segmentation_agent_factory import SegmentationAgentFactory
from HYPERA1.segmentation.segmentation_agent_coordinator import SegmentationAgentCoordinator

# Import data utilities
from HYPERA1.data.data_utils import get_bbbc039_dataloaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SegmentationAgentTesting")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test segmentation agents')
    
    # Model paths
    parser.add_argument('--model_path', type=str, 
                        default='/Users/anyadecarlo/HYPERA/results_with_agents/agent_factory/training_20250309-042259/models/best_model.pth',
                        help='Path to pre-trained U-Net model')
    parser.add_argument('--agent_dir', type=str, 
                        default='/Users/anyadecarlo/HYPERA/results_with_segmentation_agents/training_latest/models',
                        help='Directory containing trained segmentation agents')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='BBBC039', 
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for testing')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, 
                        default='results_segmentation_test',
                        help='Directory to save results')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of test samples to visualize')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print verbose output')
    
    return parser.parse_args()

def load_pretrained_model(model_path):
    """Load a pre-trained U-Net model."""
    logger.info(f"Loading pre-trained model from {model_path}")
    
    # Create a 2D U-Net model with the same architecture as the trained model
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=2,  # Assuming binary segmentation (background, foreground)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def load_segmentation_agents(agent_dir, state_manager, device):
    """Load trained segmentation agents."""
    logger.info(f"Loading segmentation agents from {agent_dir}")
    
    # Create agent factory
    agent_factory = SegmentationAgentFactory(
        state_manager=state_manager,
        device=device,
        verbose=False
    )
    
    # Load agents
    agents = agent_factory.load_agents(agent_dir)
    
    # Create agent coordinator
    agent_coordinator = SegmentationAgentCoordinator(
        agents=list(agents.values()),
        state_manager=state_manager,
        device=device,
        verbose=False
    )
    
    return agent_coordinator

def test_segmentation_agents(args):
    """Test segmentation agents on the test set."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"testing_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, "testing.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load pre-trained model
    model = load_pretrained_model(args.model_path)
    model = model.to(device)
    
    # Get data loaders (we only need the test loader)
    _, _, test_loader = get_bbbc039_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Create segmentation state manager
    state_manager = SegmentationStateManager(
        device=device,
        log_dir=os.path.join(output_dir, "logs"),
        verbose=args.verbose
    )
    
    # Load segmentation agents
    agent_coordinator = load_segmentation_agents(args.agent_dir, state_manager, device)
    
    # Test on the test set
    logger.info("Starting testing")
    
    # Initialize metrics
    test_dice_initial = 0.0
    test_dice_refined = 0.0
    test_boundary_iou_initial = 0.0
    test_boundary_iou_refined = 0.0
    
    # Store samples for visualization
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            # Get data
            inputs = batch_data["image"].to(device)
            targets = batch_data["label"].to(device)
            
            # Forward pass through U-Net
            initial_outputs = model(inputs)
            initial_probs = F.softmax(initial_outputs, dim=1)
            initial_preds = torch.argmax(initial_probs, dim=1, keepdim=True)
            
            # Update state manager with current batch
            state_manager.update_state(
                inputs=inputs,
                targets=targets,
                initial_preds=initial_preds,
                batch_idx=batch_idx,
                epoch=0,
                is_validation=True
            )
            
            # Get refined predictions from agent coordinator
            refined_preds = agent_coordinator.refine_segmentation(initial_preds)
            
            # Compute Dice score for initial predictions
            dice_initial = monai.metrics.compute_meandice(
                y_pred=initial_preds.unsqueeze(1),
                y=targets.unsqueeze(1)
            )
            
            # Compute Dice score for refined predictions
            dice_refined = monai.metrics.compute_meandice(
                y_pred=refined_preds.unsqueeze(1),
                y=targets.unsqueeze(1)
            )
            
            # Update test metrics
            test_dice_initial += dice_initial.item()
            test_dice_refined += dice_refined.item()
            
            # Store samples for visualization (up to num_visualizations)
            if len(vis_samples) < args.num_visualizations:
                for i in range(min(inputs.shape[0], args.num_visualizations - len(vis_samples))):
                    vis_samples.append({
                        "input": inputs[i].cpu().numpy(),
                        "target": targets[i].cpu().numpy(),
                        "initial_pred": initial_preds[i].cpu().numpy(),
                        "refined_pred": refined_preds[i].cpu().numpy()
                    })
    
    # Compute average test metrics
    test_dice_initial /= len(test_loader)
    test_dice_refined /= len(test_loader)
    
    # Log test results
    logger.info(f"Test Dice (Initial): {test_dice_initial:.4f}")
    logger.info(f"Test Dice (Refined): {test_dice_refined:.4f}")
    logger.info(f"Improvement: {test_dice_refined - test_dice_initial:.4f}")
    
    # Save test results to file
    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Dice (Initial): {test_dice_initial:.4f}\n")
        f.write(f"Test Dice (Refined): {test_dice_refined:.4f}\n")
        f.write(f"Improvement: {test_dice_refined - test_dice_initial:.4f}\n")
    
    # Visualize results
    visualize_results(vis_samples, output_dir)
    
    logger.info("Testing completed")

def visualize_results(samples, output_dir):
    """Visualize segmentation results."""
    logger.info("Visualizing results")
    
    # Create figure
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i, sample in enumerate(samples):
        # Plot input image
        axes[i, 0].imshow(sample["input"][0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(sample["target"][0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot initial prediction
        axes[i, 2].imshow(sample["initial_pred"][0], cmap='gray')
        axes[i, 2].set_title('Initial Prediction')
        axes[i, 2].axis('off')
        
        # Plot refined prediction
        axes[i, 3].imshow(sample["refined_pred"][0], cmap='gray')
        axes[i, 3].set_title('Refined Prediction')
        axes[i, 3].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visualizations", "segmentation_results.png"))
    plt.close()
    
    # Create individual figures for each sample
    for i, sample in enumerate(samples):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot input image
        axes[0].imshow(sample["input"][0], cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(sample["target"][0], cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot initial prediction
        axes[2].imshow(sample["initial_pred"][0], cmap='gray')
        axes[2].set_title('Initial Prediction')
        axes[2].axis('off')
        
        # Plot refined prediction
        axes[3].imshow(sample["refined_pred"][0], cmap='gray')
        axes[3].set_title('Refined Prediction')
        axes[3].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "visualizations", f"sample_{i+1}.png"))
        plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Test segmentation agents
    test_segmentation_agents(args)
