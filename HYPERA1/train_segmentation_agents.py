#!/usr/bin/env python
# Train Segmentation Agents - Uses a pre-trained U-Net model and trains segmentation agents

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    RandRotate90d, RandFlipd, ToTensord
)
from monai.data import CacheDataset
import logging
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tqdm import tqdm

# Add the parent directory to the path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from HYPERA1.data.data_utils import get_bbbc039_dataloaders
from HYPERA1.segmentation.segmentation_state_manager import SegmentationStateManager
from HYPERA1.segmentation.agents.segmentation_agent_factory import SegmentationAgentFactory
from HYPERA1.segmentation.segmentation_agent_coordinator import SegmentationAgentCoordinator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SegmentationAgentTraining")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train segmentation agents')
    
    # Model paths
    parser.add_argument('--model_path', type=str, 
                        default='/Users/anyadecarlo/HYPERA/results_with_agents/agent_factory/training_20250309-042259/models/best_model.pth',
                        help='Path to pre-trained U-Net model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='/Users/anyadecarlo/HYPERA/BBBC039', 
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for data loading')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate for segmentation agents')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor for future rewards')
    parser.add_argument('--update_frequency', type=int, default=1,
                        help='Frequency of agent updates (in batches)')
    
    # Agent parameters
    parser.add_argument('--use_region_agent', action='store_true', 
                        help='Use region agent')
    parser.add_argument('--use_boundary_agent', action='store_true', 
                        help='Use boundary agent')
    parser.add_argument('--use_shape_agent', action='store_true', 
                        help='Use shape agent')
    parser.add_argument('--use_fg_balance_agent', action='store_true', 
                        help='Use foreground-background balance agent')
    parser.add_argument('--use_object_detection_agent', action='store_true', 
                        help='Use object detection agent')
    parser.add_argument('--use_all_agents', action='store_true', 
                        help='Use all segmentation agents')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, 
                        default='results_with_segmentation_agents',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    # If use_all_agents is True, set all individual agent flags to True
    if args.use_all_agents:
        args.use_region_agent = True
        args.use_boundary_agent = True
        args.use_shape_agent = True
        args.use_fg_balance_agent = True
        args.use_object_detection_agent = True
    
    # If no agents are specified, use region agent by default
    if not any([args.use_region_agent, args.use_boundary_agent, 
                args.use_shape_agent, args.use_fg_balance_agent, 
                args.use_object_detection_agent]):
        logger.warning("No agents specified, using region agent by default")
        args.use_region_agent = True
    
    return args

def load_pretrained_model(model_path):
    """Load a pre-trained U-Net model."""
    logger.info(f"Loading pre-trained model from {model_path}")
    
    # Create a 2D U-Net model with 5 output channels to match the saved model
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=5,  # Using 5 output channels to match the saved model
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
        norm=monai.networks.layers.Norm.INSTANCE  # Use instance normalization as in legacy script
    )
    
    try:
        # Try to load the model weights
        model.load_state_dict(torch.load(model_path))
        logger.info("Successfully loaded model weights")
    except RuntimeError as e:
        logger.error(f"Error loading model weights: {e}")
        raise
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def create_segmentation_agents(args, state_manager, device):
    """Create segmentation agents based on command line arguments."""
    logger.info("Creating segmentation agents")
    
    # Create agent factory
    factory = SegmentationAgentFactory(state_manager)
    
    # Create agents based on command line arguments
    agents = {}
    
    if args.use_all_agents:
        agents = factory.create_all_agents()
    else:
        if args.use_region_agent:
            agents["region"] = factory.create_region_agent(
                lr=args.learning_rate,
                gamma=args.gamma,
                update_frequency=args.update_frequency
            )
        
        if args.use_boundary_agent:
            agents["boundary"] = factory.create_boundary_agent(
                lr=args.learning_rate,
                gamma=args.gamma,
                update_frequency=args.update_frequency
            )
        
        if args.use_shape_agent:
            agents["shape"] = factory.create_shape_agent(
                lr=args.learning_rate,
                gamma=args.gamma,
                update_frequency=args.update_frequency
            )
        
        if args.use_fg_balance_agent:
            agents["fg_balance"] = factory.create_fg_balance_agent(
                lr=args.learning_rate,
                gamma=args.gamma,
                update_frequency=args.update_frequency
            )
    
    # If no agents were specified, use region agent by default
    if not agents:
        logger.warning("No agents specified, using region agent by default")
        agents["region"] = factory.create_region_agent(
            lr=args.learning_rate,
            gamma=args.gamma,
            update_frequency=args.update_frequency
        )
    
    # Create agent coordinator
    agent_coordinator = SegmentationAgentCoordinator(
        list(agents.values()),
        state_manager,
        device=device,
        log_dir=os.path.join(args.output_dir, "logs"),
        verbose=args.verbose
    )
    
    return agent_coordinator, agents

def train_segmentation_agents(args):
    """Train segmentation agents using a pre-trained U-Net model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs", "agent_logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations", "epochs"), exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create CSV files for logging metrics, agent actions, and rewards
    metrics_file = os.path.join(output_dir, "logs", "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_dice_initial', 'train_dice_refined', 'val_dice_initial', 'val_dice_refined', 'improvement'])
    
    agent_actions_file = os.path.join(output_dir, "logs", "agent_logs", "agent_actions.csv")
    with open(agent_actions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'agent', 'action', 'action_value'])
    
    agent_rewards_file = os.path.join(output_dir, "logs", "agent_logs", "agent_rewards.csv")
    with open(agent_rewards_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'agent', 'reward', 'reward_components'])
    
    # Load pre-trained model
    model = load_pretrained_model(args.model_path)
    model = model.to(device)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_bbbc039_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_val_test_split=(0.7, 0.15, 0.15),
        num_workers=args.num_workers,
        cache_rate=1.0,
        seed=args.seed,
        spatial_size=[256, 256]  # Ensure consistent spatial dimensions
    )
    
    # Create segmentation state manager
    state_manager = SegmentationStateManager(
        log_dir=os.path.join(output_dir, "logs"),
        verbose=args.verbose
    )
    
    # Create segmentation agents
    agent_coordinator, agents = create_segmentation_agents(args, state_manager, device)
    
    # Training loop
    logger.info("Starting training")
    best_dice = 0.0
    
    # Create lists to store metrics for plotting
    epochs = []
    train_dice_initial_list = []
    train_dice_refined_list = []
    val_dice_initial_list = []
    val_dice_refined_list = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training phase
        model.eval()  # Keep the U-Net in evaluation mode
        train_loss = 0.0
        train_dice_initial = 0.0
        train_dice_refined = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            # Get data
            inputs = batch_data["image"].to(device)
            targets = batch_data["label"].to(device)
            
            # Forward pass through U-Net (without gradient computation)
            with torch.no_grad():
                initial_outputs = model(inputs)
                initial_probs = F.softmax(initial_outputs, dim=1)
                
                # Convert multi-class to binary (foreground = any non-background class)
                # Assuming channel 0 is background and channels 1-4 are different foreground classes
                foreground_probs = torch.sum(initial_probs[:, 1:, ...], dim=1, keepdim=True)
                background_probs = initial_probs[:, 0, ...].unsqueeze(1)
                binary_probs = torch.cat([background_probs, foreground_probs], dim=1)
                
                # Get binary prediction (0 = background, 1 = foreground)
                initial_preds = (binary_probs[:, 1:, ...] > 0.5).float()
            
            # Calculate metrics
            initial_dice = monai.metrics.compute_dice(
                y_pred=initial_preds.unsqueeze(1),
                y=targets.unsqueeze(1),
                include_background=False
            )
            
            # Update state manager with current batch
            state_manager.update_state(
                inputs=inputs,
                targets=targets,
                initial_preds=initial_preds,
                batch_idx=batch_idx,
                epoch=epoch
            )
            
            # Get refined predictions from agent coordinator
            refined_preds = agent_coordinator.refine_segmentation(initial_preds)
            
            refined_dice = monai.metrics.compute_dice(
                y_pred=refined_preds.unsqueeze(1),
                y=targets.unsqueeze(1),
                include_background=False
            )
            
            # Update training metrics - take mean across batch
            train_dice_initial += initial_dice.mean().item()
            train_dice_refined += refined_dice.mean().item()
            
            # Log agent actions and rewards
            for agent_name, agent in agents.items():
                # Log actions
                actions = agent.get_last_actions()
                if actions:
                    with open(agent_actions_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for action_name, action_value in actions.items():
                            writer.writerow([epoch, batch_idx, agent_name, action_name, action_value])
                
                # Log rewards
                reward_info = agent.get_last_reward_info()
                if reward_info:
                    with open(agent_rewards_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch, 
                            batch_idx, 
                            agent_name, 
                            reward_info.get('total_reward', 0.0),
                            str(reward_info.get('components', {}))
                        ])
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Train Batch {batch_idx}/{len(train_loader)} - Initial Dice: {initial_dice.mean().item():.4f}, Refined Dice: {refined_dice.mean().item():.4f}, Improvement: {refined_dice.mean().item() - initial_dice.mean().item():.4f}")
                
                # Save sample images every 50 batches
                if batch_idx % 50 == 0:
                    save_batch_visualization(
                        inputs=inputs,
                        targets=targets,
                        initial_preds=initial_preds,
                        refined_preds=refined_preds,
                        output_dir=os.path.join(output_dir, "visualizations", "batches"),
                        epoch=epoch,
                        batch_idx=batch_idx
                    )
        
        # Compute average training metrics
        train_dice_initial /= len(train_loader)
        train_dice_refined /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_dice_initial = 0.0
        val_dice_refined = 0.0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                # Get data
                inputs = batch_data["image"].to(device)
                targets = batch_data["label"].to(device)
                
                # Forward pass through U-Net
                initial_outputs = model(inputs)
                initial_probs = F.softmax(initial_outputs, dim=1)
                
                # Convert multi-class to binary (foreground = any non-background class)
                # Assuming channel 0 is background and channels 1-4 are different foreground classes
                foreground_probs = torch.sum(initial_probs[:, 1:, ...], dim=1, keepdim=True)
                background_probs = initial_probs[:, 0, ...].unsqueeze(1)
                binary_probs = torch.cat([background_probs, foreground_probs], dim=1)
                
                # Get binary prediction (0 = background, 1 = foreground)
                initial_preds = (binary_probs[:, 1:, ...] > 0.5).float()
                
                # Compute initial Dice score
                initial_dice = monai.metrics.compute_dice(
                    y_pred=initial_preds.unsqueeze(1),
                    y=targets.unsqueeze(1),
                    include_background=False
                )
                
                # Update state manager with current batch
                state_manager.update_state(
                    inputs=inputs,
                    targets=targets,
                    initial_preds=initial_preds,
                    batch_idx=batch_idx,
                    epoch=epoch,
                    is_validation=True
                )
                
                # Get refined predictions from agent coordinator
                refined_preds = agent_coordinator.refine_segmentation(initial_preds)
                
                # Compute refined Dice score
                refined_dice = monai.metrics.compute_dice(
                    y_pred=refined_preds.unsqueeze(1),
                    y=targets.unsqueeze(1),
                    include_background=False
                )
                
                # Update validation metrics
                val_dice_initial += initial_dice.mean().item()
                val_dice_refined += refined_dice.mean().item()
        
        # Compute average validation metrics
        val_dice_initial /= len(val_loader)
        val_dice_refined /= len(val_loader)
        improvement = val_dice_refined - val_dice_initial
        
        # Store metrics for plotting
        epochs.append(epoch + 1)
        train_dice_initial_list.append(train_dice_initial)
        train_dice_refined_list.append(train_dice_refined)
        val_dice_initial_list.append(val_dice_initial)
        val_dice_refined_list.append(val_dice_refined)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - "
                   f"Train Dice Initial: {train_dice_initial:.4f}, "
                   f"Train Dice Refined: {train_dice_refined:.4f}, "
                   f"Val Dice Initial: {val_dice_initial:.4f}, "
                   f"Val Dice Refined: {val_dice_refined:.4f}, "
                   f"Improvement: {improvement:.4f}")
        
        # Log metrics to CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_dice_initial, train_dice_refined, val_dice_initial, val_dice_refined, improvement])
        
        # Save best model
        if val_dice_refined > best_dice:
            best_dice = val_dice_refined
            
            # Save agent coordinator (which will save all agents)
            agent_coordinator.save(os.path.join(output_dir, "models", "agent_coordinator.pt"))
            
            logger.info(f"Saved best model with validation Dice: {best_dice:.4f}")
        
        # Visualize results for the first batch of validation data
        if epoch % 5 == 0 or epoch == args.num_epochs - 1:
            visualize_results(
                model=model,
                agent_coordinator=agent_coordinator,
                val_loader=val_loader,
                device=device,
                output_dir=os.path.join(output_dir, "visualizations", "epochs"),
                epoch=epoch
            )
        
        # Plot and save learning curves
        plot_learning_curves(
            epochs=epochs,
            train_dice_initial=train_dice_initial_list,
            train_dice_refined=train_dice_refined_list,
            val_dice_initial=val_dice_initial_list,
            val_dice_refined=val_dice_refined_list,
            output_dir=os.path.join(output_dir, "visualizations")
        )
    
    # Final evaluation on test set
    evaluate_on_test_set(
        model=model,
        agent_coordinator=agent_coordinator,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    logger.info("Training completed")

def save_batch_visualization(inputs, targets, initial_preds, refined_preds, output_dir, epoch, batch_idx):
    """Save visualization of a batch of images."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    initial_preds_np = initial_preds.cpu().numpy()
    refined_preds_np = refined_preds.cpu().numpy()
    
    # Create figure
    num_samples = min(4, inputs.shape[0])
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(num_samples):
        # Plot input image
        axes[i, 0].imshow(inputs_np[i, 0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(targets_np[i, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot initial prediction
        axes[i, 2].imshow(initial_preds_np[i, 0], cmap='gray')
        axes[i, 2].set_title('Initial Prediction')
        axes[i, 2].axis('off')
        
        # Plot refined prediction
        axes[i, 3].imshow(refined_preds_np[i, 0], cmap='gray')
        axes[i, 3].set_title('Refined Prediction')
        axes[i, 3].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}.png"))
    plt.close()

def plot_learning_curves(epochs, train_dice_initial, train_dice_refined, val_dice_initial, val_dice_refined, output_dir):
    """Plot and save learning curves."""
    plt.figure(figsize=(12, 8))
    
    # Plot Dice scores
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_dice_initial, 'b-', label='Train Initial')
    plt.plot(epochs, train_dice_refined, 'b--', label='Train Refined')
    plt.plot(epochs, val_dice_initial, 'r-', label='Val Initial')
    plt.plot(epochs, val_dice_refined, 'r--', label='Val Refined')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Scores During Training')
    plt.legend()
    plt.grid(True)
    
    # Plot improvement
    plt.subplot(2, 1, 2)
    train_improvement = [r - i for i, r in zip(train_dice_initial, train_dice_refined)]
    val_improvement = [r - i for i, r in zip(val_dice_initial, val_dice_refined)]
    plt.plot(epochs, train_improvement, 'g-', label='Train Improvement')
    plt.plot(epochs, val_improvement, 'm-', label='Val Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement')
    plt.title('Dice Score Improvement')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"))
    plt.close()

def visualize_results(model, agent_coordinator, val_loader, device, output_dir, epoch):
    """Visualize segmentation results."""
    logger.info("Visualizing results")
    
    # Get a batch of validation data
    batch_data = next(iter(val_loader))
    inputs = batch_data["image"].to(device)
    targets = batch_data["label"].to(device)
    
    # Forward pass through U-Net
    with torch.no_grad():
        initial_outputs = model(inputs)
        initial_probs = F.softmax(initial_outputs, dim=1)
        
        # Convert multi-class to binary (foreground = any non-background class)
        # Assuming channel 0 is background and channels 1-4 are different foreground classes
        foreground_probs = torch.sum(initial_probs[:, 1:, ...], dim=1, keepdim=True)
        background_probs = initial_probs[:, 0, ...].unsqueeze(1)
        binary_probs = torch.cat([background_probs, foreground_probs], dim=1)
        
        # Get binary prediction (0 = background, 1 = foreground)
        initial_preds = (binary_probs[:, 1:, ...] > 0.5).float()
        
        # Get refined predictions from agent coordinator
        refined_preds = agent_coordinator.refine_segmentation(initial_preds)
    
    # Convert tensors to numpy arrays
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    initial_preds_np = initial_preds.cpu().numpy()
    refined_preds_np = refined_preds.cpu().numpy()
    
    # Create figure
    num_samples = min(4, inputs.shape[0])
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(num_samples):
        # Plot input image
        axes[i, 0].imshow(inputs_np[i, 0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(targets_np[i, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot initial prediction
        axes[i, 2].imshow(initial_preds_np[i, 0], cmap='gray')
        axes[i, 2].set_title('Initial Prediction')
        axes[i, 2].axis('off')
        
        # Plot refined prediction
        axes[i, 3].imshow(refined_preds_np[i, 0], cmap='gray')
        axes[i, 3].set_title('Refined Prediction')
        axes[i, 3].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"segmentation_results_epoch_{epoch}.png"))
    plt.close()

def evaluate_on_test_set(model, agent_coordinator, test_loader, device, output_dir):
    """Evaluate segmentation agents on the test set."""
    logger.info("Evaluating on test set")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    test_dice_initial = 0.0
    test_dice_refined = 0.0
    
    # Create directory for segmentation masks
    masks_dir = os.path.join(output_dir, "segmentation_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    def save_segmentation_masks_for_rag(predictions, filenames, output_dir):
        """
        Save segmentation masks in a format suitable for Region Adjacency Graph (RAG) creation.
        
        Args:
            predictions: Tensor of binary segmentation masks [batch_size, 1, height, width]
            filenames: List of filenames for the images
            output_dir: Directory to save the masks
        """
        import os
        import numpy as np
        from skimage import measure
        import tifffile
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, filename in enumerate(filenames):
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(filename))[0]
            
            # Convert binary mask to numpy array
            binary_mask = predictions[i, 0].cpu().numpy()
            
            # Label connected components (each object gets a unique ID)
            labeled_mask = measure.label(binary_mask)
            
            # Save the labeled mask as TIFF
            mask_path = os.path.join(output_dir, f"{base_name}_segmentation.tiff")
            tifffile.imwrite(mask_path, labeled_mask.astype(np.uint16))
            
            # Also save as PNG for visualization
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(labeled_mask, cmap='nipy_spectral')
            plt.title(f"Segmentation Mask - {base_name}")
            plt.colorbar(label='Object ID')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_segmentation.png"))
            plt.close()
            
        logger.info(f"Saved {len(filenames)} segmentation masks to {output_dir}")
    
    def visualize_test_batch(inputs, targets, initial_preds, refined_preds, batch_idx, output_dir):
        """Visualize test batch results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert tensors to numpy arrays
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        initial_preds_np = initial_preds.cpu().numpy()
        refined_preds_np = refined_preds.cpu().numpy()
        
        # Create figure
        num_samples = min(4, inputs.shape[0])
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        # Handle the case where num_samples is 1
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            # Plot input image
            axes[i][0].imshow(inputs_np[i, 0], cmap='gray')
            axes[i][0].set_title('Input Image')
            axes[i][0].axis('off')
            
            # Plot ground truth
            axes[i][1].imshow(targets_np[i, 0], cmap='gray')
            axes[i][1].set_title('Ground Truth')
            axes[i][1].axis('off')
            
            # Plot initial prediction
            axes[i][2].imshow(initial_preds_np[i, 0], cmap='gray')
            axes[i][2].set_title('Initial Prediction')
            axes[i][2].axis('off')
            
            # Plot refined prediction
            axes[i][3].imshow(refined_preds_np[i, 0], cmap='gray')
            axes[i][3].set_title('Refined Prediction')
            axes[i][3].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"test_batch_{batch_idx}.png"))
        plt.close()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Get data
            inputs = batch_data["image"].to(device)
            targets = batch_data["label"].to(device)
            
            # Get filenames if available
            filenames = batch_data.get("filename", [f"image_{batch_idx}_{i}" for i in range(inputs.shape[0])])
            
            # Forward pass through U-Net
            initial_outputs = model(inputs)
            initial_probs = F.softmax(initial_outputs, dim=1)
            
            # Convert multi-class to binary (foreground = any non-background class)
            # Assuming channel 0 is background and channels 1-4 are different foreground classes
            foreground_probs = torch.sum(initial_probs[:, 1:, ...], dim=1, keepdim=True)
            background_probs = initial_probs[:, 0, ...].unsqueeze(1)
            binary_probs = torch.cat([background_probs, foreground_probs], dim=1)
            
            # Get binary prediction (0 = background, 1 = foreground)
            initial_preds = (binary_probs[:, 1:, ...] > 0.5).float()
            
            # Compute metrics for initial predictions
            dice_initial = monai.metrics.compute_dice(
                y_pred=initial_preds.unsqueeze(1),
                y=targets.unsqueeze(1),
                include_background=False
            )
            
            # Get refined predictions from agent coordinator
            refined_preds = agent_coordinator.refine_segmentation(initial_preds)
            
            # Compute metrics for refined predictions
            dice_refined = monai.metrics.compute_dice(
                y_pred=refined_preds.unsqueeze(1),
                y=targets.unsqueeze(1),
                include_background=False
            )
            
            # Update test metrics
            test_dice_initial += dice_initial.mean().item()
            test_dice_refined += dice_refined.mean().item()
            
            # Save segmentation masks for RAG creation
            save_segmentation_masks_for_rag(
                refined_preds, 
                filenames, 
                os.path.join(masks_dir, f"batch_{batch_idx}")
            )
            
            # Visualize results for the first few batches
            if batch_idx < 5:
                visualize_test_batch(
                    inputs=inputs,
                    targets=targets,
                    initial_preds=initial_preds,
                    refined_preds=refined_preds,
                    batch_idx=batch_idx,
                    output_dir=os.path.join(output_dir, "visualizations", "test")
                )
    
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
    
    # Visualize results on the validation set
    visualize_results(
        model=model,
        agent_coordinator=agent_coordinator,
        val_loader=test_loader,  # Use test_loader instead of val_loader for final visualization
        device=device,
        output_dir=os.path.join(output_dir, "visualizations", "test"),
        epoch="final"
    )

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Train segmentation agents
    train_segmentation_agents(args)
