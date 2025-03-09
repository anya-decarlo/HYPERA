#!/usr/bin/env python
# Training Script with Multi-Agent Hyperparameter Optimization

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd,
    RandCropByPosNegLabeld, RandRotate90d, ToTensord
)
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from agents.shared_state import SharedStateManager
from agents.agent_coordinator import AgentCoordinator
from agents.agent_factory import AgentFactory

# Add parent directory to path to import agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train medical image segmentation with multi-agent hyperparameter optimization")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to output directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (epochs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet"], help="Type of segmentation model")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels (classes)")
    
    # Agent arguments
    parser.add_argument("--use_agents", action="store_true", help="Whether to use hyperparameter agents")
    parser.add_argument("--agent_update_freq", type=int, default=1, help="Frequency of agent updates (epochs)")
    parser.add_argument("--agent_types", type=str, nargs='+', default=[], help="Types of agents to use")
    parser.add_argument("--agent_config", type=dict, default={}, help="Configuration for agents")
    parser.add_argument("--conflict_resolution_strategy", type=str, default="priority", help="Strategy for resolving conflicts between agents")
    
    return parser.parse_args()

def create_data_loaders(data_dir, batch_size):
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        
    Returns:
        train_loader, val_loader
    """
    # Define transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            pos=1,
            neg=1,
            num_samples=4
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        ToTensord(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"])
    ])
    
    # Create datasets
    train_files = [
        {"image": os.path.join(data_dir, "train", f"image_{i}.nii.gz"), 
         "label": os.path.join(data_dir, "train", f"label_{i}.nii.gz")}
        for i in range(20)  # Assuming 20 training samples
    ]
    
    val_files = [
        {"image": os.path.join(data_dir, "val", f"image_{i}.nii.gz"), 
         "label": os.path.join(data_dir, "val", f"label_{i}.nii.gz")}
        for i in range(5)  # Assuming 5 validation samples
    ]
    
    # Create data loaders
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    
    return train_loader, val_loader

def create_model(model_type, in_channels, out_channels, device):
    """
    Create a segmentation model.
    
    Args:
        model_type: Type of segmentation model
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        device: Device to use for training
        
    Returns:
        model
    """
    if model_type == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=1,
            norm="instance"
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)

def calculate_gradient_norm(model):
    """
    Calculate the gradient norm for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    return total_norm

def train_epoch(model, train_loader, optimizer, loss_function, device, epoch, shared_state_manager=None):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        loss_function: Loss function
        device: Device to use for training
        epoch: Current epoch
        shared_state_manager: Shared state manager (optional)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    step = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_data in progress_bar:
        step += 1
        
        # Get data
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm if shared state manager is provided
        if shared_state_manager is not None:
            gradient_norm = calculate_gradient_norm(model)
            shared_state_manager.set_hyperparameter("gradient_norm", gradient_norm)
        
        # Update weights
        optimizer.step()
        
        # Update progress bar
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / step})
    
    return epoch_loss / step

def validate(model, val_loader, loss_function, dice_metric, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_function: Loss function
        dice_metric: Dice metric
        device: Device to use for validation
        
    Returns:
        Average validation loss and dice score
    """
    model.eval()
    val_loss = 0
    step = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            step += 1
            
            # Get data
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            
            # Compute dice score
            dice_metric(y_pred=outputs, y=labels)
    
    # Aggregate dice score
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return val_loss / step, dice_score

def train(args):
    """
    Train the model with multi-agent hyperparameter optimization.
    
    Args:
        args: Command line arguments
        
    Returns:
        Best dice score and epoch
    """
    # Set up device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    # Create model
    model = create_model(args.model_type, args.in_channels, args.out_channels, device)
    
    # Create loss function and metrics
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create shared state manager
    shared_state_manager = SharedStateManager(
        log_dir=args.output_dir,
        verbose=True,
        total_epochs=args.epochs
    )
    
    # Create agents using the agent factory
    agents = {}
    if hasattr(args, 'agent_types') and args.agent_types:
        # Create agents from specified types and configuration
        agent_config = getattr(args, 'agent_config', {})
        agents = AgentFactory.create_agents_from_config(
            config={agent_type: agent_config.get(agent_type, {}) for agent_type in args.agent_types},
            shared_state_manager=shared_state_manager,
            log_dir=args.output_dir,
            verbose=True
        )
    else:
        # Create default agents
        agents = AgentFactory.create_default_agents(
            shared_state_manager=shared_state_manager,
            log_dir=args.output_dir,
            verbose=True
        )
    
    # Create agent coordinator
    agent_coordinator = AgentCoordinator(
        shared_state_manager=shared_state_manager,
        agents=list(agents.values()),
        update_frequency=getattr(args, 'agent_update_freq', 1),
        conflict_resolution_strategy=getattr(args, 'conflict_resolution_strategy', 'priority'),
        log_dir=args.output_dir,
        verbose=True
    )
    
    # Training loop
    best_dice = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            epoch=epoch,
            shared_state_manager=shared_state_manager
        )
        
        # Validate if it's validation interval
        if (epoch + 1) % args.val_interval == 0:
            val_loss, dice_score = validate(
                model=model,
                val_loader=val_loader,
                loss_function=loss_function,
                dice_metric=dice_metric,
                device=device
            )
            
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}")
            
            # Record metrics if using agents
            if shared_state_manager:
                shared_state_manager.record_metrics(epoch, {
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "dice_score": dice_score
                })
            
            # Save best model
            if dice_score > best_dice:
                best_dice = dice_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
                logging.info(f"New best model saved (Dice: {best_dice:.4f})")
        else:
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}")
            
            # Record metrics if using agents
            if shared_state_manager:
                shared_state_manager.record_metrics(epoch, {"loss": train_loss})
        
        # Update agents if enabled
        if args.use_agents and agent_coordinator and (epoch + 1) % args.agent_update_freq == 0:
            agent_updates = agent_coordinator.update(epoch)
            
            # Apply learning rate update if available
            lr_update = agent_updates.get("learning_rate")
            if lr_update is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_update
                logging.info(f"Learning rate updated to {lr_update}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    
    # Log best performance
    logging.info(f"Training completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    
    # Save agent models if enabled
    if args.use_agents and agent_coordinator:
        agent_coordinator.save_agents()
        
        # Visualize metrics
        if shared_state_manager:
            shared_state_manager.visualize_metrics()
            shared_state_manager.save_state()
    
    return best_dice, best_epoch

if __name__ == "__main__":
    import monai
    
    # Parse arguments
    args = parse_args()
    
    # Train model
    best_dice, best_epoch = train(args)
    
    print(f"Training completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
