#!/usr/bin/env python
# Baseline Training Script for Medical Image Segmentation

import os
import sys
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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline medical image segmentation model")
    
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

def train_epoch(model, train_loader, optimizer, loss_function, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        loss_function: Loss function
        device: Device to use for training
        epoch: Current epoch
        
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
    Train the baseline model.
    
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
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        lambda_ce=0.5,
        lambda_dice=1.5
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Training loop
    best_dice = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    dice_scores = []
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            epoch=epoch
        )
        
        train_losses.append(train_loss)
        
        # Validate if it's validation interval
        if (epoch + 1) % args.val_interval == 0:
            val_loss, dice_score = validate(
                model=model,
                val_loader=val_loader,
                loss_function=loss_function,
                dice_metric=dice_metric,
                device=device
            )
            
            val_losses.append(val_loss)
            dice_scores.append(dice_score)
            
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(dice_score)
            
            # Save best model
            if dice_score > best_dice:
                best_dice = dice_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
                logging.info(f"New best model saved (Dice: {best_dice:.4f})")
        else:
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    
    # Log best performance
    logging.info(f"Training completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    
    # Plot training curves
    epochs_range = list(range(1, args.epochs + 1))
    val_epochs = list(range(args.val_interval, args.epochs + 1, args.val_interval))
    
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, "b-", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(1, 3, 2)
    plt.plot(val_epochs, val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(True)
    
    # Plot dice scores
    plt.subplot(1, 3, 3)
    plt.plot(val_epochs, dice_scores, "g-", label="Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    plt.close()
    
    return best_dice, best_epoch

if __name__ == "__main__":
    import monai
    
    # Parse arguments
    args = parse_args()
    
    # Train model
    best_dice, best_epoch = train(args)
    
    print(f"Training completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
