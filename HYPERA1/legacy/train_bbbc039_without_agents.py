#!/usr/bin/env python
# Training script for BBBC039 dataset with standard MONAI UNet model (no agents)

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import multiprocessing
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score
import warnings

import torch
import torch.nn as nn
from monai.config import print_config
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Identity,
    LoadImaged,
    Lambdad,
    AsDiscreted,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    SelectItemsd,
    ToTensord,
    NormalizeIntensityd,
    Spacingd,
    RandRotated,
    RandAdjustContrastd,
    Resized,  # Import Resized transform
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.handlers.utils import from_engine
import subprocess
from pathlib import Path

# Import path to BBBC039 dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy.make_path import make_path

# Function to check if gsutil is available (for Google Cloud Storage)
def is_gsutil_available():
    try:
        subprocess.run(["gsutil", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Function to copy a file to Google Cloud Storage
def copy_to_gcs(local_path, gcs_path):
    if not is_gsutil_available():
        print("Warning: gsutil not available, cannot copy to Google Cloud Storage")
        return False
    
    try:
        # Create the directory structure in GCS
        gcs_dir = os.path.dirname(gcs_path)
        subprocess.run(["gsutil", "mkdir", "-p", gcs_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Copy the file
        subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
        print(f"Copied {local_path} to {gcs_path}")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error copying to Google Cloud Storage: {e}")
        return False

# Function to save a file locally and optionally to Google Cloud Storage
def save_file(content, local_path, gcs_path=None):
    # Save locally
    with open(local_path, "w") as f:
        if isinstance(content, str):
            f.write(content)
        elif isinstance(content, dict) or isinstance(content, list):
            json.dump(content, f, indent=4)
        else:
            f.write(str(content))
    
    # Copy to GCS if needed
    if gcs_path:
        copy_to_gcs(local_path, gcs_path)

# Function to print system configuration
def print_config():
    print("\n=== System Configuration ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("===========================\n")

def select_first_channel(x):
    """Select only the first channel of the input tensor."""
    if x.shape[0] > 1:
        return x[0:1]
    return x

def convert_to_onehot(x, num_classes=5):
    """Convert a tensor to one-hot encoding with the specified number of classes."""
    # Make sure x has only one channel
    if x.shape[0] > 1:
        x = x[0:1]
    return AsDiscrete(to_onehot=num_classes)(x)

def convert_to_five_class_onehot(x):
    """Convert a tensor to one-hot encoding with 5 classes (BBBC039 dataset)."""
    return convert_to_onehot(x, 5)

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on BBBC039 dataset")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["Adam", "SGD", "RMSprop"], 
                        help="Optimizer type")
    parser.add_argument("--loss", type=str, default="DiceCE", choices=["Dice", "DiceCE", "Focal"], 
                        help="Loss function")
    parser.add_argument("--augmentations", type=str, default="All", 
                        choices=["None", "Flipping", "Rotation", "Scaling", "Brightness", "All"], 
                        help="Augmentation strategy")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size for training")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--early_stopping", type=int, default=30, 
                        help="Stop training if no improvement for this many epochs (0 to disable)")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.005,  # Changed from 0.001 to 0.005 to match agent version
        help="Initial learning rate for optimizer"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Add command line argument for output directory (for Google Cloud compatibility)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (can be a Google Cloud Storage path)")
    
    args = parser.parse_args()
    
    print_config()

    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory with timestamp to avoid overwriting
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # If output_dir is provided, use it as the base directory
    if args.output_dir:
        if args.output_dir.startswith("gs://"):
            # Google Cloud Storage path
            # We'll still create local directories for temporary files
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "results", "bbbc039", f"training_{timestamp}")
            cloud_results_dir = os.path.join(args.output_dir, f"training_{timestamp}")
            using_cloud_storage = True
            print(f"Using Google Cloud Storage path: {cloud_results_dir}")
        else:
            # Local path
            results_dir = os.path.join(args.output_dir, f"training_{timestamp}")
            using_cloud_storage = False
    else:
        # Default local path
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "results", "bbbc039", f"training_{timestamp}")
        using_cloud_storage = False
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for models, logs, and visualizations
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    vis_dir = os.path.join(results_dir, "visualizations")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Save hyperparameters
    save_file(vars(args), os.path.join(logs_dir, "hyperparameters.json"), cloud_results_dir + "/logs/hyperparameters.json" if using_cloud_storage else None)

    warnings.filterwarnings("ignore", message=".*ASCII value for tag")
    
    # Load BBBC039 dataset
    print("Loading BBBC039 dataset...")
    
    # Get paths to dataset
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bbbc039_path = os.path.join(base_path, "..")
    
    # Metadata paths
    training_data_filename = os.path.join(bbbc039_path, "BBBC039_metadata/training.txt")
    validation_data_filename = os.path.join(bbbc039_path, "BBBC039_metadata/validation.txt")
    
    # Image and label paths
    image_data_dir = os.path.join(bbbc039_path, "BBBC039 /images/")
    label_data_dir = os.path.join(bbbc039_path, "BBBC039 masks/")
    
    print(f"Training data file: {training_data_filename}")
    print(f"Validation data file: {validation_data_filename}")
    print(f"Image data directory: {image_data_dir}")
    print(f"Label data directory: {label_data_dir}")
    
    # Create training data dictionaries
    training_data_dicts = []
    with open(training_data_filename, 'r') as file:
        for line in file:
            label_file_name = line.strip()
            image_file_name = line.split('.')[0] + ".tif"
            training_data_dicts.append({
                "image": os.path.join(image_data_dir, image_file_name),
                "label": os.path.join(label_data_dir, label_file_name),
            })
    
    # Create validation data dictionaries
    validation_data_dicts = []
    with open(validation_data_filename, 'r') as file:
        for line in file:
            label_file_name = line.strip()
            image_file_name = line.split('.')[0] + ".tif"
            validation_data_dicts.append({
                "image": os.path.join(image_data_dir, image_file_name),
                "label": os.path.join(label_data_dir, label_file_name),
            })
    
    train_files = training_data_dicts[:]
    val_files = validation_data_dicts[:]
    
    print(f"Training samples: {len(train_files)}")
    if len(train_files) > 0:
        print("Sample train file entry: ", train_files[0])
    print(f"Validation samples: {len(val_files)}")
    if len(val_files) > 0:
        print("Sample val file entry: ", val_files[0])
    
    # Define spatial size for patches
    spatial_size = [args.patch_size, args.patch_size]
    
    # Define augmentations based on the selected strategy
    augmentations = []
    if args.augmentations in ["Flipping", "All"]:
        augmentations.append(RandFlipd(keys=["image", "label"], prob=0.5))
        augmentations.append(RandRotate90d(keys=["image", "label"], prob=0.5))
    if args.augmentations in ["Rotation", "All"]:
        augmentations.append(RandRotated(keys=["image", "label"], range_x=0.2, range_y=0.2, prob=0.5, mode=("bilinear", "nearest"), padding_mode="zeros"))
    if args.augmentations in ["Scaling", "All"]:
        augmentations.append(RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5))
        augmentations.append(RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5))
    if args.augmentations in ["Brightness", "All"]:
        augmentations.append(RandAdjustContrastd(keys=["image"], prob=0.5))
    
    # Define transforms for training and validation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Use a fixed spatial size for all images
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
            # Removed Orientationd as it's causing warnings for 2D data
            # Resize to a fixed size to ensure consistency
            Resized(keys=["image", "label"], spatial_size=(256, 256), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Use a fixed crop size for all samples
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[96, 96],
                pos=1,
                neg=1,
                num_samples=4,
                allow_smaller=True,
            ),
            *augmentations,
            # Make sure labels have only one channel before one-hot encoding
            Lambdad(keys=["label"], func=select_first_channel),  
            Lambdad(keys=["label"], func=convert_to_five_class_onehot),  
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
            # Removed Orientationd as it's causing warnings for 2D data
            # Resize to a fixed size to ensure consistency
            Resized(keys=["image", "label"], spatial_size=(256, 256), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Make sure labels have only one channel before one-hot encoding
            Lambdad(keys=["label"], func=select_first_channel),  
            Lambdad(keys=["label"], func=convert_to_five_class_onehot),  
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    
    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=min(4, multiprocessing.cpu_count()), 
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(4, multiprocessing.cpu_count()), 
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    print("Creating UNet model...")
    model = UNet(
        spatial_dims=2,  # 2D images
        in_channels=1,   # Grayscale input
        out_channels=5,  # Background + 4 structures
        channels=(16, 32, 64, 128, 256),  # Feature channels at each level
        strides=(2, 2, 2, 2),  # Downsampling strides
        num_res_units=2,  # Residual units per block
        dropout=args.dropout,
        norm=Norm.INSTANCE,  # Instance normalization
    ).to(device)
    
    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Define loss function
    if args.loss == "Dice":
        loss_function = DiceLoss(
            include_background=True,
            to_onehot_y=False,  # Labels are already one-hot encoded
            softmax=True,
            reduction="mean"
        )
    elif args.loss == "DiceCE":
        loss_function = DiceCELoss(
            include_background=True,
            to_onehot_y=False,  # Labels are already one-hot encoded
            softmax=True,
            lambda_ce=1.0,
            lambda_dice=1.0,
            reduction="mean"
        )
    elif args.loss == "Focal":
        loss_function = FocalLoss(
            include_background=True,
            to_onehot_y=False,  # Labels are already one-hot encoded
            gamma=2.0,
            reduction="mean"
        )
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")
    
    # Define metrics for evaluation
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # Define post-processing transforms - we'll use a default threshold of 0.5 initially
    # The threshold will be updated by the agents during training
    post_pred = Compose([
        EnsureType(), 
        Activations(softmax=True), 
        AsDiscrete(argmax=True, to_onehot=5)  # Use argmax and convert to one-hot with 5 classes
    ])
    
    # For labels, they're already in one-hot format from our transforms
    post_label = Identity()  # Just pass through
    
    # Create CSV file for logging metrics
    csv_filename = os.path.join(logs_dir, "metrics.csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "dice_score", "learning_rate", 
            "lambda_ce", "lambda_dice", "class_weights", 
            "threshold", "include_background", "normalization_type"
        ])
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    best_model_file = os.path.join(results_dir, "models", "best_model.pth")
    
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    
    patience_counter = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Set model to training mode
        model.train()
        epoch_loss = 0
        step = 0
        
        # Training loop
        for batch in train_loader:
            step += 1
            
            # Get batch data
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            # Make sure labels have the right format (already one-hot encoded from transforms)
            if labels.shape[1] == 5:  # If already one-hot encoded (5 classes)
                pass
            elif labels.shape[1] == 1:  # If not one-hot encoded yet
                labels = convert_to_five_class_onehot(labels)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = loss_function(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Step {step}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        dice_metric.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                val_images, val_labels = batch["image"].to(device), batch["label"].to(device)
                
                # Make sure labels have the right format (already one-hot encoded from transforms)
                if val_labels.shape[1] == 5:  # If already one-hot encoded (5 classes)
                    pass
                elif val_labels.shape[1] == 1:  # If not one-hot encoded yet
                    val_labels = convert_to_five_class_onehot(val_labels)
                
                # Forward pass
                val_outputs = model(val_images)
                
                # Calculate loss
                val_loss += loss_function(val_outputs, val_labels).item()
                
                # Calculate Dice score
                dice_metric(y_pred=val_outputs, y=val_labels)
        
        # Aggregate validation metrics
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        
        # Get mean Dice score
        metric = dice_metric.aggregate().item()
        metric_values.append(metric)
        
        # Remove scheduler step
        # scheduler.step(metric)
        
        # Log metrics to CSV
        lambda_ce = loss_function.lambda_ce if hasattr(loss_function, "lambda_ce") else None
        lambda_dice = loss_function.lambda_dice if hasattr(loss_function, "lambda_dice") else None
        class_weights = loss_function.weight.tolist() if hasattr(loss_function, "weight") and loss_function.weight is not None else None
        
        # Get current threshold from post_pred transform
        current_threshold = 0.5
        for transform in post_pred.transforms:
            if isinstance(transform, AsDiscrete) and hasattr(transform, 'threshold'):
                current_threshold = transform.threshold
        
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                epoch_loss, 
                val_loss, 
                metric,
                optimizer.param_groups[0]["lr"],
                lambda_ce,
                lambda_dice,
                class_weights,
                current_threshold,
                loss_function.include_background if hasattr(loss_function, "include_background") else True,
                "instance_norm"  # Default normalization type
            ])
        
        # Print metrics
        print(f"Epoch {epoch + 1}/{args.epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Dice Score: {metric:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint if best model
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))
            print(f"New best model saved at epoch {epoch + 1} with Dice score: {best_metric:.4f}")
            
            # If using cloud storage, copy the best model to GCS
            if using_cloud_storage:
                best_model_path = os.path.join(models_dir, "best_model.pth")
                gcs_best_model_path = os.path.join(cloud_results_dir, "models", "best_model.pth")
                copy_to_gcs(best_model_path, gcs_best_model_path)
            
            # Reset patience counter
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Save a visualization of the validation results
        if (epoch + 1) % 10 == 0:
            # Get a validation batch for visualization
            val_data = next(iter(val_loader))
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            
            with torch.no_grad():
                val_outputs = model(val_images)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            
            # Plot and save the visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].imshow(val_images[0, 0].cpu().numpy(), cmap="gray")
            axes[0, 0].set_title("Input Image")
            axes[0, 1].imshow(val_labels[0, 0].cpu().numpy(), cmap="gray")
            axes[0, 1].set_title("Ground Truth")
            axes[1, 0].imshow(val_outputs[0][0].cpu().numpy(), cmap="gray")
            axes[1, 0].set_title("Prediction")
            
            # Overlay prediction on input
            overlay = 0.7 * val_images[0, 0].cpu().numpy() + 0.3 * val_outputs[0][0].cpu().numpy()
            axes[1, 1].imshow(overlay, cmap="gray")
            axes[1, 1].set_title("Overlay")
            
            plt.tight_layout()
            vis_path = os.path.join(vis_dir, f"epoch_{epoch + 1}.png")
            plt.savefig(vis_path)
            plt.close()
            
            # If using cloud storage, copy the visualization to GCS
            if using_cloud_storage:
                gcs_vis_path = os.path.join(cloud_results_dir, "visualizations", f"epoch_{epoch + 1}.png")
                copy_to_gcs(vis_path, gcs_vis_path)
    
    print(f"Training completed. Results saved to {results_dir}")
    
    # Print final results
    print(f"\nTraining completed. Best Dice score: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    # Save final hyperparameters
    final_hyperparameters = {
        "optimizer": args.optimizer,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"] if "weight_decay" in optimizer.param_groups[0] else args.weight_decay,
        "loss": args.loss,
        "augmentations": args.augmentations,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "best_metric": float(best_metric),
        "best_metric_epoch": best_metric_epoch,
        "total_epochs": epoch + 1
    }
    
    # Save final hyperparameters to JSON
    final_hyperparams_file = os.path.join(logs_dir, "final_hyperparameters.json")
    save_file(final_hyperparameters, final_hyperparams_file, 
             os.path.join(cloud_results_dir, "logs", "final_hyperparameters.json") if using_cloud_storage else None)
    
    print(f"Final hyperparameters saved to {final_hyperparams_file}")
    
    # If using cloud storage, copy all remaining files
    if using_cloud_storage:
        print("Copying all results to Google Cloud Storage...")
        
        # Copy the entire results directory
        try:
            subprocess.run([
                "gsutil", "-m", "cp", "-r", 
                os.path.join(results_dir, "*"), 
                cloud_results_dir
            ], check=True)
            print(f"All results copied to {cloud_results_dir}")
        except subprocess.SubprocessError as e:
            print(f"Error copying results to Google Cloud Storage: {e}")

if __name__ == "__main__":
    main()
