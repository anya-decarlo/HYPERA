#!/usr/bin/env python
# BBBC039 Training Script with Agents and Backpropagation Integration

"""
This script implements a training pipeline for the BBBC039 dataset that integrates
both backpropagation and agent-based hyperparameter optimization.

Integration of Backpropagation and Agents:
------------------------------------------
1. Standard Backpropagation:
   - Used for the primary model training through the standard PyTorch optimization
   - Computes gradients and updates model weights based on the loss function

2. Agent-Based Hyperparameter Optimization:
   - Agents observe training metrics and make decisions to optimize hyperparameters
   - Hyperparameters like learning rate, weight decay, etc. are adjusted by agents

3. Integration Mechanism:
   - Gradient statistics from backpropagation are recorded and shared with agents
   - Agents can use gradient information to make better decisions
   - The shared state manager serves as the communication channel between
     backpropagation and agents

4. Feedback Loop:
   - Backpropagation → Compute gradient statistics → Share with agents
   - Agents → Update hyperparameters → Affect next backpropagation step
   - This creates a continuous feedback loop where each system informs the other

This approach allows us to leverage the strengths of both systems:
- Backpropagation for efficient weight updates
- Agents for dynamic hyperparameter optimization based on training dynamics
"""

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
from torch.utils.tensorboard import SummaryWriter
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

def setup_agent_logging(log_dir):
    """
    Set up CSV logging for agent activities.
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Dictionary of CSV writers for each log type
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create CSV files for different types of logs
    log_files = {
        'hyperparameters': open(os.path.join(log_dir, 'hyperparameters.csv'), 'w', newline=''),
        'agent_actions': open(os.path.join(log_dir, 'agent_actions.csv'), 'w', newline=''),
        'agent_rewards': open(os.path.join(log_dir, 'agent_rewards.csv'), 'w', newline=''),
        'agent_states': open(os.path.join(log_dir, 'agent_states.csv'), 'w', newline=''),
        'metrics': open(os.path.join(log_dir, 'metrics.csv'), 'w', newline='')
    }
    
    # Create CSV writers
    csv_writers = {
        'hyperparameters': csv.writer(log_files['hyperparameters']),
        'agent_actions': csv.writer(log_files['agent_actions']),
        'agent_rewards': csv.writer(log_files['agent_rewards']),
        'agent_states': csv.writer(log_files['agent_states']),
        'metrics': csv.writer(log_files['metrics'])
    }
    
    # Write headers
    csv_writers['hyperparameters'].writerow(['Epoch', 'Agent', 'Parameter', 'Old Value', 'New Value', 'Change Magnitude'])
    csv_writers['agent_actions'].writerow(['Epoch', 'Agent', 'Action', 'Applied'])
    csv_writers['agent_rewards'].writerow(['Epoch', 'Agent', 'Reward', 'Stability', 'Generalization', 'Efficiency'])
    csv_writers['agent_states'].writerow(['Epoch', 'Agent', 'State'])
    csv_writers['metrics'].writerow(['Epoch', 'Loss', 'Dice Score', 'Val Loss', 'Val Dice Score'])
    
    return {'writers': csv_writers, 'files': log_files}

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on BBBC039 dataset with agents")
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
    parser.add_argument("--early_stopping", type=int, default=0, 
                        help="Stop training if no improvement for this many epochs (0 to disable)")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.005,  # Changed from 0.001 to 0.005
        help="Initial learning rate for optimizer"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Agent system specific arguments
    parser.add_argument("--conflict_resolution", type=str, default="priority", 
                        choices=["priority", "voting", "consensus"], 
                        help="Strategy for resolving conflicts between agents")
    parser.add_argument("--agent_update_frequency", type=int, default=1, 
                        help="Global update frequency for all agents")
    
    # Add command line argument for experiment type
    parser.add_argument("--experiment_type", type=str, default="agent_factory", 
                        choices=["agent_factory", "no_agent", "grid_search"], 
                        help="Type of experiment to run")
    
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
                                      "results", "bbbc039", args.experiment_type, f"training_{timestamp}")
            cloud_results_dir = os.path.join(args.output_dir, args.experiment_type, f"training_{timestamp}")
            using_cloud_storage = True
            print(f"Using Google Cloud Storage path: {cloud_results_dir}")
        else:
            # Local path
            results_dir = os.path.join(args.output_dir, args.experiment_type, f"training_{timestamp}")
            using_cloud_storage = False
    else:
        # Default local path
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "results", "bbbc039", args.experiment_type, f"training_{timestamp}")
        using_cloud_storage = False
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for models, logs, and visualizations
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    vis_dir = os.path.join(results_dir, "visualizations")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Create TensorBoard writer with a unique directory
    tb_log_dir = os.path.join(logs_dir, f"tensorboard_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

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
    
    # Image and label paths - Fix paths to use correct nested directories without spaces
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
            
            # Check if files exist
            image_path = os.path.join(image_data_dir, image_file_name)
            label_path = os.path.join(label_data_dir, label_file_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                # Try to find the file by name in the directory
                possible_files = [f for f in os.listdir(image_data_dir) if image_file_name.split('.')[0] in f]
                if possible_files:
                    image_path = os.path.join(image_data_dir, possible_files[0])
                    print(f"Found matching file: {image_path}")
            
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
                # Try to find the file by name in the directory
                possible_files = [f for f in os.listdir(label_data_dir) if label_file_name.split('.')[0] in f]
                if possible_files:
                    label_path = os.path.join(label_data_dir, possible_files[0])
                    print(f"Found matching file: {label_path}")
            
            training_data_dicts.append({
                "image": image_path,
                "label": label_path,
            })
    
    # Create validation data dictionaries
    validation_data_dicts = []
    with open(validation_data_filename, 'r') as file:
        for line in file:
            label_file_name = line.strip()
            image_file_name = line.split('.')[0] + ".tif"
            
            # Check if files exist
            image_path = os.path.join(image_data_dir, image_file_name)
            label_path = os.path.join(label_data_dir, label_file_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                # Try to find the file by name in the directory
                possible_files = [f for f in os.listdir(image_data_dir) if image_file_name.split('.')[0] in f]
                if possible_files:
                    image_path = os.path.join(image_data_dir, possible_files[0])
                    print(f"Found matching file: {image_path}")
            
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
                # Try to find the file by name in the directory
                possible_files = [f for f in os.listdir(label_data_dir) if label_file_name.split('.')[0] in f]
                if possible_files:
                    label_path = os.path.join(label_data_dir, possible_files[0])
                    print(f"Found matching file: {label_path}")
            
            validation_data_dicts.append({
                "image": image_path,
                "label": label_path,
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
    
    # Create a training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TIFF files
        pin_memory=torch.cuda.is_available(),
    )

    # Create a validation data loader
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TIFF files
        pin_memory=torch.cuda.is_available(),
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
    
    # Define loss function
    if args.loss == "Dice":
        loss_function = DiceLoss(
            to_onehot_y=False,  # Labels are already one-hot encoded
            softmax=True,
            include_background=True,
        )
    elif args.loss == "DiceCE":
        # Start with balanced weights for all classes
        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        loss_function = DiceCELoss(
            to_onehot_y=False,  # Labels are already one-hot encoded
            softmax=True,
            include_background=True,
            lambda_ce=0.5,
            lambda_dice=1.0,
            weight=class_weights
        )
    elif args.loss == "Focal":
        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        loss_function = FocalLoss(
            to_onehot_y=False,  # Labels are already one-hot encoded
            weight=class_weights,
            gamma=2.0,
            softmax=True
        )
    
    # Create optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate,
            momentum=0.9, 
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    
    # Define metrics for evaluation
    # Since we're using argmax and one-hot encoding in post-processing,
    # we should set include_background=False as the background is now class 0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
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
            "epoch", "train_loss", "val_loss", "dice_score", 
            "lr", "lambda_ce", "lambda_dice", "class_weights", 
            "threshold", "include_background", "normalization_type"
        ])
    
    # Create a file for logging agent changes
    agent_changes_file = os.path.join(logs_dir, "agent_changes.log")
    save_file(f"Agent changes log - {timestamp}\n" + "=" * 50 + "\n\n", agent_changes_file, cloud_results_dir + "/logs/agent_changes.log" if using_cloud_storage else None)
    
    # Setup metrics logging
    metrics_logs_dir = os.path.join(logs_dir, "metrics_logs")
    os.makedirs(metrics_logs_dir, exist_ok=True)
    metrics_logging = setup_agent_logging(metrics_logs_dir)
    
    # Import agents from HYPERA1/agents
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.shared_state import SharedStateManager
    from agents.agent_factory import AgentFactory
    from agents.agent_coordinator import AgentCoordinator
    
    # Initialize shared state manager
    shared_state_manager = SharedStateManager(
        history_size=100,
        log_dir=results_dir,
        verbose=True,
        total_epochs=args.epochs,
        enable_enhanced_metrics=True
    )
    
    # Initialize agent factory
    agent_factory = AgentFactory(
        shared_state_manager=shared_state_manager,
        log_dir=results_dir,
        device=device,
        verbose=True
    )
    
    # Create agents
    learning_rate_agent = agent_factory.create_learning_rate_agent(
        min_lr=1e-6,
        max_lr=1e-2,
        update_frequency=1,  
        name="learning_rate_agent",
        priority=5  # Highest priority
    )
    
    weight_decay_agent = agent_factory.create_weight_decay_agent(
        min_wd=1e-8,
        max_wd=1e-3,
        update_frequency=1,  
        name="weight_decay_agent",
        priority=4
    )
    
    class_weights_agent = agent_factory.create_class_weights_agent(
        num_classes=5,
        min_weight=0.5,
        max_weight=5.0,
        update_frequency=1,  
        name="class_weights_agent",
        priority=3
    )
    
    loss_function_agent = agent_factory.create_loss_function_agent(
        min_lambda_ce=0.1,
        max_lambda_ce=2.0,
        min_lambda_dice=0.1,
        max_lambda_dice=2.0,
        update_frequency=1,  
        name="loss_function_agent",
        priority=2
    )
    
    normalization_agent = agent_factory.create_normalization_agent(
        update_frequency=1,  
        name="normalization_agent",
        priority=1  # Lowest priority
    )
    
    # Set up agent logging
    agent_logs_dir = os.path.join(logs_dir, "agent_logs")
    agent_logging = setup_agent_logging(agent_logs_dir)
    
    # Define callback for agent changes
    def log_agent_change(parameter_name, result):
        agent_name = result.get("agent_name", "unknown")
        old_value = result.get("old_value", "N/A")
        new_value = result.get("new_value", "N/A")
        change_magnitude = result.get("change_magnitude", 0.0)
        current_epoch = 0 if 'epoch' not in globals() else epoch + 1
        
        # Log to file
        with open(os.path.join(logs_dir, "agent_changes.txt"), "a") as f:
            f.write(f"Epoch {current_epoch}: {agent_name} changed {parameter_name} from {old_value} to {new_value}\n")
        
        # Log to CSV
        agent_logging['writers']['hyperparameters'].writerow([
            current_epoch, agent_name, parameter_name, old_value, new_value, change_magnitude
        ])
        
        # Print to console
        print(f"Agent change: {agent_name} changed {parameter_name} from {old_value} to {new_value}")
    
    # Create agent coordinator with custom callback to log changes
    agent_coordinator = AgentCoordinator(
        shared_state_manager=shared_state_manager,
        conflict_resolution_strategy=args.conflict_resolution,
        update_frequency=args.agent_update_frequency,
        log_dir=logs_dir,
        verbose=True,
        change_callback=log_agent_change
    )
    
    # Register agents based on experiment type
    if args.experiment_type == "agent_factory":
        print("Running experiment with agent factory")
        # Register all agents
        agent_coordinator.register_agent(learning_rate_agent)
        agent_coordinator.register_agent(weight_decay_agent)
        agent_coordinator.register_agent(class_weights_agent)
        agent_coordinator.register_agent(loss_function_agent)
        agent_coordinator.register_agent(normalization_agent)
    elif args.experiment_type == "no_agent":
        print("Running experiment without agents")
        # No agents will be registered, using fixed hyperparameters
        pass
    elif args.experiment_type == "grid_search":
        print("Running experiment with grid search")
        # TODO: Implement grid search logic
        # For now, we'll just use the default hyperparameters
        pass
    else:
        print(f"Unknown experiment type: {args.experiment_type}, using agent_factory")
        # Register all agents as default
        agent_coordinator.register_agent(learning_rate_agent)
        agent_coordinator.register_agent(weight_decay_agent)
        agent_coordinator.register_agent(class_weights_agent)
        agent_coordinator.register_agent(loss_function_agent)
        agent_coordinator.register_agent(normalization_agent)
    
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
                
                # Apply post-processing transforms for Dice calculation
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels)
                
                # Apply post-processing transforms
                val_outputs_post = [post_pred(val_output) for val_output in val_outputs_list]
                val_labels_post = [post_label(val_label) for val_label in val_labels_list]
                
                # Calculate Dice score on post-processed outputs
                dice_metric(y_pred=val_outputs_post, y=val_labels_post)
        
        # Aggregate validation metrics
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        
        # Get mean Dice score
        metric = dice_metric.aggregate().item()
        metric_values.append(metric)
        
        # Print validation metrics
        print(f"Epoch {epoch + 1} validation: Loss: {val_loss:.4f}, Dice: {metric:.4f}")
        
        # Log to TensorBoard
        tb_writer.add_scalar("metrics/dice_score", metric, epoch + 1)
        tb_writer.add_scalar("metrics/val_loss", val_loss, epoch + 1)
        tb_writer.add_scalar("metrics/train_loss", epoch_loss, epoch + 1)
        
        # Update shared state with metrics
        shared_state_manager.record_metrics(epoch + 1, {
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "dice_score": metric,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"] if "weight_decay" in optimizer.param_groups[0] else None,
        })
        
        # Collect gradient statistics for analysis
        if epoch > 0 and len(train_loader) > 0:
            # Sample a batch for gradient analysis
            sample_batch = next(iter(train_loader))
            sample_images, sample_labels = sample_batch["image"].to(device), sample_batch["label"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sample_outputs = model(sample_images)
            sample_loss = loss_function(sample_outputs, sample_labels)
            
            # Backward pass to get gradients
            sample_loss.backward()
            
            # Calculate gradient statistics
            grad_norm = 0.0
            grad_max = 0.0
            grad_mean = 0.0
            grad_std = 0.0
            grad_median = 0.0
            param_count = 0
            
            # Collect all gradients for detailed statistics
            all_grads = []
            
            # Calculate per-layer gradient statistics
            layer_grad_stats = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Overall statistics
                    grad_norm += torch.norm(param.grad).item()
                    grad_max = max(grad_max, torch.max(torch.abs(param.grad)).item())
                    grad_mean += torch.mean(torch.abs(param.grad)).item()
                    param_count += 1
                    
                    # Collect gradients for detailed statistics
                    flat_grad = param.grad.flatten().detach().cpu().numpy()
                    all_grads.extend(flat_grad)
                    
                    # Per-layer statistics
                    layer_name = name.split('.')[0]  # Get the top-level module name
                    if layer_name not in layer_grad_stats:
                        layer_grad_stats[layer_name] = {
                            'norm': 0.0,
                            'max': 0.0,
                            'mean': 0.0,
                            'count': 0
                        }
                    
                    layer_grad_stats[layer_name]['norm'] += torch.norm(param.grad).item()
                    layer_grad_stats[layer_name]['max'] = max(
                        layer_grad_stats[layer_name]['max'], 
                        torch.max(torch.abs(param.grad)).item()
                    )
                    layer_grad_stats[layer_name]['mean'] += torch.mean(torch.abs(param.grad)).item()
                    layer_grad_stats[layer_name]['count'] += 1
            
            # Calculate final statistics
            if param_count > 0:
                grad_mean /= param_count
                
                # Calculate additional statistics if we have gradients
                if all_grads:
                    all_grads = np.array(all_grads)
                    grad_std = np.std(all_grads)
                    grad_median = np.median(np.abs(all_grads))
                
                # Finalize per-layer statistics
                for layer_name in layer_grad_stats:
                    if layer_grad_stats[layer_name]['count'] > 0:
                        layer_grad_stats[layer_name]['mean'] /= layer_grad_stats[layer_name]['count']
                
                # Add gradient statistics to shared state for agents to use
                shared_state_manager.record_metrics(epoch + 1, {
                    "grad_norm": grad_norm,
                    "grad_max": grad_max,
                    "grad_mean": grad_mean,
                    "grad_std": grad_std,
                    "grad_median": grad_median,
                    "grad_zero_fraction": np.mean(np.abs(all_grads) < 1e-8) if len(all_grads) > 0 else 0,
                })
                
                # Add per-layer statistics to shared state
                for layer_name, stats in layer_grad_stats.items():
                    shared_state_manager.record_metrics(epoch + 1, {
                        f"grad_{layer_name}_norm": stats['norm'],
                        f"grad_{layer_name}_max": stats['max'],
                        f"grad_{layer_name}_mean": stats['mean'],
                    })
                
                # Log gradient statistics
                tb_writer.add_scalar("gradients/norm", grad_norm, epoch + 1)
                tb_writer.add_scalar("gradients/max", grad_max, epoch + 1)
                tb_writer.add_scalar("gradients/mean", grad_mean, epoch + 1)
                tb_writer.add_scalar("gradients/std", grad_std, epoch + 1)
                tb_writer.add_scalar("gradients/median", grad_median, epoch + 1)
                
                # Log per-layer gradient statistics
                for layer_name, stats in layer_grad_stats.items():
                    tb_writer.add_scalar(f"gradients/{layer_name}/norm", stats['norm'], epoch + 1)
                    tb_writer.add_scalar(f"gradients/{layer_name}/max", stats['max'], epoch + 1)
                    tb_writer.add_scalar(f"gradients/{layer_name}/mean", stats['mean'], epoch + 1)
                
                # Log to CSV
                metrics_logging['writers']['metrics'].writerow([
                    epoch + 1,
                    "gradient_statistics",
                    f"norm: {grad_norm:.6f}, max: {grad_max:.6f}, mean: {grad_mean:.6f}, " +
                    f"std: {grad_std:.6f}, median: {grad_median:.6f}"
                ])
        
        # Get agent recommendations for hyperparameters
        agent_updates = agent_coordinator.update(epoch + 1)
        
        # Apply agent updates to optimizer and loss function parameters
        for param_name, result in agent_updates.items():
            agent_name = result.get("agent_name", "unknown")
            action = result.get("action", "unknown")
            
            # Log to CSV
            agent_logging['writers']['agent_actions'].writerow([
                epoch + 1, 
                agent_name,
                str(action),
                "Yes"  # If it's in agent_updates, it was applied
            ])
        
        # Analyze validation loss improvement for better agent feedback
        if epoch > 0 and len(val_loss_values) >= 2:
            val_loss_change = val_loss_values[-2] - val_loss_values[-1]
            val_loss_improvement = val_loss_change / val_loss_values[-2] if val_loss_values[-2] > 0 else 0
            
            # Log validation improvement
            tb_writer.add_scalar("metrics/val_loss_improvement", val_loss_improvement, epoch + 1)
            
            # Log to CSV
            metrics_logging['writers']['metrics'].writerow([
                epoch + 1,
                "val_loss_improvement",
                f"{val_loss_improvement:.6f}"
            ])
        
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
        
        # Log metrics, agent states, and rewards to CSV files
        agent_logging['writers']['metrics'].writerow([
            epoch + 1, epoch_loss, metric, val_loss, metric
        ])
        
        # Log agent states and rewards
        for agent_name, agent in agent_coordinator.agents.items():
            if isinstance(agent_name, str):
                # Skip if it's just a string
                continue
            try:
                agent_state = agent.get_state_representation()
                state_str = ','.join([str(x) for x in agent_state])
                agent_logging['writers']['agent_states'].writerow([
                    epoch + 1, agent_name.name, state_str
                ])
            except Exception as e:
                print(f"Error getting state for agent: {e}")
            
            # Log rewards if available
            if hasattr(agent, 'last_reward'):
                reward_components = {}
                if hasattr(agent, 'last_reward_components'):
                    reward_components = agent.last_reward_components
                
                stability = reward_components.get('stability', 0.0)
                generalization = reward_components.get('generalization', 0.0)
                efficiency = reward_components.get('efficiency', 0.0)
                
                agent_logging['writers']['agent_rewards'].writerow([
                    epoch + 1, agent_name.name, agent.last_reward, stability, generalization, efficiency
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
    
    # Save final model
    final_model_file = os.path.join(models_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_file)
    if using_cloud_storage:
        copy_to_gcs(final_model_file, cloud_results_dir + "/models/final_model.pth")
    
    # Close TensorBoard writer
    tb_writer.close()
    
    print(f"Training completed. Results saved to {results_dir}")
    if using_cloud_storage:
        print(f"Results also saved to {cloud_results_dir}")
    
    # Close all log files
    for file_handle in agent_logging['files'].values():
        file_handle.close()
    
    print(f"Agent logs saved to {agent_logs_dir}")
    
    # Copy logs to cloud storage if needed
    if using_cloud_storage:
        cloud_agent_logs_dir = os.path.join(cloud_results_dir, "agent_logs")
        for log_file in os.listdir(agent_logs_dir):
            local_path = os.path.join(agent_logs_dir, log_file)
            cloud_path = os.path.join(cloud_agent_logs_dir, log_file)
            copy_to_gcs(local_path, cloud_path)
        print(f"Agent logs copied to {cloud_agent_logs_dir}")
    
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
    
    # If using agent_factory, add agent-specific hyperparameters
    if args.experiment_type == "agent_factory":
        for agent_name, agent in agent_coordinator.agents.items():
            if isinstance(agent_name, str):
                # Skip if it's just a string
                continue
            agent_state = agent.get_current_state()
            final_hyperparameters[f"agent_{agent_name.name}"] = agent_state
    
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
