#!/usr/bin/env python
# Setup Script for Experiments

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import monai
from monai.transforms import (
    LoadImaged, SaveImaged, AddChanneld, ScaleIntensityd,
    RandCropByPosNegLabeld, RandRotate90d, ToTensord
)
from monai.data import list_data_collate, Dataset
from torch.utils.data import DataLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup experiment environment")
    
    parser.add_argument("--source_data", type=str, required=True, help="Path to source data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for prepared data")
    parser.add_argument("--train_samples", type=int, default=20, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=5, help="Number of validation samples")
    parser.add_argument("--test_samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Set up experiment directories."""
    # Create main directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    return train_dir, val_dir, test_dir

def prepare_data(args, train_dir, val_dir, test_dir):
    """Prepare data for experiments."""
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get list of files
    image_files = sorted([f for f in os.listdir(args.source_data) if f.endswith("_image.nii.gz")])
    label_files = sorted([f for f in os.listdir(args.source_data) if f.endswith("_label.nii.gz")])
    
    # Create pairs
    data_pairs = []
    for img_file, lbl_file in zip(image_files, label_files):
        data_pairs.append({
            "image": os.path.join(args.source_data, img_file),
            "label": os.path.join(args.source_data, lbl_file)
        })
    
    # Shuffle and split
    random.shuffle(data_pairs)
    
    total_samples = min(len(data_pairs), args.train_samples + args.val_samples + args.test_samples)
    
    if total_samples < args.train_samples + args.val_samples + args.test_samples:
        logging.warning(f"Not enough data samples. Using {total_samples} samples in total.")
    
    train_samples = min(args.train_samples, total_samples)
    val_samples = min(args.val_samples, total_samples - train_samples)
    test_samples = min(args.test_samples, total_samples - train_samples - val_samples)
    
    train_pairs = data_pairs[:train_samples]
    val_pairs = data_pairs[train_samples:train_samples + val_samples]
    test_pairs = data_pairs[train_samples + val_samples:train_samples + val_samples + test_samples]
    
    # Define transforms
    train_transforms = monai.transforms.Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        SaveImaged(
            keys=["image", "label"],
            output_dir=[train_dir, train_dir],
            output_postfix=["", ""],
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False
        )
    ])
    
    val_transforms = monai.transforms.Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        SaveImaged(
            keys=["image", "label"],
            output_dir=[val_dir, val_dir],
            output_postfix=["", ""],
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False
        )
    ])
    
    test_transforms = monai.transforms.Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        SaveImaged(
            keys=["image", "label"],
            output_dir=[test_dir, test_dir],
            output_postfix=["", ""],
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False
        )
    ])
    
    # Process data
    logging.info(f"Processing {len(train_pairs)} training samples...")
    train_ds = monai.data.Dataset(data=train_pairs, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=list_data_collate)
    for batch_data in train_loader:
        pass
    
    logging.info(f"Processing {len(val_pairs)} validation samples...")
    val_ds = monai.data.Dataset(data=val_pairs, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=list_data_collate)
    for batch_data in val_loader:
        pass
    
    logging.info(f"Processing {len(test_pairs)} test samples...")
    test_ds = monai.data.Dataset(data=test_pairs, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=list_data_collate)
    for batch_data in test_loader:
        pass
    
    # Rename files to standard format
    rename_files(train_dir, "train", train_samples)
    rename_files(val_dir, "val", val_samples)
    rename_files(test_dir, "test", test_samples)
    
    return train_samples, val_samples, test_samples

def rename_files(directory, prefix, num_samples):
    """Rename files to standard format."""
    image_files = sorted([f for f in os.listdir(directory) if f.endswith("_image.nii.gz")])
    label_files = sorted([f for f in os.listdir(directory) if f.endswith("_label.nii.gz")])
    
    for i, (img_file, lbl_file) in enumerate(zip(image_files, label_files)):
        if i >= num_samples:
            break
        
        # Rename image file
        img_src = os.path.join(directory, img_file)
        img_dst = os.path.join(directory, f"image_{i}.nii.gz")
        os.rename(img_src, img_dst)
        
        # Rename label file
        lbl_src = os.path.join(directory, lbl_file)
        lbl_dst = os.path.join(directory, f"label_{i}.nii.gz")
        os.rename(lbl_src, lbl_dst)

def main():
    """Main function to setup experiment environment."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("Setting up experiment environment...")
    
    # Set up directories
    train_dir, val_dir, test_dir = setup_directories(args.output_dir)
    
    # Prepare data
    train_samples, val_samples, test_samples = prepare_data(args, train_dir, val_dir, test_dir)
    
    logging.info(f"Experiment environment setup completed.")
    logging.info(f"Training samples: {train_samples}")
    logging.info(f"Validation samples: {val_samples}")
    logging.info(f"Test samples: {test_samples}")
    logging.info(f"Data prepared in {args.output_dir}")

if __name__ == "__main__":
    main()
