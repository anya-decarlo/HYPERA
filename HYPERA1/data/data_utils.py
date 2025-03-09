#!/usr/bin/env python
# Data utilities for HYPERA project

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    RandRotate90d, RandFlipd, ToTensord, SpatialPadd,
    RandGaussianNoised, RandAdjustContrastd, RandGaussianSmoothd,
    RandZoomd, Spacingd, Resized, NormalizeIntensityd
)
from monai.data import CacheDataset
import logging
import glob

def get_bbbc039_dataloaders(
    data_dir: str = "BBBC039",
    batch_size: int = 4,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    num_workers: int = 4,
    cache_rate: float = 1.0,
    seed: int = 42,
    spatial_size: list = [256, 256]
):
    """
    Get data loaders for BBBC039 dataset.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        train_val_test_split: Split ratios for train, validation, and test sets
        num_workers: Number of workers for data loading
        cache_rate: Cache rate for dataset
        seed: Random seed for reproducibility
        spatial_size: Spatial size for padding images
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders for train, validation, and test sets
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Handle directory names with spaces - the actual directories have spaces in their names
    base_dir = os.path.dirname(data_dir)
    images_dir = os.path.join(base_dir, "BBBC039 ", "images")
    masks_dir = os.path.join(base_dir, "BBBC039 masks")
    
    # Get all image and label files
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    
    # Check if files exist
    if len(image_files) == 0 or len(mask_files) == 0:
        raise ValueError(f"No image or label files found in {images_dir} or {masks_dir}")
    
    # Check if number of image and label files match
    if len(image_files) != len(mask_files):
        raise ValueError(f"Number of image files ({len(image_files)}) does not match number of label files ({len(mask_files)})")
    
    # Create data dictionaries
    data_dicts = [
        {"image": image_file, "label": mask_file}
        for image_file, mask_file in zip(image_files, mask_files)
    ]
    
    # Shuffle data
    np.random.shuffle(data_dicts)
    
    # Split data into train, validation, and test sets
    n_train = int(len(data_dicts) * train_val_test_split[0])
    n_val = int(len(data_dicts) * train_val_test_split[1])
    
    train_dicts = data_dicts[:n_train]
    val_dicts = data_dicts[n_train:n_train+n_val]
    test_dicts = data_dicts[n_train+n_val:]
    
    # Define transforms for training
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=(256, 256), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.8, max_zoom=1.2),
        ToTensord(keys=["image", "label"])
    ])
    
    # Define transforms for validation and testing
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=(256, 256), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"])
    ])
    
    # Create datasets
    train_dataset = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
        cache_rate=cache_rate
    )
    
    val_dataset = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
        cache_rate=cache_rate
    )
    
    test_dataset = CacheDataset(
        data=test_dicts,
        transform=val_transforms,
        cache_rate=cache_rate
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TIFF files
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TIFF files
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with TIFF files
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def get_synthetic_dataloaders(
    data_dir: str = "synthetic_data",
    batch_size: int = 4,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    num_workers: int = 4,
    cache_rate: float = 1.0,
    seed: int = 42
):
    """
    Get data loaders for synthetic dataset.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        train_val_test_split: Split ratios for train, validation, and test sets
        num_workers: Number of workers for data loading
        cache_rate: Cache rate for dataset
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders for train, validation, and test sets
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get all image and label files
    image_files = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    
    # Check if files exist
    if len(image_files) == 0 or len(label_files) == 0:
        raise ValueError(f"No image or label files found in {data_dir}")
    
    # Check if number of image and label files match
    if len(image_files) != len(label_files):
        raise ValueError(f"Number of image files ({len(image_files)}) does not match number of label files ({len(label_files)})")
    
    # Create data dictionaries
    data_dicts = [
        {"image": image_file, "label": label_file}
        for image_file, label_file in zip(image_files, label_files)
    ]
    
    # Shuffle data
    np.random.shuffle(data_dicts)
    
    # Split data into train, validation, and test sets
    n_train = int(len(data_dicts) * train_val_test_split[0])
    n_val = int(len(data_dicts) * train_val_test_split[1])
    
    train_dicts = data_dicts[:n_train]
    val_dicts = data_dicts[n_train:n_train+n_val]
    test_dicts = data_dicts[n_train+n_val:]
    
    # Define transforms for training
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        ToTensord(keys=["image", "label"])
    ])
    
    # Define transforms for validation and testing
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"])
    ])
    
    # Create datasets
    train_dataset = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
        cache_rate=cache_rate
    )
    
    val_dataset = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
        cache_rate=cache_rate
    )
    
    test_dataset = CacheDataset(
        data=test_dicts,
        transform=val_transforms,
        cache_rate=cache_rate
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
