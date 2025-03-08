#!/usr/bin/env python
# MONAI Wrapper - Integrates MONAI models with our segmentation agent framework

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord,
    CropForegroundd, Orientationd, Spacingd, EnsureTyped, AsDiscrete
)

class MONAISegmentationWrapper:
    """
    Wrapper for MONAI segmentation models to integrate with our agent framework.
    
    This class provides a bridge between MONAI's UNet and our segmentation agents,
    allowing agents to modify the behavior of the model during training and inference.
    
    Attributes:
        model: MONAI UNet model
        device (torch.device): Device to use for computation
        spatial_size (tuple): Size of the input patches for sliding window inference
        sw_batch_size (int): Batch size for sliding window inference
        overlap (float): Overlap between patches for sliding window inference
        post_pred: Post-processing transform for predictions
        post_label: Post-processing transform for labels
        log_dir (str): Directory for saving logs and checkpoints
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        model: Optional[UNet] = None,
        device: torch.device = None,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        up_kernel_size: Union[int, Tuple[int, ...]] = 3,
        num_res_units: int = 2,
        spatial_size: Tuple[int, ...] = (96, 96, 96),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        log_dir: str = "logs",
        verbose: bool = False
    ):
        """
        Initialize the MONAI segmentation wrapper.
        
        Args:
            model: MONAI UNet model (if None, a new one will be created)
            device: Device to use for computation
            spatial_dims: Number of spatial dimensions
            in_channels: Number of input channels
            out_channels: Number of output channels
            channels: Number of channels in each layer
            strides: Strides for each layer
            kernel_size: Kernel size for convolutions
            up_kernel_size: Kernel size for transposed convolutions
            num_res_units: Number of residual units
            spatial_size: Size of the input patches for sliding window inference
            sw_batch_size: Batch size for sliding window inference
            overlap: Overlap between patches for sliding window inference
            log_dir: Directory for saving logs and checkpoints
            verbose: Whether to print verbose output
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spatial_size = spatial_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.log_dir = log_dir
        self.verbose = verbose
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("MONAISegmentationWrapper")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Create or use provided model
        if model is None:
            self.model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                kernel_size=kernel_size,
                up_kernel_size=up_kernel_size,
                num_res_units=num_res_units
            ).to(self.device)
            
            if self.verbose:
                self.logger.info(f"Created new UNet model with {sum(p.numel() for p in self.model.parameters())} parameters")
        else:
            self.model = model.to(self.device)
            
            if self.verbose:
                self.logger.info(f"Using provided model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Set up post-processing transforms
        self.post_pred = Compose([
            EnsureTyped(keys=["pred"]),
            AsDiscrete(keys=["pred"], argmax=True, to_onehot=out_channels)
        ])
        
        self.post_label = Compose([
            EnsureTyped(keys=["label"]),
            AsDiscrete(keys=["label"], to_onehot=out_channels)
        ])
        
        # Initialize training parameters
        self.optimizer = None
        self.loss_function = None
        self.lr_scheduler = None
        self.current_epoch = 0
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.metric_values = []
        
        # Initialize agent-modifiable parameters
        self.segmentation_threshold = 0.5
        self.normalization_type = "instance"
        self.loss_weights = None
    
    def set_optimizer(self, optimizer_type: str = "Adam", **kwargs):
        """
        Set the optimizer for the model.
        
        Args:
            optimizer_type: Type of optimizer to use
            **kwargs: Additional arguments for the optimizer
        """
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)
        elif optimizer_type == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        if self.verbose:
            self.logger.info(f"Set optimizer to {optimizer_type} with parameters: {kwargs}")
    
    def set_loss_function(self, loss_type: str = "DiceCE", **kwargs):
        """
        Set the loss function for the model.
        
        Args:
            loss_type: Type of loss function to use
            **kwargs: Additional arguments for the loss function
        """
        if loss_type == "DiceCE":
            from monai.losses import DiceCELoss
            self.loss_function = DiceCELoss(**kwargs)
        elif loss_type == "Dice":
            from monai.losses import DiceLoss
            self.loss_function = DiceLoss(**kwargs)
        elif loss_type == "CE":
            from monai.losses import CrossEntropyLoss
            self.loss_function = CrossEntropyLoss(**kwargs)
        elif loss_type == "Focal":
            from monai.losses import FocalLoss
            self.loss_function = FocalLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        if self.verbose:
            self.logger.info(f"Set loss function to {loss_type} with parameters: {kwargs}")
    
    def set_lr_scheduler(self, scheduler_type: str = "CosineAnnealing", **kwargs):
        """
        Set the learning rate scheduler for the model.
        
        Args:
            scheduler_type: Type of scheduler to use
            **kwargs: Additional arguments for the scheduler
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be set before setting LR scheduler")
        
        if scheduler_type == "CosineAnnealing":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)
        elif scheduler_type == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)
        elif scheduler_type == "StepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        if self.verbose:
            self.logger.info(f"Set LR scheduler to {scheduler_type} with parameters: {kwargs}")
    
    def set_segmentation_threshold(self, threshold: float):
        """
        Set the threshold for segmentation.
        
        Args:
            threshold: Threshold value (0-1)
        """
        self.segmentation_threshold = threshold
        
        # Update post-processing transform
        self.post_pred = Compose([
            EnsureTyped(keys=["pred"]),
            AsDiscrete(keys=["pred"], threshold=threshold)
        ])
        
        if self.verbose:
            self.logger.info(f"Set segmentation threshold to {threshold}")
    
    def set_normalization_type(self, norm_type: str):
        """
        Set the normalization type for the model.
        
        Args:
            norm_type: Type of normalization to use (instance, batch, group, layer)
        """
        # This would require recreating the model with the new normalization type
        # For simplicity, we'll just log it here
        self.normalization_type = norm_type
        
        if self.verbose:
            self.logger.info(f"Set normalization type to {norm_type} (note: requires model recreation)")
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """
        Set weights for different components of the loss function.
        
        Args:
            weights: Dictionary of weight names and values
        """
        self.loss_weights = weights
        
        # Update loss function if it has weight attributes
        if self.loss_function is not None:
            for weight_name, weight_value in weights.items():
                if hasattr(self.loss_function, weight_name):
                    setattr(self.loss_function, weight_name, weight_value)
        
        if self.verbose:
            self.logger.info(f"Set loss weights to {weights}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing input and target tensors
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Get input and target
        inputs = batch["image"].to(self.device)
        targets = batch["label"].to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        # Calculate loss
        loss = self.loss_function(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        metrics = {
            "loss": loss.item()
        }
        
        return metrics
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Dictionary containing input and target tensors
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get input and target
            inputs = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)
            
            # Forward pass with sliding window inference
            outputs = sliding_window_inference(
                inputs, self.spatial_size, self.sw_batch_size, self.model, self.overlap
            )
            
            # Calculate loss
            loss = self.loss_function(outputs, targets)
            
            # Post-process outputs and targets
            batch["pred"] = outputs
            batch["label"] = targets
            
            batch = self.post_pred(batch)
            batch = self.post_label(batch)
            
            # Calculate Dice score
            from monai.metrics import DiceMetric
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            dice_score = dice_metric(y_pred=batch["pred"], y=batch["label"]).item()
            
            # Calculate metrics
            metrics = {
                "val_loss": loss.item(),
                "dice_score": dice_score
            }
            
            return metrics
    
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on an image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Segmentation mask tensor
        """
        self.model.eval()
        
        with torch.no_grad():
            # Ensure image is on the correct device
            image = image.to(self.device)
            
            # Forward pass with sliding window inference
            output = sliding_window_inference(
                image, self.spatial_size, self.sw_batch_size, self.model, self.overlap
            )
            
            # Post-process output
            batch = {"pred": output}
            batch = self.post_pred(batch)
            
            return batch["pred"]
    
    def save_checkpoint(self, path: Optional[str] = None, save_optimizer: bool = True) -> str:
        """
        Save a checkpoint of the model.
        
        Args:
            path: Path to save the checkpoint. If None, use default path.
            save_optimizer: Whether to save optimizer state
            
        Returns:
            Path where the checkpoint was saved
        """
        if path is None:
            path = os.path.join(self.log_dir, f"model_epoch_{self.current_epoch}.pt")
        
        # Create checkpoint dictionary
        checkpoint = {
            "model_state": self.model.state_dict(),
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "best_metric_epoch": self.best_metric_epoch,
            "metric_values": self.metric_values,
            "segmentation_threshold": self.segmentation_threshold,
            "normalization_type": self.normalization_type,
            "loss_weights": self.loss_weights
        }
        
        # Add optimizer state if requested
        if save_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
        
        # Add scheduler state if available
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state"] = self.lr_scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        if self.verbose:
            self.logger.info(f"Saved checkpoint to {path}")
        
        return path
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> bool:
        """
        Load a checkpoint of the model.
        
        Args:
            path: Path to load the checkpoint from
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Whether the load was successful
        """
        if not os.path.exists(path):
            self.logger.error(f"Cannot load checkpoint: {path} does not exist")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state"])
            
            # Load other attributes
            self.current_epoch = checkpoint["epoch"]
            self.best_metric = checkpoint["best_metric"]
            self.best_metric_epoch = checkpoint["best_metric_epoch"]
            self.metric_values = checkpoint["metric_values"]
            self.segmentation_threshold = checkpoint.get("segmentation_threshold", 0.5)
            self.normalization_type = checkpoint.get("normalization_type", "instance")
            self.loss_weights = checkpoint.get("loss_weights", None)
            
            # Load optimizer state if requested
            if load_optimizer and "optimizer_state" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            # Load scheduler state if available
            if "scheduler_state" in checkpoint and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            
            if self.verbose:
                self.logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            "segmentation_threshold": self.segmentation_threshold,
            "normalization_type": self.normalization_type,
            "loss_weights": self.loss_weights,
            "learning_rate": self.optimizer.param_groups[0]["lr"] if self.optimizer else None,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "best_metric_epoch": self.best_metric_epoch
        }
