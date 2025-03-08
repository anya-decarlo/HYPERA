#!/usr/bin/env python
# Segmentation State Manager - Manages shared state for segmentation agents

import os
import numpy as np
import torch
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import deque

class SegmentationStateManager:
    """
    Manages the shared state for all segmentation agents.
    
    This class tracks metrics, segmentation outputs, ground truth, and other
    information that needs to be shared between segmentation agents.
    
    Attributes:
        log_dir (str): Directory for saving logs and checkpoints
        verbose (bool): Whether to print verbose output
        metrics (dict): Dictionary of metrics
        history_length (int): Length of history to maintain for metrics
        training_phase (str): Current training phase (exploration, exploitation, fine-tuning)
        total_epochs (int): Total number of epochs for training
        current_epoch (int): Current epoch
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        verbose: bool = False,
        history_length: int = 100,
        total_epochs: int = 100
    ):
        """
        Initialize the segmentation state manager.
        
        Args:
            log_dir: Directory for saving logs and checkpoints
            verbose: Whether to print verbose output
            history_length: Length of history to maintain for metrics
            total_epochs: Total number of epochs for training
        """
        self.log_dir = log_dir
        self.verbose = verbose
        self.history_length = history_length
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("SegmentationStateManager")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Initialize metrics tracking
        self.metrics = {}
        self.metric_history = {}
        
        # Initialize segmentation state
        self.current_image = None
        self.current_ground_truth = None
        self.current_prediction = None
        self.current_features = {}
        
        # Initialize training phase
        self.training_phase = "exploration"
        
        # Initialize time tracking
        self.start_time = time.time()
        self.epoch_start_time = self.start_time
        
        if self.verbose:
            self.logger.info("Initialized Segmentation State Manager")
    
    def update_epoch(self, epoch: int):
        """
        Update the current epoch and related information.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        # Update training phase based on epoch
        if epoch < self.total_epochs * 0.3:
            self.training_phase = "exploration"
        elif epoch < self.total_epochs * 0.7:
            self.training_phase = "exploitation"
        else:
            self.training_phase = "fine-tuning"
        
        if self.verbose:
            self.logger.info(f"Updated to epoch {epoch}, phase: {self.training_phase}")
    
    def record_metrics(self, epoch: int, metrics_dict: Dict[str, float]):
        """
        Record metrics for the current epoch.
        
        Args:
            epoch: Current epoch number
            metrics_dict: Dictionary of metrics to record
        """
        self.current_epoch = epoch
        
        # Record metrics
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.metric_history[key] = deque(maxlen=self.history_length)
            
            self.metrics[key].append(value)
            self.metric_history[key].append(value)
        
        # Log metrics
        if self.verbose:
            log_str = f"Epoch {epoch}: "
            for key, value in metrics_dict.items():
                log_str += f"{key}: {value:.4f} | "
            self.logger.info(log_str)
    
    def get_metric(self, key: str, window: int = None) -> Optional[float]:
        """
        Get the value of a metric.
        
        Args:
            key: Name of the metric
            window: Number of recent values to average (None for latest value)
            
        Returns:
            Metric value or None if not available
        """
        if key not in self.metrics or not self.metrics[key]:
            return None
        
        if window is None:
            return self.metrics[key][-1]
        else:
            window = min(window, len(self.metrics[key]))
            return np.mean(self.metrics[key][-window:])
    
    def get_metric_history(self, key: str) -> Optional[List[float]]:
        """
        Get the history of a metric.
        
        Args:
            key: Name of the metric
            
        Returns:
            List of metric values or None if not available
        """
        if key not in self.metrics:
            return None
        
        return self.metrics[key]
    
    def get_metric_delta(self, key: str, window: int = 5) -> Optional[float]:
        """
        Get the rate of change of a metric.
        
        Args:
            key: Name of the metric
            window: Number of recent values to use for calculating delta
            
        Returns:
            Rate of change or None if not available
        """
        if key not in self.metrics or len(self.metrics[key]) < window + 1:
            return None
        
        recent = np.mean(self.metrics[key][-window:])
        previous = np.mean(self.metrics[key][-(window*2):-window])
        
        return recent - previous
    
    def set_current_image(self, image: torch.Tensor):
        """
        Set the current image being processed.
        
        Args:
            image: Current image tensor
        """
        self.current_image = image
    
    def set_current_ground_truth(self, ground_truth: torch.Tensor):
        """
        Set the current ground truth segmentation.
        
        Args:
            ground_truth: Current ground truth segmentation tensor
        """
        self.current_ground_truth = ground_truth
    
    def set_current_prediction(self, prediction: torch.Tensor):
        """
        Set the current prediction segmentation.
        
        Args:
            prediction: Current prediction segmentation tensor
        """
        self.current_prediction = prediction
    
    def set_feature(self, key: str, feature: Any):
        """
        Set a feature in the current feature dictionary.
        
        Args:
            key: Feature name
            feature: Feature value
        """
        self.current_features[key] = feature
    
    def get_feature(self, key: str) -> Optional[Any]:
        """
        Get a feature from the current feature dictionary.
        
        Args:
            key: Feature name
            
        Returns:
            Feature value or None if not available
        """
        return self.current_features.get(key, None)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the current state.
        
        Returns:
            Dictionary of current state
        """
        return {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "training_phase": self.training_phase,
            "metrics": {k: v[-1] if v else None for k, v in self.metrics.items()},
            "metric_deltas": {k: self.get_metric_delta(k) for k in self.metrics.keys()},
            "features": {k: v for k, v in self.current_features.items() if isinstance(v, (int, float, bool, str))}
        }
    
    def save_state(self, path: Optional[str] = None) -> str:
        """
        Save the current state to a file.
        
        Args:
            path: Path to save the state. If None, use default path.
            
        Returns:
            Path where the state was saved
        """
        if path is None:
            path = os.path.join(self.log_dir, f"segmentation_state_epoch_{self.current_epoch}.json")
        
        # Create a serializable state dictionary
        save_dict = {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "training_phase": self.training_phase,
            "metrics": {k: v for k, v in self.metrics.items()},
            "elapsed_time": time.time() - self.start_time
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        if self.verbose:
            self.logger.info(f"Saved state to {path}")
        
        return path
    
    def load_state(self, path: str) -> bool:
        """
        Load state from a file.
        
        Args:
            path: Path to load the state from
            
        Returns:
            Whether the load was successful
        """
        if not os.path.exists(path):
            self.logger.error(f"Cannot load state: {path} does not exist")
            return False
        
        try:
            with open(path, 'r') as f:
                load_dict = json.load(f)
            
            self.current_epoch = load_dict["current_epoch"]
            self.total_epochs = load_dict["total_epochs"]
            self.training_phase = load_dict["training_phase"]
            self.metrics = load_dict["metrics"]
            
            # Rebuild metric history from metrics
            for key, values in self.metrics.items():
                self.metric_history[key] = deque(values[-self.history_length:], maxlen=self.history_length)
            
            if self.verbose:
                self.logger.info(f"Loaded state from {path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    def reset(self):
        """Reset the state manager for a new episode."""
        self.current_image = None
        self.current_ground_truth = None
        self.current_prediction = None
        self.current_features = {}
        
        if self.verbose:
            self.logger.info("Reset state manager")
