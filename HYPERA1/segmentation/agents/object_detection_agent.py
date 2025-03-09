#!/usr/bin/env python
# Object Detection Agent - Specialized agent for optimizing object-level metrics

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HYPERA1.segmentation.base_segmentation_agent import BaseSegmentationAgent
from HYPERA1.segmentation.segmentation_state_manager import SegmentationStateManager

class ObjectDetectionAgent(BaseSegmentationAgent):
    """
    Specialized agent for optimizing object-level detection metrics in segmentation.
    
    This agent focuses on improving object-level metrics such as object count accuracy,
    object precision/recall, and instance-level IoU. It helps ensure that the segmentation
    model correctly identifies individual objects rather than just pixel-level accuracy.
    
    Attributes:
        name (str): Name of the agent
        state_manager (SegmentationStateManager): Manager for shared state
        device (torch.device): Device to use for computation
        feature_extractor (nn.Module): Neural network for extracting features
        policy_network (nn.Module): Neural network for decision making
        optimizer (torch.optim.Optimizer): Optimizer for policy network
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor for future rewards
        update_frequency (int): Frequency of agent updates
        last_update_step (int): Last step when agent was updated
        action_history (List): History of actions taken by agent
        reward_history (List): History of rewards received by agent
        observation_history (List): History of observations
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        state_manager: SegmentationStateManager,
        device: torch.device = None,
        state_dim: int = 10,
        action_dim: int = 1,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        automatic_entropy_tuning: bool = True,
        update_frequency: int = 1,
        log_dir: str = "logs",
        verbose: bool = False
    ):
        """
        Initialize the object detection agent.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor for future rewards
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            lr: Learning rate
            automatic_entropy_tuning: Whether to automatically tune entropy
            update_frequency: Frequency of agent updates
            log_dir: Directory for saving logs and checkpoints
            verbose: Whether to print verbose output
        """
        # Store feature extraction parameters
        self.feature_channels = 32
        self.hidden_channels = 64
        
        super().__init__(
            name="ObjectDetectionAgent",
            state_manager=state_manager,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            automatic_entropy_tuning=automatic_entropy_tuning,
            update_frequency=update_frequency,
            log_dir=log_dir,
            verbose=verbose
        )
        
        if self.verbose:
            self.logger.info("Initialized ObjectDetectionAgent")
            
    def _initialize_agent(self):
        """
        Initialize the agent's networks and components.
        """
        # Feature extractor for processing segmentation masks
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.feature_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        ).to(self.device)
        
        # Initialize object detection specific components
        self.detection_threshold = 0.5  # Threshold for object detection
        self.min_object_size = 10  # Minimum object size in pixels
        
        if self.verbose:
            self.logger.info("Initialized ObjectDetectionAgent networks and components")
    
    def _extract_features(self, observation):
        """
        Extract features from observation.
        
        Args:
            observation: Dictionary containing observation data
            
        Returns:
            Dictionary of extracted features
        """
        # Extract current segmentation and ground truth from observation
        if isinstance(observation, dict):
            current_seg = observation.get('current_segmentation', None)
            ground_truth = observation.get('ground_truth', None)
        else:
            # If observation is not a dictionary, return empty features
            return {"features": torch.zeros(self.state_dim, device=self.device)}
        
        if current_seg is None or ground_truth is None:
            # Return empty features if segmentation or ground truth is missing
            return {"features": torch.zeros(self.state_dim, device=self.device)}
        
        # Ensure tensors are on the correct device
        if isinstance(current_seg, np.ndarray):
            current_seg = torch.from_numpy(current_seg).float().to(self.device)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth).float().to(self.device)
        
        # Add batch and channel dimensions if needed
        if current_seg.dim() == 2:
            current_seg = current_seg.unsqueeze(0).unsqueeze(0)
        elif current_seg.dim() == 3:
            current_seg = current_seg.unsqueeze(0)
            
        if ground_truth.dim() == 2:
            ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)
        elif ground_truth.dim() == 3:
            ground_truth = ground_truth.unsqueeze(0)
        
        # Extract features using the feature extractor
        with torch.no_grad():
            features = self.feature_extractor(current_seg)
        
        # Ensure we have 2D or 3D images for regionprops
        # Get the first batch and channel for regionprops
        current_seg_np = current_seg[0, 0].cpu().numpy()
        ground_truth_np = ground_truth[0, 0].cpu().numpy()
        
        # Calculate object-level metrics
        try:
            object_metrics = self._calculate_object_metrics(current_seg_np, ground_truth_np)
        except Exception as e:
            # If there's an error in calculating metrics, return default metrics
            print(f"Error calculating object metrics: {e}")
            object_metrics = {
                "count_accuracy": 0.0,
                "size_accuracy": 0.0,
                "shape_accuracy": 0.0,
                "object_iou": 0.0
            }
        
        # Combine features with object metrics
        combined_features = {
            "features": features,
            "object_metrics": object_metrics
        }
        
        return combined_features
    
    def get_state_representation(self, observation):
        """
        Get state representation from observation.
        
        Args:
            observation: Dictionary containing observation data
            
        Returns:
            State representation
        """
        # Extract features from observation
        features = self._extract_features(observation)
        
        # Get the extracted feature tensor
        feature_tensor = features["features"]
        
        # Ensure the feature tensor has the correct shape
        if feature_tensor.dim() == 1:
            # If 1D, reshape to [1, feature_dim]
            feature_tensor = feature_tensor.unsqueeze(0)
        
        # Ensure the second dimension matches state_dim
        if feature_tensor.shape[1] != self.state_dim:
            # Resize to match state_dim
            if feature_tensor.shape[1] > self.state_dim:
                # Truncate if too large
                feature_tensor = feature_tensor[:, :self.state_dim]
            else:
                # Pad with zeros if too small
                padding = torch.zeros(feature_tensor.shape[0], self.state_dim - feature_tensor.shape[1], device=feature_tensor.device)
                feature_tensor = torch.cat([feature_tensor, padding], dim=1)
        
        # Return as tensor for SAC
        return feature_tensor
    
    def apply_action(self, action, observation=None):
        """
        Apply the selected action to modify the segmentation.
        
        Args:
            action: Action to apply
            observation: Optional observation dictionary
            
        Returns:
            Modified segmentation
        """
        # Get current prediction from state manager or observation
        if observation is not None and 'current_segmentation' in observation:
            current_seg = observation['current_segmentation']
        else:
            # Get current prediction from state manager
            current_seg = self.state_manager.get_current_prediction()
        
        if current_seg is None:
            # Return None if no prediction is available
            return None
        
        # Ensure tensor is on the correct device
        current_seg = current_seg.to(self.device)
        
        # Interpret action
        # Action is a tensor with shape [batch_size, action_dim]
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)
            
        # Ensure action has the correct shape
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Extract action components
        # Each component controls a different aspect of object detection refinement
        threshold_adjustment = action[0, 0].item()  # Adjust detection threshold
        
        # Apply threshold adjustment to the segmentation
        # Map action from [-1, 1] to [0.1, 0.9] for threshold
        threshold = 0.5 + (threshold_adjustment * 0.4)
        threshold = max(0.1, min(0.9, threshold))  # Clamp to valid range
        
        # Apply threshold to get binary segmentation
        refined_seg = (current_seg > threshold).float()
        
        return refined_seg
    
    def _save_agent_state(self, path):
        """
        Save agent state to file.
        
        Args:
            path: Path to save agent state
        """
        save_dict = {
            "feature_extractor": self.feature_extractor.state_dict(),
            "detection_threshold": self.detection_threshold,
            "min_object_size": self.min_object_size
        }
        
        torch.save(save_dict, path)
        
        if self.verbose:
            self.logger.info(f"Saved ObjectDetectionAgent state to {path}")
    
    def _load_agent_state(self, path):
        """
        Load agent state from file.
        
        Args:
            path: Path to load agent state from
        """
        if not os.path.exists(path):
            if self.verbose:
                self.logger.warning(f"Agent state file {path} does not exist")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            self.detection_threshold = checkpoint.get("detection_threshold", 0.5)
            self.min_object_size = checkpoint.get("min_object_size", 10)
            
            if self.verbose:
                self.logger.info(f"Loaded ObjectDetectionAgent state from {path}")
            
            return True
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Failed to load agent state: {e}")
            return False
    
    def _reset_agent(self):
        """
        Reset the agent to its initial state.
        """
        # Reset object detection specific parameters
        self.detection_threshold = 0.5
        self.min_object_size = 10
        
        if self.verbose:
            self.logger.info("Reset ObjectDetectionAgent")
    
    def _calculate_object_metrics(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Calculate object-level metrics.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dictionary of object-level metrics
        """
        # Label connected components (objects)
        pred_labels = label(prediction > 0.5)
        gt_labels = label(ground_truth > 0.5)
        
        # Get region properties
        pred_props = regionprops(pred_labels)
        gt_props = regionprops(gt_labels)
        
        # Count objects
        pred_count = len(pred_props)
        gt_count = len(gt_props)
        
        # Calculate object count accuracy
        count_diff = abs(pred_count - gt_count)
        max_count = max(pred_count, gt_count, 1)  # Avoid division by zero
        count_accuracy = 1.0 - (count_diff / max_count)
        
        # Calculate object-level IoU
        if pred_count > 0 and gt_count > 0:
            # Create IoU matrix
            iou_matrix = np.zeros((gt_count, pred_count))
            
            for i, gt_prop in enumerate(gt_props):
                gt_mask = gt_labels == gt_prop.label
                
                for j, pred_prop in enumerate(pred_props):
                    pred_mask = pred_labels == pred_prop.label
                    
                    # Calculate IoU
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    iou_matrix[i, j] = iou
            
            # Use Hungarian algorithm to find optimal assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Negative for maximization
            
            # Calculate mean IoU for matched objects
            matched_ious = iou_matrix[row_ind, col_ind]
            mean_iou = matched_ious.mean() if len(matched_ious) > 0 else 0
            
            # Calculate precision and recall
            matches = (matched_ious > 0.5).sum()
            precision = matches / pred_count if pred_count > 0 else 0
            recall = matches / gt_count if gt_count > 0 else 0
            
            # Calculate F1 score
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            mean_iou = 0
            precision = 0
            recall = 0
            f1_score = 0
        
        # Create metrics dictionary
        object_metrics = {
            "pred_count": pred_count,
            "gt_count": gt_count,
            "count_accuracy": count_accuracy,
            "mean_iou": mean_iou,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        return object_metrics
