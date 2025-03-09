#!/usr/bin/env python
# Object Detection Agent - Specialized agent for optimizing object-level metrics

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment

from ..base_segmentation_agent import BaseSegmentationAgent
from ..segmentation_state_manager import SegmentationStateManager

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
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the object detection agent.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            verbose: Whether to print verbose output
        """
        super().__init__(
            name="ObjectDetectionAgent",
            state_manager=state_manager,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=verbose
        )
        
        # Initialize object detection-specific components
        self._init_object_detection_components()
        
        if self.verbose:
            self.logger.info("Initialized ObjectDetectionAgent")
    
    def _init_object_detection_components(self):
        """
        Initialize object detection-specific components.
        """
        # Feature extractor for object detection features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        ).to(self.device)
        
        # Policy network for object detection-specific actions
        self.policy_network = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 4, 256),  # +4 for object metrics
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions: increase, decrease, or maintain object sensitivity
        ).to(self.device)
        
        # Optimizer for policy network
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate
        )
        
        # Object detection-specific parameters
        self.min_object_size = 10  # Minimum object size in pixels
        self.object_sensitivity = 0.5  # Sensitivity for object detection (threshold)
        self.min_sensitivity = 0.1
        self.max_sensitivity = 0.9
        self.sensitivity_step = 0.05
        
        # Object detection history
        self.object_count_history = []
        self.object_iou_history = []
    
    def observe(self) -> Dict[str, Any]:
        """
        Observe the current state.
        
        Returns:
            Dictionary of observations
        """
        # Get current state from state manager
        current_image = self.state_manager.get_current_image()
        current_mask = self.state_manager.get_current_mask()
        current_prediction = self.state_manager.get_current_prediction()
        
        if current_image is None or current_mask is None or current_prediction is None:
            # If any required state is missing, return empty observation
            return {}
        
        # Ensure tensors are on the correct device
        current_image = current_image.to(self.device)
        current_mask = current_mask.to(self.device)
        current_prediction = current_prediction.to(self.device)
        
        # Calculate object-level metrics
        object_metrics = self._calculate_object_metrics(current_prediction, current_mask)
        
        # Get recent metrics from state manager
        recent_metrics = self.state_manager.get_recent_metrics()
        
        # Create observation dictionary
        observation = {
            "current_image": current_image,
            "current_mask": current_mask,
            "current_prediction": current_prediction,
            "object_metrics": object_metrics,
            "recent_metrics": recent_metrics,
            "object_sensitivity": self.object_sensitivity
        }
        
        # Store observation in history
        self.observation_history.append(observation)
        
        return observation
    
    def _calculate_object_metrics(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        Calculate object-level metrics.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dictionary of object-level metrics
        """
        # Convert tensors to numpy arrays
        pred_np = (prediction > self.object_sensitivity).float().cpu().numpy()
        gt_np = (ground_truth > 0.5).float().cpu().numpy()
        
        # Remove batch dimension if present
        if pred_np.ndim > 3:
            pred_np = pred_np[0]
        if gt_np.ndim > 3:
            gt_np = gt_np[0]
        
        # Remove channel dimension if present
        if pred_np.ndim > 2:
            pred_np = pred_np[0]
        if gt_np.ndim > 2:
            gt_np = gt_np[0]
        
        # Label connected components (objects)
        pred_labels = label(pred_np > 0.5)
        gt_labels = label(gt_np > 0.5)
        
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
        
        # Update history
        self.object_count_history.append((pred_count, gt_count))
        self.object_iou_history.append(mean_iou)
        
        return object_metrics
    
    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on an action based on the observation.
        
        Args:
            observation: Dictionary of observations
            
        Returns:
            Dictionary of actions
        """
        if not observation:
            # If observation is empty, return no action
            return {}
        
        # Extract features from current image
        current_image = observation["current_image"]
        
        # Ensure image has channel dimension
        if len(current_image.shape) == 3:
            current_image = current_image.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(current_image)
        features = features.view(features.size(0), -1)
        
        # Add object metrics to features
        object_metrics = observation["object_metrics"]
        
        # Create object metrics tensor
        metrics_tensor = torch.tensor([
            [
                object_metrics["count_accuracy"],
                object_metrics["mean_iou"],
                object_metrics["precision"],
                object_metrics["recall"]
            ]
        ], device=self.device)
        
        # Concatenate features with object metrics
        extended_features = torch.cat([features, metrics_tensor], dim=1)
        
        # Get action logits from policy network
        action_logits = self.policy_network(extended_features)
        
        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Sample action from probabilities
        if self.training:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = torch.argmax(action_probs, dim=1).item()
        
        # Map action to object sensitivity adjustment
        if action == 0:
            # Increase sensitivity (decreases threshold)
            new_sensitivity = max(self.object_sensitivity - self.sensitivity_step, self.min_sensitivity)
        elif action == 1:
            # Decrease sensitivity (increases threshold)
            new_sensitivity = min(self.object_sensitivity + self.sensitivity_step, self.max_sensitivity)
        else:
            # Maintain current sensitivity
            new_sensitivity = self.object_sensitivity
        
        # Create action dictionary
        action_dict = {
            "object_sensitivity": new_sensitivity,
            "action_type": action,
            "action_probs": action_probs.detach().cpu().numpy(),
            "current_sensitivity": self.object_sensitivity,
            "object_metrics": object_metrics
        }
        
        # Store action in history
        self.action_history.append(action_dict)
        
        # Update object sensitivity
        self.object_sensitivity = new_sensitivity
        
        return action_dict
    
    def learn(self, reward: float) -> Dict[str, float]:
        """
        Learn from the reward.
        
        Args:
            reward: Reward value
            
        Returns:
            Dictionary of learning metrics
        """
        # Store reward in history
        self.reward_history.append(reward)
        
        # Check if it's time to update
        current_step = self.state_manager.get_current_step()
        if current_step - self.last_update_step < self.update_frequency:
            return {}
        
        # Update last update step
        self.last_update_step = current_step
        
        # Check if we have enough history for learning
        if len(self.action_history) < 2 or len(self.reward_history) < 2:
            return {}
        
        # Get the most recent observation, action, and reward
        observation = self.observation_history[-1]
        action = self.action_history[-1]
        
        # Extract features from current image
        current_image = observation["current_image"]
        
        # Ensure image has channel dimension
        if len(current_image.shape) == 3:
            current_image = current_image.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(current_image)
        features = features.view(features.size(0), -1)
        
        # Add object metrics to features
        object_metrics = observation["object_metrics"]
        
        # Create object metrics tensor
        metrics_tensor = torch.tensor([
            [
                object_metrics["count_accuracy"],
                object_metrics["mean_iou"],
                object_metrics["precision"],
                object_metrics["recall"]
            ]
        ], device=self.device)
        
        # Concatenate features with object metrics
        extended_features = torch.cat([features, metrics_tensor], dim=1)
        
        # Get action logits from policy network
        action_logits = self.policy_network(extended_features)
        
        # Calculate policy loss using REINFORCE algorithm
        action_type = action["action_type"]
        policy_loss = -action_logits[0, action_type] * reward
        
        # Backpropagate and optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Create learning metrics dictionary
        metrics = {
            "policy_loss": policy_loss.item(),
            "reward": reward,
            "object_sensitivity": self.object_sensitivity,
            "count_accuracy": object_metrics["count_accuracy"],
            "mean_iou": object_metrics["mean_iou"],
            "precision": object_metrics["precision"],
            "recall": object_metrics["recall"],
            "f1_score": object_metrics["f1_score"]
        }
        
        return metrics
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary of agent state
        """
        return {
            "name": self.name,
            "object_sensitivity": self.object_sensitivity,
            "min_object_size": self.min_object_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "update_frequency": self.update_frequency,
            "last_update_step": self.last_update_step,
            "action_history_length": len(self.action_history),
            "reward_history_length": len(self.reward_history),
            "observation_history_length": len(self.observation_history),
            "training": self.training
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the agent.
        
        Args:
            state: Dictionary of agent state
        """
        if "object_sensitivity" in state:
            self.object_sensitivity = state["object_sensitivity"]
        
        if "min_object_size" in state:
            self.min_object_size = state["min_object_size"]
        
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
            # Update optimizer with new learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        
        if "gamma" in state:
            self.gamma = state["gamma"]
        
        if "update_frequency" in state:
            self.update_frequency = state["update_frequency"]
        
        if "training" in state:
            self.training = state["training"]
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        # Create state dictionary
        state_dict = {
            "feature_extractor": self.feature_extractor.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "object_sensitivity": self.object_sensitivity,
            "min_object_size": self.min_object_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "update_frequency": self.update_frequency,
            "last_update_step": self.last_update_step,
            "action_history": self.action_history,
            "reward_history": self.reward_history,
            "training": self.training
        }
        
        # Save state dictionary
        torch.save(state_dict, path)
        
        if self.verbose:
            self.logger.info(f"Saved ObjectDetectionAgent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        # Load state dictionary
        state_dict = torch.load(path, map_location=self.device)
        
        # Load model parameters
        self.feature_extractor.load_state_dict(state_dict["feature_extractor"])
        self.policy_network.load_state_dict(state_dict["policy_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        
        # Load agent parameters
        self.object_sensitivity = state_dict["object_sensitivity"]
        self.min_object_size = state_dict["min_object_size"]
        self.learning_rate = state_dict["learning_rate"]
        self.gamma = state_dict["gamma"]
        self.update_frequency = state_dict["update_frequency"]
        self.last_update_step = state_dict["last_update_step"]
        self.action_history = state_dict["action_history"]
        self.reward_history = state_dict["reward_history"]
        self.training = state_dict["training"]
        
        if self.verbose:
            self.logger.info(f"Loaded ObjectDetectionAgent from {path}")
