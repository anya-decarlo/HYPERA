#!/usr/bin/env python
# Boundary Agent - Specialized agent for optimizing boundary accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

from ..base_segmentation_agent import BaseSegmentationAgent
from ..segmentation_state_manager import SegmentationStateManager

class BoundaryAgent(BaseSegmentationAgent):
    """
    Specialized agent for optimizing boundary accuracy in segmentation.
    
    This agent focuses on improving the boundary accuracy component of the reward function,
    which is measured by the Hausdorff distance between predicted and ground truth boundaries.
    
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
        update_frequency: int = 5,
        verbose: bool = False
    ):
        """
        Initialize the boundary agent.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            verbose: Whether to print verbose output
        """
        super().__init__(
            name="BoundaryAgent",
            state_manager=state_manager,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=verbose
        )
        
        # Initialize boundary-specific components
        self._init_boundary_components()
        
        if self.verbose:
            self.logger.info("Initialized BoundaryAgent")
    
    def _init_boundary_components(self):
        """
        Initialize boundary-specific components.
        """
        # Feature extractor for boundary features
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
        
        # Policy network for boundary-specific actions
        self.policy_network = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions: increase, decrease, or maintain boundary sensitivity
        ).to(self.device)
        
        # Optimizer for policy network
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate
        )
        
        # Boundary-specific parameters
        self.boundary_sensitivity = 0.5  # Initial boundary sensitivity
        self.min_sensitivity = 0.1
        self.max_sensitivity = 0.9
        self.sensitivity_step = 0.1
        
        # Edge detection kernel for boundary extraction
        self.edge_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
    
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
        
        # Extract boundaries from mask and prediction
        gt_boundary = self._extract_boundary(current_mask)
        pred_boundary = self._extract_boundary(current_prediction)
        
        # Calculate boundary accuracy metrics
        hausdorff_distance = self._calculate_hausdorff_distance(pred_boundary, gt_boundary)
        boundary_dice = self._calculate_boundary_dice(pred_boundary, gt_boundary)
        
        # Get recent metrics from state manager
        recent_metrics = self.state_manager.get_recent_metrics()
        
        # Create observation dictionary
        observation = {
            "current_image": current_image,
            "gt_boundary": gt_boundary,
            "pred_boundary": pred_boundary,
            "hausdorff_distance": hausdorff_distance,
            "boundary_dice": boundary_dice,
            "boundary_sensitivity": self.boundary_sensitivity,
            "recent_metrics": recent_metrics
        }
        
        # Store observation in history
        self.observation_history.append(observation)
        
        return observation
    
    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary from a mask using edge detection.
        
        Args:
            mask: Binary mask tensor
            
        Returns:
            Boundary tensor
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Apply edge detection kernel
        boundary = F.conv2d(
            binary_mask.unsqueeze(1),
            self.edge_kernel,
            padding=1
        )
        
        # Threshold to get binary boundary
        boundary = (boundary.abs() > self.boundary_sensitivity).float()
        
        return boundary.squeeze(1)
    
    def _calculate_hausdorff_distance(self, pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> float:
        """
        Calculate Hausdorff distance between predicted and ground truth boundaries.
        
        Args:
            pred_boundary: Predicted boundary tensor
            gt_boundary: Ground truth boundary tensor
            
        Returns:
            Hausdorff distance
        """
        # Convert tensors to numpy arrays
        if isinstance(pred_boundary, torch.Tensor):
            pred_boundary = pred_boundary.detach().cpu().numpy()
        if isinstance(gt_boundary, torch.Tensor):
            gt_boundary = gt_boundary.detach().cpu().numpy()
        
        # If either boundary is empty, return maximum distance
        if not np.any(pred_boundary) or not np.any(gt_boundary):
            return 100.0  # Large value for empty boundaries
        
        # Get boundary pixel coordinates
        pred_points = np.argwhere(pred_boundary > 0.5)
        gt_points = np.argwhere(gt_boundary > 0.5)
        
        # Calculate distances from each pred point to closest gt point
        pred_to_gt = np.array([np.min(np.linalg.norm(p - gt_points, axis=1)) for p in pred_points])
        
        # Calculate distances from each gt point to closest pred point
        gt_to_pred = np.array([np.min(np.linalg.norm(g - pred_points, axis=1)) for g in gt_points])
        
        # Hausdorff distance is the maximum of the two directed distances
        hausdorff = max(np.max(pred_to_gt), np.max(gt_to_pred))
        
        return hausdorff
    
    def _calculate_boundary_dice(self, pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> float:
        """
        Calculate Dice coefficient between predicted and ground truth boundaries.
        
        Args:
            pred_boundary: Predicted boundary tensor
            gt_boundary: Ground truth boundary tensor
            
        Returns:
            Boundary Dice coefficient
        """
        # Ensure tensors are on the correct device
        pred_boundary = pred_boundary.to(self.device)
        gt_boundary = gt_boundary.to(self.device)
        
        # Calculate intersection and union
        intersection = torch.sum(pred_boundary * gt_boundary)
        pred_sum = torch.sum(pred_boundary)
        gt_sum = torch.sum(gt_boundary)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-6)
        
        # Convert to scalar if it's a tensor
        if isinstance(dice, torch.Tensor):
            dice = dice.item()
        
        return dice
    
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
        
        # Get action logits from policy network
        action_logits = self.policy_network(features)
        
        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Sample action from probabilities
        if self.training:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = torch.argmax(action_probs, dim=1).item()
        
        # Map action to boundary sensitivity adjustment
        if action == 0:
            # Increase sensitivity
            new_sensitivity = min(self.boundary_sensitivity + self.sensitivity_step, self.max_sensitivity)
        elif action == 1:
            # Decrease sensitivity
            new_sensitivity = max(self.boundary_sensitivity - self.sensitivity_step, self.min_sensitivity)
        else:
            # Maintain current sensitivity
            new_sensitivity = self.boundary_sensitivity
        
        # Create action dictionary
        action_dict = {
            "boundary_sensitivity": new_sensitivity,
            "action_type": action,
            "action_probs": action_probs.detach().cpu().numpy()
        }
        
        # Store action in history
        self.action_history.append(action_dict)
        
        # Update boundary sensitivity
        self.boundary_sensitivity = new_sensitivity
        
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
        
        # Get action logits from policy network
        action_logits = self.policy_network(features)
        
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
            "boundary_sensitivity": self.boundary_sensitivity
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
            "boundary_sensitivity": self.boundary_sensitivity,
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
        if "boundary_sensitivity" in state:
            self.boundary_sensitivity = state["boundary_sensitivity"]
        
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
            "boundary_sensitivity": self.boundary_sensitivity,
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
            self.logger.info(f"Saved BoundaryAgent to {path}")
    
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
        self.boundary_sensitivity = state_dict["boundary_sensitivity"]
        self.learning_rate = state_dict["learning_rate"]
        self.gamma = state_dict["gamma"]
        self.update_frequency = state_dict["update_frequency"]
        self.last_update_step = state_dict["last_update_step"]
        self.action_history = state_dict["action_history"]
        self.reward_history = state_dict["reward_history"]
        self.training = state_dict["training"]
        
        if self.verbose:
            self.logger.info(f"Loaded BoundaryAgent from {path}")
