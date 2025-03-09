#!/usr/bin/env python
# Boundary Agent - Specialized agent for optimizing boundary accuracy

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HYPERA1.segmentation.base_segmentation_agent import BaseSegmentationAgent
from HYPERA1.segmentation.segmentation_state_manager import SegmentationStateManager

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
        Initialize the boundary agent.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            lr: Learning rate
            automatic_entropy_tuning: Whether to automatically tune entropy
            update_frequency: How often the agent should update (in steps)
            log_dir: Directory for saving logs and checkpoints
            verbose: Whether to print verbose output
        """
        super().__init__(
            name="BoundaryAgent",
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
            lr=self.lr
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
    
    def _initialize_agent(self):
        """
        Initialize agent-specific parameters and networks.
        """
        # Use existing _init_boundary_components method
        self._init_boundary_components()
    
    def _extract_features(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Extract boundary-specific features from the observation.
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary of extracted features
        """
        # Get current state from state manager
        current_image = self.state_manager.get_current_image()
        current_mask = self.state_manager.get_current_mask()
        current_prediction = self.state_manager.get_current_prediction()
        
        if current_image is None or current_mask is None or current_prediction is None:
            # If any required state is missing, return empty features
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
        
        # Create features dictionary
        features = {
            "current_image": current_image,
            "gt_boundary": gt_boundary,
            "pred_boundary": pred_boundary,
            "hausdorff_distance": hausdorff_distance,
            "boundary_dice": boundary_dice,
            "boundary_sensitivity": self.boundary_sensitivity
        }
        
        return features
    
    def get_state_representation(self, observation: torch.Tensor) -> np.ndarray:
        """
        Extract a state representation from the observation.
        
        Args:
            observation: The current observation
            
        Returns:
            State representation as a numpy array
        """
        # Get features
        features = self._extract_features(observation)
        
        if not features:
            # Return zero state if features are empty
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Extract boundary features using feature extractor
        current_image = features["current_image"]
        pred_boundary = features["pred_boundary"]
        
        # Combine image and predicted boundary for feature extraction
        combined_input = torch.cat([
            current_image.unsqueeze(1),
            pred_boundary.unsqueeze(1)
        ], dim=1)
        
        # Extract features using the feature extractor
        with torch.no_grad():
            boundary_features = self.feature_extractor(combined_input)
            boundary_features = boundary_features.view(-1).cpu().numpy()
        
        # Add scalar metrics to the state
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Fill in the state with boundary features (truncate or pad as needed)
        feature_size = min(len(boundary_features), self.state_dim - 2)
        state[:feature_size] = boundary_features[:feature_size]
        
        # Add scalar metrics to the state
        if "hausdorff_distance" in features:
            state[-2] = features["hausdorff_distance"]
        if "boundary_dice" in features:
            state[-1] = features["boundary_dice"]
        
        return state
    
    def apply_action(self, action: np.ndarray, features: Dict[str, Any]) -> torch.Tensor:
        """
        Apply the action to produce a segmentation decision.
        
        Args:
            action: Action from the SAC policy
            features: Dictionary of extracted features
            
        Returns:
            Segmentation decision tensor
        """
        if not features:
            # Return None if features are empty
            return None
        
        # Get current prediction from state manager
        current_prediction = self.state_manager.get_current_prediction()
        if current_prediction is None:
            return None
        
        # Ensure prediction is on the correct device
        current_prediction = current_prediction.to(self.device)
        
        # Interpret the action
        # Action is a continuous value that we map to boundary sensitivity
        # We map the action range (-1, 1) to a sensitivity adjustment
        sensitivity_adjustment = action[0] * self.sensitivity_step
        
        # Update boundary sensitivity
        new_sensitivity = self.boundary_sensitivity + sensitivity_adjustment
        new_sensitivity = max(self.min_sensitivity, min(self.max_sensitivity, new_sensitivity))
        self.boundary_sensitivity = new_sensitivity
        
        # Re-extract boundaries with the new sensitivity
        pred_boundary = self._extract_boundary(current_prediction)
        
        # Apply boundary refinement to the current prediction
        refined_prediction = self._refine_prediction(current_prediction, pred_boundary)
        
        return refined_prediction
    
    def _refine_prediction(self, prediction: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """
        Refine the prediction using boundary information.
        
        Args:
            prediction: Current prediction tensor
            boundary: Extracted boundary tensor
            
        Returns:
            Refined prediction tensor
        """
        # Simple refinement: dilate the boundary and use it to refine the prediction
        # This is a placeholder implementation
        kernel_size = 3
        padding = kernel_size // 2
        
        # Dilate the boundary
        dilated_boundary = F.max_pool2d(
            boundary.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).squeeze(1)
        
        # Use the dilated boundary to refine the prediction
        # This is a simple approach that can be improved
        refined_prediction = prediction.clone()
        
        # Where the boundary is active, make the prediction more decisive
        # (closer to 0 or 1)
        boundary_mask = dilated_boundary > 0.5
        refined_prediction[boundary_mask] = torch.where(
            refined_prediction[boundary_mask] > 0.5,
            torch.ones_like(refined_prediction[boundary_mask]),
            torch.zeros_like(refined_prediction[boundary_mask])
        )
        
        return refined_prediction
    
    def _save_agent_state(self, save_dict: Dict[str, Any]):
        """
        Add agent-specific state to the save dictionary.
        
        Args:
            save_dict: Dictionary to add agent-specific state to
        """
        # Save boundary-specific parameters
        save_dict["boundary_sensitivity"] = self.boundary_sensitivity
        save_dict["min_sensitivity"] = self.min_sensitivity
        save_dict["max_sensitivity"] = self.max_sensitivity
        save_dict["sensitivity_step"] = self.sensitivity_step
        
        # Save network states
        save_dict["feature_extractor_state"] = self.feature_extractor.state_dict()
        save_dict["policy_network_state"] = self.policy_network.state_dict()
    
    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Load agent-specific state from the checkpoint.
        
        Args:
            checkpoint: Dictionary containing the agent state
            
        Returns:
            Whether the load was successful
        """
        try:
            # Load boundary-specific parameters
            self.boundary_sensitivity = checkpoint.get("boundary_sensitivity", self.boundary_sensitivity)
            self.min_sensitivity = checkpoint.get("min_sensitivity", self.min_sensitivity)
            self.max_sensitivity = checkpoint.get("max_sensitivity", self.max_sensitivity)
            self.sensitivity_step = checkpoint.get("sensitivity_step", self.sensitivity_step)
            
            # Load network states
            if "feature_extractor_state" in checkpoint:
                self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state"])
            
            if "policy_network_state" in checkpoint:
                self.policy_network.load_state_dict(checkpoint["policy_network_state"])
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load agent-specific state: {e}")
            return False
    
    def _reset_agent(self):
        """
        Reset agent-specific episode state.
        """
        # Reset boundary sensitivity to initial value
        self.boundary_sensitivity = 0.5
        
        # Clear any cached data
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
