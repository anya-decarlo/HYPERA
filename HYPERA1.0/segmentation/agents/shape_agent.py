#!/usr/bin/env python
# Shape Agent - Specialized agent for optimizing shape regularization

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

class ShapeAgent(BaseSegmentationAgent):
    """
    Specialized agent for optimizing shape regularization in segmentation.
    
    This agent focuses on improving the shape regularization component of the reward function,
    which encourages biologically realistic shapes by penalizing irregular or unrealistic shapes.
    
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
        Initialize the shape agent.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            verbose: Whether to print verbose output
        """
        super().__init__(
            name="ShapeAgent",
            state_manager=state_manager,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=verbose
        )
        
        # Initialize shape-specific components
        self._init_shape_components()
        
        if self.verbose:
            self.logger.info("Initialized ShapeAgent")
    
    def _init_shape_components(self):
        """
        Initialize shape-specific components.
        """
        # Feature extractor for shape features
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
        
        # Policy network for shape-specific actions
        self.policy_network = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions: increase, decrease, or maintain shape regularization
        ).to(self.device)
        
        # Optimizer for policy network
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate
        )
        
        # Shape-specific parameters
        self.shape_regularization_strength = 0.5  # Initial shape regularization strength
        self.min_strength = 0.1
        self.max_strength = 0.9
        self.strength_step = 0.1
        
        # Shape priors for different cell types
        self.shape_priors = {
            "circular": {
                "circularity_target": 1.0,
                "circularity_tolerance": 0.2
            },
            "elongated": {
                "aspect_ratio_target": 3.0,
                "aspect_ratio_tolerance": 0.5
            },
            "irregular": {
                "solidity_target": 0.7,
                "solidity_tolerance": 0.1
            }
        }
        
        # Default shape prior
        self.current_shape_prior = "circular"
    
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
        
        # Calculate shape metrics
        shape_metrics = self._calculate_shape_metrics(current_prediction)
        
        # Get recent metrics from state manager
        recent_metrics = self.state_manager.get_recent_metrics()
        
        # Create observation dictionary
        observation = {
            "current_image": current_image,
            "current_mask": current_mask,
            "current_prediction": current_prediction,
            "shape_metrics": shape_metrics,
            "shape_regularization_strength": self.shape_regularization_strength,
            "current_shape_prior": self.current_shape_prior,
            "recent_metrics": recent_metrics
        }
        
        # Store observation in history
        self.observation_history.append(observation)
        
        return observation
    
    def _calculate_shape_metrics(self, prediction: torch.Tensor) -> Dict[str, float]:
        """
        Calculate shape metrics for the predicted mask.
        
        Args:
            prediction: Predicted segmentation mask
            
        Returns:
            Dictionary of shape metrics
        """
        # Convert tensor to numpy array
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        
        # Label connected components
        from skimage.measure import label, regionprops
        pred_labels = label(prediction > 0.5)
        
        # Get region properties
        regions = regionprops(pred_labels)
        
        if not regions:
            return {
                "circularity": 0.0,
                "aspect_ratio": 0.0,
                "solidity": 0.0,
                "num_objects": 0
            }
        
        # Calculate shape metrics for each region
        circularities = []
        aspect_ratios = []
        solidities = []
        
        for region in regions:
            # Calculate perimeter and area
            perimeter = region.perimeter
            area = region.area
            
            if area > 0 and perimeter > 0:
                # Calculate circularity (1.0 for perfect circle)
                circularity = (4 * np.pi * area) / (perimeter**2)
                circularities.append(circularity)
                
                # Calculate aspect ratio
                if hasattr(region, 'major_axis_length') and hasattr(region, 'minor_axis_length'):
                    if region.minor_axis_length > 0:
                        aspect_ratio = region.major_axis_length / region.minor_axis_length
                        aspect_ratios.append(aspect_ratio)
                
                # Calculate solidity (convex hull area ratio)
                if hasattr(region, 'solidity'):
                    solidity = region.solidity
                    solidities.append(solidity)
        
        # Calculate average metrics
        avg_circularity = np.mean(circularities) if circularities else 0.0
        avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0.0
        avg_solidity = np.mean(solidities) if solidities else 0.0
        
        # Create metrics dictionary
        shape_metrics = {
            "circularity": avg_circularity,
            "aspect_ratio": avg_aspect_ratio,
            "solidity": avg_solidity,
            "num_objects": len(regions)
        }
        
        return shape_metrics
    
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
        
        # Map action to shape regularization strength adjustment
        if action == 0:
            # Increase strength
            new_strength = min(self.shape_regularization_strength + self.strength_step, self.max_strength)
        elif action == 1:
            # Decrease strength
            new_strength = max(self.shape_regularization_strength - self.strength_step, self.min_strength)
        else:
            # Maintain current strength
            new_strength = self.shape_regularization_strength
        
        # Determine best shape prior based on shape metrics
        shape_metrics = observation["shape_metrics"]
        new_shape_prior = self._select_shape_prior(shape_metrics)
        
        # Create action dictionary
        action_dict = {
            "shape_regularization_strength": new_strength,
            "shape_prior": new_shape_prior,
            "action_type": action,
            "action_probs": action_probs.detach().cpu().numpy()
        }
        
        # Store action in history
        self.action_history.append(action_dict)
        
        # Update shape regularization strength and shape prior
        self.shape_regularization_strength = new_strength
        self.current_shape_prior = new_shape_prior
        
        return action_dict
    
    def _select_shape_prior(self, shape_metrics: Dict[str, float]) -> str:
        """
        Select the best shape prior based on shape metrics.
        
        Args:
            shape_metrics: Dictionary of shape metrics
            
        Returns:
            Selected shape prior
        """
        # Extract shape metrics
        circularity = shape_metrics["circularity"]
        aspect_ratio = shape_metrics["aspect_ratio"]
        solidity = shape_metrics["solidity"]
        
        # Calculate distance to each shape prior
        circular_distance = abs(circularity - self.shape_priors["circular"]["circularity_target"])
        elongated_distance = abs(aspect_ratio - self.shape_priors["elongated"]["aspect_ratio_target"])
        irregular_distance = abs(solidity - self.shape_priors["irregular"]["solidity_target"])
        
        # Normalize distances by tolerance
        circular_distance /= self.shape_priors["circular"]["circularity_tolerance"]
        elongated_distance /= self.shape_priors["elongated"]["aspect_ratio_tolerance"]
        irregular_distance /= self.shape_priors["irregular"]["solidity_tolerance"]
        
        # Select shape prior with minimum normalized distance
        distances = {
            "circular": circular_distance,
            "elongated": elongated_distance,
            "irregular": irregular_distance
        }
        
        return min(distances, key=distances.get)
    
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
            "shape_regularization_strength": self.shape_regularization_strength,
            "shape_prior": self.current_shape_prior
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
            "shape_regularization_strength": self.shape_regularization_strength,
            "current_shape_prior": self.current_shape_prior,
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
        if "shape_regularization_strength" in state:
            self.shape_regularization_strength = state["shape_regularization_strength"]
        
        if "current_shape_prior" in state:
            self.current_shape_prior = state["current_shape_prior"]
        
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
            "shape_regularization_strength": self.shape_regularization_strength,
            "current_shape_prior": self.current_shape_prior,
            "shape_priors": self.shape_priors,
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
            self.logger.info(f"Saved ShapeAgent to {path}")
    
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
        self.shape_regularization_strength = state_dict["shape_regularization_strength"]
        self.current_shape_prior = state_dict["current_shape_prior"]
        self.shape_priors = state_dict["shape_priors"]
        self.learning_rate = state_dict["learning_rate"]
        self.gamma = state_dict["gamma"]
        self.update_frequency = state_dict["update_frequency"]
        self.last_update_step = state_dict["last_update_step"]
        self.action_history = state_dict["action_history"]
        self.reward_history = state_dict["reward_history"]
        self.training = state_dict["training"]
        
        if self.verbose:
            self.logger.info(f"Loaded ShapeAgent from {path}")
