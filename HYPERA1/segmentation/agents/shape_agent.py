#!/usr/bin/env python
# Shape Agent - Focuses on optimizing shape regularization

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HYPERA1.segmentation.base_segmentation_agent import BaseSegmentationAgent
from HYPERA1.segmentation.segmentation_state_manager import SegmentationStateManager

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
        Initialize the shape agent.
        
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
            name="ShapeAgent",
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
            lr=self.lr
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
    
    def _initialize_agent(self):
        """
        Initialize agent-specific parameters and networks.
        """
        # This method is called by the BaseSegmentationAgent constructor
        # Shape-specific initialization is done in _init_shape_components
        pass
    
    def _extract_features(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Extract shape-specific features from the observation.
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary of extracted features
        """
        if not isinstance(observation, dict):
            return {}
        
        # Extract current prediction from observation
        current_prediction = observation.get("current_prediction", None)
        if current_prediction is None:
            return {}
        
        # Ensure prediction is a tensor
        if not isinstance(current_prediction, torch.Tensor):
            current_prediction = torch.tensor(current_prediction, device=self.device)
        
        # Ensure prediction is on the correct device
        current_prediction = current_prediction.to(self.device)
        
        # Add batch dimension if needed
        if len(current_prediction.shape) == 2:
            current_prediction = current_prediction.unsqueeze(0)
        
        # Add channel dimension if needed
        if len(current_prediction.shape) == 3:
            current_prediction = current_prediction.unsqueeze(1)
        
        # Extract features using the feature extractor
        shape_features = self.feature_extractor(current_prediction.float())
        
        # Calculate shape metrics
        shape_metrics = observation.get("shape_metrics", {})
        
        return {
            "shape_features": shape_features,
            "shape_metrics": shape_metrics,
            "current_prediction": current_prediction
        }
    
    def get_state_representation(self, observation: torch.Tensor) -> np.ndarray:
        """
        Extract a state representation from the observation.
        
        Args:
            observation: The current observation
            
        Returns:
            State representation as a numpy array
        """
        # Extract features from observation
        features = self._extract_features(observation)
        
        if not features:
            # Return zero state if features extraction failed
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Get shape features
        shape_features = features.get("shape_features", None)
        if shape_features is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Flatten shape features
        shape_features_flat = shape_features.view(-1).detach().cpu().numpy()
        
        # Get shape metrics
        shape_metrics = features.get("shape_metrics", {})
        
        # Create state representation
        # Use a subset of features to match state_dim
        state_representation = np.zeros(self.state_dim, dtype=np.float32)
        
        # Fill state representation with shape features and metrics
        feature_count = min(self.state_dim - 3, len(shape_features_flat))
        state_representation[:feature_count] = shape_features_flat[:feature_count]
        
        # Add shape regularization strength
        state_representation[self.state_dim - 3] = self.shape_regularization_strength
        
        # Add shape prior indicators (one-hot encoding)
        if self.current_shape_prior == "circular":
            state_representation[self.state_dim - 2] = 1.0
        elif self.current_shape_prior == "elongated":
            state_representation[self.state_dim - 1] = 1.0
        
        return state_representation
    
    def apply_action(self, action: np.ndarray, features: Dict[str, Any]) -> torch.Tensor:
        """
        Apply the action to produce a segmentation decision.
        
        Args:
            action: Action from the SAC policy
            features: Dictionary of extracted features
            
        Returns:
            Segmentation decision tensor
        """
        # Get current prediction from features
        current_prediction = features.get("current_prediction", None)
        if current_prediction is None:
            return None
        
        # Scale action to shape regularization strength adjustment
        action_scaled = action[0] * self.strength_step
        
        # Update shape regularization strength
        new_strength = self.shape_regularization_strength + action_scaled
        self.shape_regularization_strength = max(self.min_strength, min(self.max_strength, new_strength))
        
        # Apply shape regularization to refine the prediction
        refined_prediction = self._apply_shape_regularization(current_prediction)
        
        return refined_prediction
    
    def _apply_shape_regularization(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Apply shape regularization to refine the prediction.
        
        Args:
            prediction: Current prediction tensor
            
        Returns:
            Refined prediction tensor
        """
        # Convert to numpy for morphological operations
        pred_np = prediction.detach().cpu().numpy()
        
        # Apply morphological operations based on shape prior and regularization strength
        from skimage import morphology
        
        # Determine structuring element size based on regularization strength
        struct_size = max(1, int(3 * self.shape_regularization_strength))
        
        if self.current_shape_prior == "circular":
            # Use disk structuring element for circular shapes
            selem = morphology.disk(struct_size)
            # Apply opening to remove small protrusions
            pred_np = morphology.opening(pred_np > 0.5, selem)
            # Apply closing to fill small holes
            pred_np = morphology.closing(pred_np, selem)
        
        elif self.current_shape_prior == "elongated":
            # Use elliptical structuring element for elongated shapes
            selem = morphology.ellipse(struct_size, 2 * struct_size)
            # Apply opening and closing
            pred_np = morphology.opening(pred_np > 0.5, selem)
            pred_np = morphology.closing(pred_np, selem)
        
        else:  # irregular
            # Use smaller structuring element for irregular shapes
            selem = morphology.square(struct_size)
            # Apply less aggressive morphological operations
            pred_np = morphology.opening(pred_np > 0.5, selem)
            pred_np = morphology.closing(pred_np, selem)
        
        # Convert back to tensor
        refined_prediction = torch.tensor(pred_np.astype(np.float32), device=self.device)
        
        # Blend with original prediction based on regularization strength
        if len(prediction.shape) == 4:
            refined_prediction = refined_prediction.view(prediction.shape)
        
        blended_prediction = (1 - self.shape_regularization_strength) * prediction + \
                             self.shape_regularization_strength * refined_prediction
        
        return blended_prediction
    
    def _save_agent_state(self, save_dict: Dict[str, Any]):
        """
        Add agent-specific state to the save dictionary.
        
        Args:
            save_dict: Dictionary to add agent-specific state to
        """
        save_dict["shape_regularization_strength"] = self.shape_regularization_strength
        save_dict["current_shape_prior"] = self.current_shape_prior
        save_dict["shape_priors"] = self.shape_priors
    
    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Load agent-specific state from the checkpoint.
        
        Args:
            checkpoint: Dictionary containing the agent state
            
        Returns:
            Whether the load was successful
        """
        if "shape_regularization_strength" in checkpoint:
            self.shape_regularization_strength = checkpoint["shape_regularization_strength"]
        
        if "current_shape_prior" in checkpoint:
            self.current_shape_prior = checkpoint["current_shape_prior"]
        
        if "shape_priors" in checkpoint:
            self.shape_priors = checkpoint["shape_priors"]
        
        return True
    
    def _reset_agent(self):
        """
        Reset agent-specific episode state.
        """
        # Reset shape regularization strength to initial value
        self.shape_regularization_strength = 0.5
        
        # Reset to default shape prior
        self.current_shape_prior = "circular"
