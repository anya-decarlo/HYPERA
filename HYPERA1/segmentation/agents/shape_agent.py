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
        
        # Store the last prediction shape for reference
        self.last_prediction_shape = None
        
        # Default shape prior (can be "circular", "elongated", or "irregular")
        self.current_shape_prior = "circular"
        
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
    
    def get_state_representation(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Extract a state representation from the observation.
        
        Args:
            observation: The current observation
            
        Returns:
            State representation as a tensor
        """
        # Extract features from observation
        features = self._extract_features(observation)
        
        if not features:
            # Return zero state if features extraction failed
            return torch.zeros(1, self.state_dim, device=self.device)
        
        # Get shape features
        shape_features = features.get("shape_features", None)
        if shape_features is None:
            return torch.zeros(1, self.state_dim, device=self.device)
        
        # Flatten shape features
        shape_features_flat = shape_features.view(-1).detach().cpu().numpy()
        
        # Get shape metrics
        shape_metrics = features.get("shape_metrics", {})
        
        # Create state representation
        # Use a subset of features to match state_dim
        state_representation = np.zeros((1, self.state_dim), dtype=np.float32)
        
        # Fill state representation with shape features and metrics
        feature_count = min(self.state_dim - 3, len(shape_features_flat))
        state_representation[0, :feature_count] = shape_features_flat[:feature_count]
        
        # Add shape regularization strength
        state_representation[0, self.state_dim - 3] = self.shape_regularization_strength
        
        # Add shape prior indicators (one-hot encoding)
        if self.current_shape_prior == "circular":
            state_representation[0, self.state_dim - 2] = 1.0
        elif self.current_shape_prior == "elongated":
            state_representation[0, self.state_dim - 1] = 1.0
        
        # Convert to tensor
        return torch.FloatTensor(state_representation).to(self.device)
    
    def apply_action(self, action: np.ndarray, observation=None) -> torch.Tensor:
        """
        Apply the action to produce a segmentation decision.
        
        Args:
            action: Action from the SAC policy
            observation: Optional observation dictionary
            
        Returns:
            Segmentation decision tensor
        """
        # Get current prediction from observation or state manager
        if observation is not None and 'current_segmentation' in observation:
            current_prediction = observation['current_segmentation']
            if isinstance(current_prediction, np.ndarray):
                current_prediction = torch.from_numpy(current_prediction).float().to(self.device)
        else:
            # Get from state manager
            current_prediction = self.state_manager.get_current_prediction()
            
        if current_prediction is None:
            print("ShapeAgent: No current prediction available")
            # Return a default prediction (all zeros) with the same shape as expected
            # This is better than returning None
            if hasattr(self, 'last_prediction_shape') and self.last_prediction_shape is not None:
                return torch.zeros(self.last_prediction_shape, device=self.device)
            else:
                # Default shape if we don't know the expected shape
                return torch.zeros((1, 1, 64, 64), device=self.device)
        
        # Store the shape for future reference
        self.last_prediction_shape = current_prediction.shape
        
        # Ensure tensor is on the correct device
        current_prediction = current_prediction.to(self.device)
        
        # Add batch and channel dimensions if needed
        if current_prediction.dim() == 2:
            current_prediction = current_prediction.unsqueeze(0).unsqueeze(0)
        elif current_prediction.dim() == 3:
            current_prediction = current_prediction.unsqueeze(0)
        
        # Scale action to shape regularization strength adjustment
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
            
        # Ensure action has the correct shape
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        action_scaled = action[0, 0].item() * self.strength_step
        
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
        # Store original shape and dimensions
        original_shape = prediction.shape
        original_device = prediction.device
        
        # Convert to numpy for morphological operations
        # Extract the first image from batch and first channel
        pred_np = prediction.detach().cpu().numpy()
        
        # Handle different dimensions
        if len(pred_np.shape) == 4:  # [B, C, H, W]
            # Process only the first image from the batch and first channel
            pred_np_2d = pred_np[0, 0]
        elif len(pred_np.shape) == 3:  # [C, H, W]
            # Process only the first channel
            pred_np_2d = pred_np[0]
        else:  # [H, W]
            pred_np_2d = pred_np
        
        # Apply morphological operations based on shape prior and regularization strength
        from skimage import morphology
        
        # Determine structuring element size based on regularization strength
        struct_size = max(1, int(3 * self.shape_regularization_strength))
        
        # Apply morphological operations to 2D image
        if self.current_shape_prior == "circular":
            # Use disk structuring element for circular shapes
            selem = morphology.disk(struct_size)
            # Apply opening to remove small protrusions
            refined_np_2d = morphology.opening(pred_np_2d > 0.5, selem)
            # Apply closing to fill small holes
            refined_np_2d = morphology.closing(refined_np_2d, selem)
        
        elif self.current_shape_prior == "elongated":
            # Use elliptical structuring element for elongated shapes
            selem = morphology.ellipse(struct_size, 2 * struct_size)
            # Apply opening and closing
            refined_np_2d = morphology.opening(pred_np_2d > 0.5, selem)
            refined_np_2d = morphology.closing(refined_np_2d, selem)
        
        else:  # irregular
            # Use smaller structuring element for irregular shapes
            selem = morphology.square(struct_size)
            # Apply less aggressive morphological operations
            refined_np_2d = morphology.opening(pred_np_2d > 0.5, selem)
            refined_np_2d = morphology.closing(refined_np_2d, selem)
        
        # Convert back to tensor and restore original dimensions
        refined_tensor = torch.tensor(refined_np_2d.astype(np.float32), device=original_device)
        
        # Reshape to match original dimensions
        if len(original_shape) == 4:  # [B, C, H, W]
            refined_tensor = refined_tensor.unsqueeze(0).unsqueeze(0)
            # Repeat for all batch items if needed
            if original_shape[0] > 1:
                refined_tensor = refined_tensor.repeat(original_shape[0], 1, 1, 1)
            # Repeat for all channels if needed
            if original_shape[1] > 1:
                refined_tensor = refined_tensor.repeat(1, original_shape[1], 1, 1)
        elif len(original_shape) == 3:  # [C, H, W]
            refined_tensor = refined_tensor.unsqueeze(0)
            # Repeat for all channels if needed
            if original_shape[0] > 1:
                refined_tensor = refined_tensor.repeat(original_shape[0], 1, 1)
        
        # Blend with original prediction based on regularization strength
        blended_prediction = (1 - self.shape_regularization_strength) * prediction + \
                            self.shape_regularization_strength * refined_tensor
        
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
