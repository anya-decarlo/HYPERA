#!/usr/bin/env python
# Region Agent - Focuses on optimizing regional overlap (Dice score)

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

class RegionAgent(BaseSegmentationAgent):
    """
    Segmentation agent that focuses on regional overlap (Dice score).
    
    This agent is specialized in optimizing the Dice score between the
    predicted segmentation and the ground truth.
    
    Attributes:
        name (str): Name of the agent
        state_manager: Reference to the shared state manager
        device (torch.device): Device to use for computation
        feature_extractor: Neural network for feature extraction
        log_dir (str): Directory for saving logs and checkpoints
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        state_manager,
        device: torch.device = None,
        feature_channels: int = 32,
        hidden_channels: int = 64,
        state_dim: int = 128,
        action_dim: int = 5,  # Number of segmentation parameters to control
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
        Initialize the region agent.
        
        Args:
            state_manager: Reference to the shared state manager
            device: Device to use for computation
            feature_channels: Number of channels in feature extractor
            hidden_channels: Number of channels in hidden layers
            state_dim: Dimension of state representation
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
        # Store additional parameters before calling super().__init__
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        
        super().__init__(
            name="RegionAgent",
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
        
        # Initialize agent-specific components
        self._initialize_agent()
        
        if self.verbose:
            self.logger.info("Initialized RegionAgent")
            
    def _initialize_agent(self):
        """
        Initialize the agent's networks and components.
        """
        # Feature extractor for region features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.feature_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        ).to(self.device)
        
        if self.verbose:
            self.logger.info("Initialized RegionAgent components")
            
    def _extract_features(self, observation):
        """
        Extract features from the observation.
        
        Args:
            observation: Dictionary containing observation data
            
        Returns:
            Extracted features
        """
        # Extract image and prediction from observation
        current_image = observation.get("current_image")
        current_prediction = observation.get("current_prediction")
        current_mask = observation.get("current_mask")
        
        if current_image is None or current_prediction is None:
            # Return zero features if data is missing
            return torch.zeros((1, self.state_dim), device=self.device)
        
        # Ensure tensors are on the correct device
        current_image = current_image.to(self.device)
        current_prediction = current_prediction.to(self.device)
        
        if current_mask is not None:
            current_mask = current_mask.to(self.device)
        
        # Use prediction as input to feature extractor
        if current_prediction.dim() == 3:
            # Add channel dimension if needed
            current_prediction = current_prediction.unsqueeze(1)
        elif current_prediction.dim() == 4:
            # Use first channel if multiple channels
            current_prediction = current_prediction[:, 0:1]
        
        # Extract features
        visual_features = self.feature_extractor(current_prediction)
        visual_features = visual_features.view(-1, self.hidden_channels * 2 * 8 * 8)
        
        # Project to state dimension
        state_features = torch.nn.functional.linear(
            visual_features, 
            torch.randn(self.state_dim, self.hidden_channels * 2 * 8 * 8, device=self.device)
        )
        
        return state_features
    
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
        
        # Ensure the features have the correct shape for the SAC policy
        if isinstance(features, torch.Tensor):
            # If features is a tensor, reshape it to match the expected state_dim
            if features.dim() == 2:
                # If features is [batch_size, feature_dim]
                if features.shape[1] != self.state_dim:
                    # Resize to match state_dim
                    if features.shape[1] > self.state_dim:
                        # Truncate if too large
                        features = features[:, :self.state_dim]
                    else:
                        # Pad with zeros if too small
                        padding = torch.zeros(features.shape[0], self.state_dim - features.shape[1], device=features.device)
                        features = torch.cat([features, padding], dim=1)
            else:
                # If not 2D, reshape to [1, state_dim]
                features = features.view(1, -1)
                if features.shape[1] != self.state_dim:
                    # Resize to match state_dim
                    if features.shape[1] > self.state_dim:
                        # Truncate if too large
                        features = features[:, :self.state_dim]
                    else:
                        # Pad with zeros if too small
                        padding = torch.zeros(1, self.state_dim - features.shape[1], device=features.device)
                        features = torch.cat([features, padding], dim=1)
        elif isinstance(features, np.ndarray):
            # If features is a numpy array, reshape it to match the expected state_dim
            if len(features.shape) == 1:
                # If features is [feature_dim]
                features = features.reshape(1, -1)
            
            # Resize to match state_dim
            if features.shape[1] != self.state_dim:
                if features.shape[1] > self.state_dim:
                    # Truncate if too large
                    features = features[:, :self.state_dim]
                else:
                    # Pad with zeros if too small
                    padding = np.zeros((features.shape[0], self.state_dim - features.shape[1]), dtype=features.dtype)
                    features = np.concatenate([features, padding], axis=1)
            
            # Convert to tensor if needed
            features = torch.FloatTensor(features).to(self.device)
        
        # Return features as state representation
        return features
    
    def apply_action(self, action):
        """
        Apply action to modify segmentation.
        
        Args:
            action: Action to apply
            
        Returns:
            Modified segmentation
        """
        # Get current prediction from state manager
        current_prediction = self.state_manager.get_current_prediction()
        
        if current_prediction is None:
            # Return None if no prediction is available
            return None
        
        # Ensure tensor is on the correct device
        current_prediction = current_prediction.to(self.device)
        
        # For RegionAgent, we interpret the action as parameters for morphological operations
        # and refinement of the segmentation
        
        # Apply action to modify segmentation
        # This is a simplified implementation - in practice, you would use the action
        # to control parameters of more sophisticated segmentation refinement operations
        
        # Handle different action types
        if isinstance(action, torch.Tensor):
            if action.numel() == 1:
                action_value = action.item()
            else:
                # If multi-dimensional tensor, take the first element
                action_value = action[0].item() if action.numel() > 0 else 0.0
        elif isinstance(action, np.ndarray):
            if action.size == 1:
                action_value = float(action[0])
            else:
                # If multi-dimensional array, take the first element
                action_value = float(action.flat[0]) if action.size > 0 else 0.0
        else:
            # Assume it's a scalar
            action_value = float(action)
            
        # Map action to threshold in [0.2, 0.8]
        threshold = 0.5 + 0.3 * action_value
        
        # Apply threshold to create binary segmentation
        binary_prediction = (current_prediction > threshold).float()
        
        # Update prediction in state manager
        self.state_manager.set_current_prediction(binary_prediction)
        
        return binary_prediction
    
    def _save_agent_state(self, save_dict: Dict[str, Any]):
        """
        Add agent-specific state to the save dictionary.
        
        Args:
            save_dict: Dictionary to add agent state to
        """
        # Add agent-specific parameters to save_dict
        save_dict.update({
            'feature_channels': self.feature_channels,
            'hidden_channels': self.hidden_channels,
            'feature_extractor_state': self.feature_extractor.state_dict() if self.feature_extractor else None,
        })
    
    def _load_agent_state(self, path):
        """
        Load agent state from file.
        
        Args:
            path: Path to load agent state from
        """
        # Load state dict from file
        state_dict = torch.load(path)
        
        # Load agent-specific parameters
        if 'feature_channels' in state_dict:
            self.feature_channels = state_dict['feature_channels']
        
        if 'hidden_channels' in state_dict:
            self.hidden_channels = state_dict['hidden_channels']
            
        # Load feature extractor state
        if 'feature_extractor_state' in state_dict and state_dict['feature_extractor_state'] and self.feature_extractor:
            self.feature_extractor.load_state_dict(state_dict['feature_extractor_state'])
            
        # Load SAC model separately
        sac_path = os.path.join(os.path.dirname(path), f"{self.name}_sac")
        if os.path.exists(f"{sac_path}_critic.pth"):
            self.sac.load_models(sac_path)
            
        if self.verbose:
            self.logger.info(f"Loaded RegionAgent state from {path}")
    
    def _reset_agent(self):
        """
        Reset the agent to its initial state.
        """
        # Reset history
        self.action_history = []
        self.reward_history = []
        self.observation_history = []
        
        # Re-initialize networks
        self._initialize_agent()
        
        if self.verbose:
            self.logger.info("Reset RegionAgent to initial state")
