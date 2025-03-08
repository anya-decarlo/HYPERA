#!/usr/bin/env python
# Region Agent - Focuses on regional overlap (Dice score)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

# Import base agent class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_segmentation_agent import BaseSegmentationAgent

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
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        learning_rate: float = 1e-4,
        automatic_entropy_tuning: bool = True,
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
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in SAC networks
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            gamma: Discount factor for future rewards
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            learning_rate: Learning rate for optimizer
            automatic_entropy_tuning: Whether to automatically tune entropy
            log_dir: Directory for saving logs and checkpoints
            verbose: Whether to print verbose output
        """
        # Initialize feature extractor parameters
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        
        # Call parent constructor with SAC parameters
        super().__init__(
            name="region",
            state_manager=state_manager,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=learning_rate,
            automatic_entropy_tuning=automatic_entropy_tuning,
            log_dir=log_dir,
            verbose=verbose
        )
    
    def _initialize_agent(self):
        """Initialize agent-specific parameters and networks."""
        # Initialize feature extractor (U-Net-like encoder)
        self.feature_extractor = nn.Sequential(
            # Input: [B, C, H, W, D]
            nn.Conv3d(1, self.feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.feature_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [B, feature_channels, H/2, W/2, D/2]
            
            nn.Conv3d(self.feature_channels, self.feature_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [B, feature_channels*2, H/4, W/4, D/4]
            
            nn.Conv3d(self.feature_channels * 2, self.feature_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.feature_channels * 4),
            nn.ReLU(inplace=True)
        ).to(self.device)
        
        # Initialize state encoder (converts extracted features to state representation)
        self.state_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),  # Adaptive pooling to fixed size
            nn.Flatten(),  # Flatten to 1D
            nn.Linear(self.feature_channels * 4 * 4 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, self.state_dim)
        ).to(self.device)
        
        # Initialize action decoder (converts actions to segmentation parameters)
        self.action_decoder = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh()  # Output normalized parameters
        ).to(self.device)
        
        # Initialize optimizer for feature extractor and state encoder
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.state_encoder.parameters()) + 
            list(self.action_decoder.parameters()),
            lr=self.lr
        )
    
    def get_state_representation(self, observation: torch.Tensor) -> np.ndarray:
        """
        Extract a state representation from the observation.
        
        Args:
            observation: The current observation (image or feature map)
            
        Returns:
            State representation as a numpy array
        """
        # Ensure observation is on the correct device
        observation = observation.to(self.device)
        
        # Extract features using the feature extractor
        with torch.no_grad():
            features = self.feature_extractor(observation)
            state = self.state_encoder(features)
        
        # Convert to numpy array
        return state.cpu().numpy()
    
    def _extract_features(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Extract features from the observation.
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary of extracted features
        """
        # Ensure observation is on the correct device
        observation = observation.to(self.device)
        
        # Extract features using the feature extractor
        with torch.no_grad():
            features = self.feature_extractor(observation)
        
        # Process features for region-specific information
        # For example, calculate regional statistics
        with torch.no_grad():
            # Calculate mean and variance of features
            mean_features = torch.mean(features, dim=(2, 3, 4))
            var_features = torch.var(features, dim=(2, 3, 4))
            
            # Calculate spatial distribution
            spatial_distribution = torch.mean(features, dim=1)
        
        return {
            "features": features,
            "mean_features": mean_features,
            "var_features": var_features,
            "spatial_distribution": spatial_distribution
        }
    
    def apply_action(self, action: np.ndarray, features: Dict[str, Any]) -> torch.Tensor:
        """
        Apply the action to produce a segmentation decision.
        
        Args:
            action: Action from the SAC policy
            features: Dictionary of extracted features
            
        Returns:
            Segmentation decision tensor
        """
        # Convert action to tensor
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        # Decode action into segmentation parameters
        segmentation_params = self.action_decoder(action_tensor)
        
        # Apply segmentation parameters to features
        # This is a simplified example - actual implementation would depend on
        # how segmentation parameters affect the segmentation process
        raw_features = features["features"]
        batch_size = raw_features.shape[0]
        
        # Example: Use parameters to adjust thresholds, weights, etc.
        # In a real implementation, these would control specific aspects of segmentation
        threshold = 0.5 + 0.3 * segmentation_params[0].item()  # Adjust threshold
        
        # Apply threshold to features to get segmentation mask
        # This is a simplified example
        segmentation_mask = torch.sigmoid(raw_features[:, 0:1, :, :, :]) > threshold
        
        return segmentation_mask.float()
    
    def _save_agent_state(self, save_dict: Dict[str, Any]):
        """
        Add agent-specific state to the save dictionary.
        
        Args:
            save_dict: Dictionary to add agent-specific state to
        """
        save_dict["feature_extractor"] = self.feature_extractor.state_dict()
        save_dict["state_encoder"] = self.state_encoder.state_dict()
        save_dict["action_decoder"] = self.action_decoder.state_dict()
        save_dict["optimizer"] = self.optimizer.state_dict()
    
    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Load agent-specific state from the checkpoint.
        
        Args:
            checkpoint: Dictionary containing the agent state
            
        Returns:
            Whether the load was successful
        """
        try:
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            self.state_encoder.load_state_dict(checkpoint["state_encoder"])
            self.action_decoder.load_state_dict(checkpoint["action_decoder"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            return True
        except Exception as e:
            self.logger.error(f"Failed to load agent-specific state: {e}")
            return False
    
    def _reset_agent(self):
        """Reset agent-specific episode state."""
        # No episode-specific state to reset for this agent
        pass
