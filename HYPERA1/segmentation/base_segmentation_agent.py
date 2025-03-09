#!/usr/bin/env python
# Base Segmentation Agent - Foundation for all segmentation agents

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HYPERA1.segmentation.utils.sac.sac import SAC
from HYPERA1.segmentation.utils.replay_buffer import ReplayBuffer

class BaseSegmentationAgent(ABC):
    """
    Abstract base class for all segmentation agents in the HYPERA system.
    
    This class defines the interface that all specialized segmentation agents must implement.
    Each agent is responsible for a specific aspect of the segmentation process, such as
    region overlap, boundary accuracy, object detection, shape preservation, etc.
    
    The agent uses Soft Actor-Critic (SAC) for reinforcement learning, providing a
    consistent approach with the hyperparameter optimization agents.
    
    Attributes:
        name (str): Name of the agent
        state_manager: Reference to the shared state manager
        device (torch.device): Device to use for computation
        observation_shape (tuple): Shape of the observation space
        action_shape (tuple): Shape of the action space
        log_dir (str): Directory for saving logs and checkpoints
        verbose (bool): Whether to print verbose output
        sac (SAC): Soft Actor-Critic implementation
        replay_buffer (ReplayBuffer): Buffer for storing experience
    """
    
    def __init__(
        self,
        name: str,
        state_manager,
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
        Initialize the base segmentation agent.
        
        Args:
            name: Name of the agent
            state_manager: Reference to the shared state manager
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
        self.name = name
        self.state_manager = state_manager
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.update_frequency = update_frequency
        self.log_dir = log_dir
        self.verbose = verbose
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"SegmentationAgent.{name}")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        # Initialize metrics tracking
        self.metrics = {}
        self.training_step = 0
        self.episode_rewards = []
        
        # Initialize SAC agent
        self.sac = SAC(
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
            device=self.device,
            log_dir=log_dir,
            name=name
        )
        
        # Store the last state, action, and next state for learning
        self.last_state = None
        self.last_action = None
        self.current_state = None
        
        # Initialize agent-specific parameters
        self._initialize_agent()
        
        if self.verbose:
            self.logger.info(f"Initialized {self.name} agent with SAC")
    
    @abstractmethod
    def _initialize_agent(self):
        """
        Initialize agent-specific parameters and networks.
        Must be implemented by each specialized agent.
        """
        pass
    
    @abstractmethod
    def get_state_representation(self, observation: torch.Tensor) -> np.ndarray:
        """
        Extract a state representation from the observation.
        
        Args:
            observation: The current observation (image or feature map)
            
        Returns:
            State representation as a numpy array
        """
        pass
    
    @abstractmethod
    def apply_action(self, action: np.ndarray, features: Dict[str, Any]) -> torch.Tensor:
        """
        Apply the action to produce a segmentation decision.
        
        Args:
            action: Action from the SAC policy
            features: Dictionary of extracted features
            
        Returns:
            Segmentation decision tensor
        """
        pass
    
    def observe(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Process an observation and extract relevant features.
        
        Args:
            observation: The current observation (image or feature map)
            
        Returns:
            Dictionary of extracted features
        """
        # Get state representation
        state = self.get_state_representation(observation)
        self.current_state = state
        
        # Extract features (agent-specific)
        features = self._extract_features(observation)
        
        return features
    
    @abstractmethod
    def _extract_features(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Extract features from the observation.
        Must be implemented by each specialized agent.
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary of extracted features
        """
        pass
    
    def decide(self, features: Dict[str, Any]) -> torch.Tensor:
        """
        Make a segmentation decision based on the extracted features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Action tensor representing the segmentation decision
        """
        # Get state representation from features
        if self.current_state is None:
            raise ValueError("Current state is None. Call observe() first.")
        
        # Select action using SAC
        action = self.sac.select_action(self.current_state)
        self.last_action = action
        
        # Apply action to produce segmentation decision
        decision = self.apply_action(action, features)
        
        return decision
    
    def learn(self, reward: float, done: bool = False) -> Dict[str, float]:
        """
        Update the agent's policy based on the received reward.
        
        Args:
            reward: The reward received for the last action
            done: Whether the episode is done
            
        Returns:
            Dictionary of learning metrics
        """
        # Skip if we don't have a complete transition yet
        if self.last_state is None or self.last_action is None or self.current_state is None:
            return {"loss": 0.0}
        
        # Add experience to replay buffer
        self.sac.add_experience(
            self.last_state,
            self.last_action,
            reward,
            self.current_state,
            done
        )
        
        # Increment training step
        self.training_step += 1
        
        # Only update parameters based on update frequency
        metrics = {"loss": 0.0}
        if self.training_step % self.update_frequency == 0:
            # Update SAC parameters
            metrics = self.sac.update_parameters()
        
        # Update last state
        self.last_state = self.current_state
        
        return metrics
    
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Process an observation and return an action.
        
        Args:
            observation: The current observation
            
        Returns:
            Action tensor
        """
        # Store the last state before updating
        self.last_state = self.current_state
        
        # Extract features and decide
        features = self.observe(observation)
        action = self.decide(features)
        
        return action
    
    def update(self, reward: float, done: bool = False) -> Dict[str, float]:
        """
        Update the agent with a reward signal.
        
        Args:
            reward: The reward received
            done: Whether the episode is done
            
        Returns:
            Dictionary of update metrics
        """
        self.training_step += 1
        self.episode_rewards.append(reward)
        
        # Learn from the reward
        metrics = self.learn(reward, done)
        
        # Update metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log metrics periodically
        if self.verbose and self.training_step % 100 == 0:
            self._log_metrics()
        
        return metrics
    
    def _log_metrics(self):
        """Log the current metrics."""
        log_str = f"{self.name} - Step {self.training_step}: "
        for key, values in self.metrics.items():
            if values:
                log_str += f"{key}: {np.mean(values[-100:]):.4f} | "
        self.logger.info(log_str)
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the agent's state.
        
        Args:
            path: Path to save the agent state. If None, use default path.
            
        Returns:
            Path where the agent was saved
        """
        if path is None:
            path = os.path.join(self.log_dir, f"{self.name}_agent.pt")
        
        save_dict = {
            "name": self.name,
            "training_step": self.training_step,
            "metrics": self.metrics,
            "episode_rewards": self.episode_rewards,
            # Agent-specific state will be added by subclasses
        }
        
        self._save_agent_state(save_dict)
        
        torch.save(save_dict, path)
        
        # Save SAC model separately
        sac_path = os.path.join(os.path.dirname(path), f"{self.name}_sac")
        self.sac.save_models(sac_path)
        
        if self.verbose:
            self.logger.info(f"Saved agent state to {path}")
        
        return path
    
    @abstractmethod
    def _save_agent_state(self, save_dict: Dict[str, Any]):
        """
        Add agent-specific state to the save dictionary.
        Must be implemented by each specialized agent.
        
        Args:
            save_dict: Dictionary to add agent-specific state to
        """
        pass
    
    def load(self, path: str) -> bool:
        """
        Load the agent's state.
        
        Args:
            path: Path to load the agent state from
            
        Returns:
            Whether the load was successful
        """
        if not os.path.exists(path):
            self.logger.error(f"Cannot load agent state: {path} does not exist")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load common attributes
            self.name = checkpoint["name"]
            self.training_step = checkpoint["training_step"]
            self.metrics = checkpoint["metrics"]
            self.episode_rewards = checkpoint["episode_rewards"]
            
            # Load agent-specific state
            success = self._load_agent_state(checkpoint)
            
            # Load SAC model separately
            sac_path = os.path.join(os.path.dirname(path), f"{self.name}_sac")
            if os.path.exists(sac_path):
                self.sac.load_models(sac_path)
            
            if self.verbose:
                self.logger.info(f"Loaded agent state from {path}")
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to load agent state: {e}")
            return False
    
    @abstractmethod
    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Load agent-specific state from the checkpoint.
        Must be implemented by each specialized agent.
        
        Args:
            checkpoint: Dictionary containing the agent state
            
        Returns:
            Whether the load was successful
        """
        pass
    
    def reset(self):
        """Reset the agent's episode-specific state."""
        self.episode_rewards = []
        self.last_state = None
        self.last_action = None
        self.current_state = None
        self._reset_agent()
        
        if self.verbose:
            self.logger.info(f"Reset {self.name} agent")
    
    @abstractmethod
    def _reset_agent(self):
        """
        Reset agent-specific episode state.
        Must be implemented by each specialized agent.
        """
        pass
    
    def get_last_actions(self):
        """
        Get the last actions taken by the agent.
        
        Returns:
            Dictionary of action names and values
        """
        if self.last_action is None:
            return {}
            
        # Convert tensor to float if needed
        if isinstance(self.last_action, torch.Tensor):
            if self.last_action.numel() == 1:
                action_value = self.last_action.item()
            else:
                action_value = self.last_action[0].item() if self.last_action.numel() > 0 else 0.0
        elif isinstance(self.last_action, np.ndarray):
            if self.last_action.size == 1:
                action_value = float(self.last_action[0])
            else:
                action_value = float(self.last_action.flat[0]) if self.last_action.size > 0 else 0.0
        else:
            action_value = float(self.last_action)
            
        return {'segmentation_refinement': action_value}
    
    def get_last_reward_info(self):
        """
        Get information about the last reward.
        
        Returns:
            Dictionary with reward components
        """
        if not hasattr(self, 'last_reward_info') or self.last_reward_info is None:
            return {'total': 0.0, 'dice_improvement': 0.0, 'boundary_improvement': 0.0, 'efficiency': 0.0}
            
        return self.last_reward_info
