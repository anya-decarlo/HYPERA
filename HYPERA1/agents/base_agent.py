#!/usr/bin/env python
# Base Agent for Multi-Agent Hyperparameter Optimization System

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
import json
import os
from datetime import datetime
from pathlib import Path

from .utils.sac import SAC
from .utils.enhanced_rewards import EnhancedRewardSystem
from .shared_state import SharedStateManager

class BaseHyperparameterAgent(ABC):
    """
    Abstract Base Class for all hyperparameter optimization agents.
    
    This class defines the common interface and functionality that all specialized
    hyperparameter agents must implement. It provides the foundation for the
    Soft Actor-Critic (SAC) approach to hyperparameter optimization.
    
    Each specialized agent will focus on optimizing a specific hyperparameter or
    a small set of related hyperparameters, while communicating with other agents
    through a shared state manager.
    """
    
    def __init__(
        self,
        name: str,
        hyperparameter_key: str,
        shared_state_manager: SharedStateManager,
        # SAC parameters
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
        # Agent timing parameters
        update_frequency: int = 1,  # How often to consider updates (in epochs)
        patience: int = 3,          # Epochs to wait before considering action
        cooldown: int = 5,          # Epochs to wait after an action
        # Logging parameters
        log_dir: str = "results",
        verbose: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        eligibility_trace_length: int = 10,
        n_step: int = 3,
        stability_weight: float = 0.3,
        generalization_weight: float = 0.4,
        efficiency_weight: float = 0.3,
        use_adaptive_scaling: bool = True,
        use_phase_aware_scaling: bool = True,
        auto_balance_components: bool = True,
        reward_clip_range: Tuple[float, float] = (-10.0, 10.0),
        reward_scaling_window: int = 100,
        priority: int = 0,
    ):
        """
        Initialize the base hyperparameter agent.
        
        Args:
            name: Unique identifier for this agent
            hyperparameter_key: The key used to identify this hyperparameter in the shared state
            shared_state_manager: Reference to the shared state manager
            state_dim: Dimension of state space for SAC
            action_dim: Dimension of action space for SAC
            action_space: Tuple of (min_action, max_action) for SAC
            hidden_dim: Dimension of hidden layers in SAC networks
            replay_buffer_size: Size of replay buffer for SAC
            batch_size: Batch size for SAC training
            gamma: Discount factor for SAC
            tau: Target network update rate for SAC
            alpha: Temperature parameter for entropy in SAC
            lr: Learning rate for SAC
            automatic_entropy_tuning: Whether to automatically tune entropy in SAC
            update_frequency: How often to consider updates (in epochs)
            patience: Epochs to wait before considering action
            cooldown: Epochs to wait after taking an action
            log_dir: Directory to save agent logs
            verbose: Whether to print agent actions
            device: Device to use for tensors
            eligibility_trace_length: Length of eligibility traces for reward calculation
            n_step: Number of steps for n-step returns
            stability_weight: Weight for stability component in reward
            generalization_weight: Weight for generalization component in reward
            efficiency_weight: Weight for efficiency component in reward
            use_adaptive_scaling: Whether to use adaptive reward scaling
            use_phase_aware_scaling: Whether to use phase-aware scaling
            auto_balance_components: Whether to auto-balance reward components
            reward_clip_range: Range for clipping rewards
            reward_scaling_window: Window size for reward statistics
            priority: Priority of the agent
        """
        self.name = name
        self.hyperparameter_key = hyperparameter_key
        self.shared_state_manager = shared_state_manager
        self.device = device
        self.priority = priority
        
        # SAC parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        
        # Agent timing parameters
        self.update_frequency = update_frequency
        self.patience = patience
        self.cooldown = cooldown
        
        # Logging parameters
        self.log_dir = log_dir
        self.verbose = verbose
        
        # State tracking
        self.epochs_without_improvement = 0
        self.cooldown_counter = 0
        self.best_metric = -float('inf')
        self.last_action = None
        self.last_state = None
        self.last_reward = 0
        self.current_epoch = 0
        
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
            device=device,
            log_dir=log_dir,
            name=name
        )
        
        # Initialize enhanced reward system
        self.reward_system = EnhancedRewardSystem(
            eligibility_trace_length=eligibility_trace_length,
            n_step=n_step,
            stability_weight=stability_weight,
            generalization_weight=generalization_weight,
            efficiency_weight=efficiency_weight,
            decay_factor=0.9,
            discount_factor=gamma,
            reward_scaling_window=reward_scaling_window,
            reward_clip_range=reward_clip_range,
            use_adaptive_scaling=use_adaptive_scaling,
            use_phase_aware_scaling=use_phase_aware_scaling,
            auto_balance_components=auto_balance_components
        )
        
        # Setup logging
        self.setup_logging()
        
        if self.verbose:
            logging.info(f"Initialized {self.name} agent for {self.hyperparameter_key}")
    
    def setup_logging(self):
        """Set up logging for this agent."""
        log_path = os.path.join(self.log_dir, f"{self.name}_agent")
        os.makedirs(log_path, exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join(log_path, f"{self.name}_log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[file_handler, console_handler]
        )
    
    @abstractmethod
    def _get_state_representation(self) -> np.ndarray:
        """
        Get a numerical representation of the current state.
        
        Returns:
            State vector as numpy array
        """
        pass
    
    @abstractmethod
    def _process_action(self, action: np.ndarray) -> Any:
        """
        Process the continuous action from SAC into a concrete hyperparameter change.
        
        Args:
            action: Action vector from SAC
            
        Returns:
            Processed action that can be applied to the hyperparameter
        """
        pass
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate reward based on current metrics.
        
        Args:
            metrics: Dictionary of metrics from the training process
            
        Returns:
            Calculated reward value
        """
        # Store previous metrics
        self.previous_metrics = self.current_metrics
        self.current_metrics = metrics
        
        # Add experience to reward system
        current_state = self._get_state_representation()
        self.reward_system.add_experience(current_state, self.last_action, metrics)
        
        # Get processed experiences with rewards
        experiences = self.reward_system.get_processed_experiences()
        
        # Add experiences to replay buffer
        if experiences:
            for exp in experiences:
                # Amplify rewards to encourage more action
                exp["reward"] = exp["reward"] * 2.0  # Amplify rewards
                
                self.sac.add_experience(
                    exp["state"],
                    exp["action"],
                    exp.get("n_step_return", exp["reward"]),
                    exp["next_state"],
                    exp["done"]
                )
        
        # Get latest reward components for logging
        reward_components = self.reward_system.get_latest_reward_components()
        
        # Amplify reward components to encourage more action
        for key in reward_components:
            if isinstance(reward_components[key], (int, float)):
                reward_components[key] = reward_components[key] * 2.0
        
        # Update training phase information
        self.training_info["phase"] = reward_components.get("training_phase", "exploration")
        
        # Log reward components if writer is available
        if self.writer:
            self.writer.add_scalar(
                f"{self.name}/reward/stability",
                reward_components.get("stability", 0.0),
                self.step_counter
            )
            self.writer.add_scalar(
                f"{self.name}/reward/generalization",
                reward_components.get("generalization", 0.0),
                self.step_counter
            )
            self.writer.add_scalar(
                f"{self.name}/reward/efficiency",
                reward_components.get("efficiency", 0.0),
                self.step_counter
            )
            self.writer.add_scalar(
                f"{self.name}/reward/total",
                reward_components.get("total", 0.0),
                self.step_counter
            )
            self.writer.add_scalar(
                f"{self.name}/reward/normalized_total",
                reward_components.get("normalized_total", 0.0),
                self.step_counter
            )
            self.writer.add_scalar(
                f"{self.name}/training_phase",
                {"exploration": 0, "exploitation": 1, "fine_tuning": 2}.get(self.training_info["phase"], 0),
                self.step_counter
            )
        
        # Return the normalized and clipped reward
        return reward_components.get("normalized_total", 0.0) * 2.0  # Amplify final reward
    
    @abstractmethod
    def _apply_action(self, processed_action: Any) -> None:
        """
        Apply the processed action to modify the hyperparameter.
        
        Args:
            processed_action: The processed action to apply
        """
        pass
    
    def apply_action(self, action, epoch):
        """
        Apply the action selected by the agent.
        
        Args:
            action: The action to apply
            epoch: Current training epoch
            
        Returns:
            Result of applying the action
        """
        processed_action = self._process_action(action)
        return self._apply_action(processed_action)
    
    def should_update(self, epoch: int) -> bool:
        """
        Determine if the agent should consider an update in the current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            True if the agent should consider an update, False otherwise
        """
        self.current_epoch = epoch
        
        # Don't update during cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Only update at the specified frequency
        if epoch % self.update_frequency != 0:
            return False
        
        # Always consider an update when frequency and cooldown conditions are met
        return True
    
    def update(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Update the agent based on current metrics and potentially take an action.
        
        Args:
            metrics: Dictionary of current training metrics
            
        Returns:
            Dictionary with update information
        """
        self.step_counter += 1
        
        # Update best performance
        if "val_dice_score" in metrics:
            self.training_info["best_performance"] = max(
                self.training_info["best_performance"],
                metrics["val_dice_score"]
            )
        
        # Only update on specified frequency
        if self.step_counter % self.update_frequency != 0:
            return {}
            
        # Get current state
        state = self._get_state_representation()
        
        # Select action
        action = self.sac.select_action(state)
        
        # Add action to history
        self.action_history.append(action)
        
        # Calculate reward (this also adds experiences to replay buffer)
        reward = self._calculate_reward(metrics)
        
        # Train SAC agent
        if len(self.sac.replay_buffer) > self.sac.batch_size:
            critic_loss, actor_loss, alpha_loss = self.sac.update_parameters()
            
            # Log losses if writer is available
            if self.writer:
                self.writer.add_scalar(
                    f"{self.name}/loss/critic",
                    critic_loss,
                    self.step_counter
                )
                self.writer.add_scalar(
                    f"{self.name}/loss/actor",
                    actor_loss,
                    self.step_counter
                )
                if alpha_loss is not None:
                    self.writer.add_scalar(
                        f"{self.name}/loss/alpha",
                        alpha_loss,
                        self.step_counter
                    )
        
        # Convert action to hyperparameters
        hyperparameters = self._process_action(action)
        
        # Log hyperparameters if writer is available
        if self.writer:
            for name, value in hyperparameters.items():
                self.writer.add_scalar(
                    f"{self.name}/hyperparameters/{name}",
                    value,
                    self.step_counter
                )
        
        # Apply the processed action
        self._apply_action(hyperparameters)
        
        # Update tracking variables
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        self.cooldown_counter = self.cooldown
        
        # Track best metric and epochs without improvement
        current_metric = metrics.get('dice_score', 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Log the update
        if self.verbose:
            logging.info(f"{self.name} agent - Action: {hyperparameters}, Reward: {reward:.4f}")
        
        # Return update information
        return {
            'agent': self.name,
            'hyperparameter': self.hyperparameter_key,
            'state': state.tolist(),
            'action': action.tolist(),
            'processed_action': hyperparameters,

            'epoch': self.current_epoch,
            'reward_components': self.reward_system.get_latest_reward_components()
        }
    
    def save_models(self) -> None:
        """Save the SAC models."""
        self.sac.save_models()
    
    def load_models(self, path: str) -> None:
        """
        Load the SAC models.
        
        Args:
            path: Path to load models from
        """
        self.sac.load_models(path)
    
    def get_param_name(self):
        """
        Get the name of the hyperparameter that this agent controls.
        
        Returns:
            The hyperparameter key
        """
        return self.hyperparameter_key
