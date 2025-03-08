#!/usr/bin/env python
# Learning Rate Agent for Multi-Agent Hyperparameter Optimization System

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import math
import torch

from ..base_agent import BaseHyperparameterAgent
from ..shared_state import SharedStateManager

class LearningRateAgent(BaseHyperparameterAgent):
    """
    Agent responsible for optimizing the learning rate hyperparameter.
    
    This agent uses Soft Actor-Critic (SAC) to learn an optimal policy for
    adjusting the learning rate based on training metrics like loss, validation
    dice score, and gradient statistics.
    """
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        # SAC parameters
        state_dim: int = 8,  # Increased state dimension for more metrics
        action_dim: int = 1,
        hidden_dim: int = 128,
        replay_buffer_size: int = 5000,
        batch_size: int = 32,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        automatic_entropy_tuning: bool = True,
        # Agent timing parameters
        update_frequency: int = 1,
        patience: int = 2,
        cooldown: int = 3,
        # Logging parameters
        log_dir: str = "results",
        verbose: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the learning rate agent.
        
        Args:
            shared_state_manager: Reference to the shared state manager
            initial_lr: Initial learning rate value
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
            state_dim: Dimension of state space for SAC
            action_dim: Dimension of action space for SAC
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
        """
        # Calculate action space based on min/max learning rate
        # We'll use log scale for learning rate, so action space is log(min_lr) to log(max_lr)
        log_min_lr = math.log10(min_lr)
        log_max_lr = math.log10(max_lr)
        action_space = (log_min_lr, log_max_lr)
        
        super().__init__(
            name="learning_rate_agent",
            hyperparameter_key="learning_rate",
            shared_state_manager=shared_state_manager,
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
            patience=patience,
            cooldown=cooldown,
            log_dir=log_dir,
            verbose=verbose,
            device=device
        )
        
        # Learning rate specific parameters
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr
        
        # Initialize the learning rate in shared state
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, self.current_lr)
        
        if self.verbose:
            logging.info(f"Initialized learning rate agent with initial LR: {self.initial_lr}")
    
    def _get_state_representation(self) -> np.ndarray:
        """
        Get a numerical representation of the current state.
        
        The state includes:
        1. Current learning rate (log scale)
        2. Current loss
        3. Loss change (current - previous)
        4. Current validation dice score
        5. Dice score change
        6. Gradient norm
        7. Current epoch / total epochs
        8. Epochs without improvement
        
        Returns:
            State vector as numpy array
        """
        # Get metrics from shared state
        metrics = self.shared_state_manager.get_latest_metrics()
        
        # Get previous metrics for calculating changes
        prev_metrics = self.shared_state_manager.get_metrics_at_epoch(self.current_epoch - 1) if self.current_epoch > 0 else metrics
        
        # Extract relevant metrics
        current_loss = metrics.get('loss', 0.0)
        prev_loss = prev_metrics.get('loss', current_loss)
        loss_change = current_loss - prev_loss
        
        current_dice = metrics.get('dice_score', 0.0)
        prev_dice = prev_metrics.get('dice_score', current_dice)
        dice_change = current_dice - prev_dice
        
        gradient_norm = metrics.get('gradient_norm', 1.0)
        
        # Normalize epoch
        total_epochs = self.shared_state_manager.get_total_epochs()
        normalized_epoch = self.current_epoch / total_epochs if total_epochs > 0 else 0
        
        # Normalize epochs without improvement
        max_patience = 10  # Arbitrary max value for normalization
        normalized_patience = self.epochs_without_improvement / max_patience
        
        # Create state vector
        state = np.array([
            math.log10(self.current_lr),  # Log scale for learning rate
            current_loss,
            loss_change,
            current_dice,
            dice_change,
            gradient_norm,
            normalized_epoch,
            normalized_patience
        ], dtype=np.float32)
        
        return state
    
    def _process_action(self, action: np.ndarray) -> float:
        """
        Process the continuous action from SAC into a concrete learning rate.
        
        The action is a log10 value that we convert to an actual learning rate.
        
        Args:
            action: Action vector from SAC (log10 of learning rate)
            
        Returns:
            New learning rate value
        """
        # Extract the action (log10 of learning rate)
        log_lr = float(action[0])
        
        # Convert to actual learning rate and clip to min/max
        new_lr = 10 ** log_lr
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        return new_lr
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate the reward based on the current metrics.
        
        The reward is based on:
        1. Improvement in dice score (primary objective)
        2. Reduction in loss (secondary objective)
        3. Stability of training (penalize large oscillations)
        
        Args:
            metrics: Dictionary of current training metrics
            
        Returns:
            Calculated reward value
        """
        # Get current metrics
        current_dice = metrics.get('dice_score', 0.0)
        current_loss = metrics.get('loss', float('inf'))
        
        # Get previous metrics
        prev_metrics = self.shared_state_manager.get_metrics_at_epoch(self.current_epoch - 1)
        prev_dice = prev_metrics.get('dice_score', 0.0) if prev_metrics else 0.0
        prev_loss = prev_metrics.get('loss', float('inf')) if prev_metrics else float('inf')
        
        # Calculate changes
        dice_improvement = current_dice - prev_dice
        loss_reduction = prev_loss - current_loss
        
        # Get gradient norm as a measure of stability
        gradient_norm = metrics.get('gradient_norm', 1.0)
        
        # Calculate reward components
        dice_reward = dice_improvement * 10.0  # Primary objective, higher weight
        loss_reward = loss_reduction * 2.0  # Secondary objective
        
        # Penalize instability (high gradient norm)
        stability_penalty = -max(0, gradient_norm - 10.0) * 0.1
        
        # Penalize very small learning rates that might slow down training
        lr_penalty = -1.0 if self.current_lr < 1e-5 else 0.0
        
        # Combine rewards
        reward = dice_reward + loss_reward + stability_penalty + lr_penalty
        
        # Clip reward to reasonable range
        reward = max(-10.0, min(10.0, reward))
        
        if self.verbose:
            logging.debug(f"Reward components - Dice: {dice_reward:.4f}, Loss: {loss_reward:.4f}, "
                         f"Stability: {stability_penalty:.4f}, LR: {lr_penalty:.4f}")
        
        return reward
    
    def _apply_action(self, new_lr: float) -> None:
        """
        Apply the new learning rate.
        
        Args:
            new_lr: New learning rate value to apply
        """
        # Update current learning rate
        old_lr = self.current_lr
        self.current_lr = new_lr
        
        # Update learning rate in shared state
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, self.current_lr)
        
        # Log the change
        if self.verbose:
            logging.info(f"Learning rate changed: {old_lr:.6f} -> {self.current_lr:.6f}")
    
    def get_current_lr(self) -> float:
        """
        Get the current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.current_lr
