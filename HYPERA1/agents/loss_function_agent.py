#!/usr/bin/env python
# Loss Function Agent for Multi-Agent Hyperparameter Optimization

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

from .base_agent import BaseHyperparameterAgent
from .shared_state import SharedStateManager

class LossFunctionAgent(BaseHyperparameterAgent):
    """
    Specialized agent for optimizing loss function parameters.
    
    This agent uses SAC to learn the optimal loss function parameter adjustment
    policy based on observed training metrics. It can dynamically adjust the
    loss function parameters during training to improve performance.
    """
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        initial_lambda_ce: float = 1.0,
        initial_lambda_dice: float = 1.0,
        initial_focal_gamma: float = 2.0,
        min_lambda_ce: float = 0.1,
        max_lambda_ce: float = 5.0,
        min_lambda_dice: float = 0.1,
        max_lambda_dice: float = 5.0,
        min_focal_gamma: float = 0.5,
        max_focal_gamma: float = 4.0,
        update_frequency: int = 15,  # Less frequent updates
        patience: int = 7,
        cooldown: int = 20,
        log_dir: str = "results",
        verbose: bool = True,
        metrics_to_track: List[str] = ["loss", "val_loss", "dice_score", "ce_loss", "dice_loss"],
        state_dim: int = 24,
        hidden_dim: int = 256,
        use_enhanced_state: bool = True,
        eligibility_trace_length: int = 10,
        n_step: int = 3,
        stability_weight: float = 0.3,
        generalization_weight: float = 0.4,
        efficiency_weight: float = 0.3,
        use_adaptive_scaling: bool = True,
        use_phase_aware_scaling: bool = True,
        auto_balance_components: bool = True,
        reward_clip_range: Optional[Tuple[float, float]] = None,
        reward_scaling_window: int = 100,
        device: Optional[torch.device] = None,
        name: str = "loss_function_agent",
        priority: int = 0
    ):
        """
        Initialize the loss function agent.
        
        Args:
            shared_state_manager: Manager for shared state between agents
            initial_lambda_ce: Initial weight for cross-entropy loss
            initial_lambda_dice: Initial weight for dice loss
            initial_focal_gamma: Initial gamma parameter for focal loss
            min_lambda_ce: Minimum allowed lambda_ce value
            max_lambda_ce: Maximum allowed lambda_ce value
            min_lambda_dice: Minimum allowed lambda_dice value
            max_lambda_dice: Maximum allowed lambda_dice value
            min_focal_gamma: Minimum allowed focal_gamma value
            max_focal_gamma: Maximum allowed focal_gamma value
            update_frequency: How often to update loss function parameters (in epochs)
            patience: Epochs to wait before considering action
            cooldown: Epochs to wait after an action
            log_dir: Directory for saving logs and agent states
            verbose: Whether to print verbose output
            metrics_to_track: List of metrics to track
            state_dim: Dimension of state space
            hidden_dim: Dimension of hidden layers
            use_enhanced_state: Whether to use enhanced state representation
            eligibility_trace_length: Length of eligibility traces
            n_step: Number of steps for n-step returns
            stability_weight: Weight for stability component of reward
            generalization_weight: Weight for generalization component of reward
            efficiency_weight: Weight for efficiency component of reward
            use_adaptive_scaling: Whether to use adaptive reward scaling
            use_phase_aware_scaling: Whether to use phase-aware scaling
            auto_balance_components: Whether to auto-balance reward components
            reward_clip_range: Range for clipping rewards
            reward_scaling_window: Window size for reward statistics
            device: Device to use for training
            name: Name of agent
            priority: Priority of the agent (higher means more important)
        """
        # Set default device if not provided
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Set default reward clip range if not provided
        if reward_clip_range is None:
            reward_clip_range = (-10.0, 10.0)
            
        super().__init__(
            name=name,
            hyperparameter_key="loss_params",
            shared_state_manager=shared_state_manager,
            state_dim=state_dim,
            action_dim=3,  # lambda_ce, lambda_dice, focal_gamma
            action_space=(-1.0, 1.0),
            hidden_dim=hidden_dim,
            update_frequency=update_frequency,
            patience=patience,
            cooldown=cooldown,
            log_dir=log_dir,
            verbose=verbose,
            device=device,
            eligibility_trace_length=eligibility_trace_length,
            n_step=n_step,
            stability_weight=stability_weight,
            generalization_weight=generalization_weight,
            efficiency_weight=efficiency_weight,
            use_adaptive_scaling=use_adaptive_scaling,
            use_phase_aware_scaling=use_phase_aware_scaling,
            auto_balance_components=auto_balance_components,
            reward_clip_range=reward_clip_range,
            reward_scaling_window=reward_scaling_window,
            priority=priority
        )
        
        # Loss function specific parameters
        self.min_lambda_ce = min_lambda_ce
        self.max_lambda_ce = max_lambda_ce
        self.min_lambda_dice = min_lambda_dice
        self.max_lambda_dice = max_lambda_dice
        self.min_focal_gamma = min_focal_gamma
        self.max_focal_gamma = max_focal_gamma
        self.metrics_to_track = metrics_to_track
        self.use_enhanced_state = use_enhanced_state
        
        # Initialize loss parameters
        self.current_params = {
            "lambda_ce": initial_lambda_ce,
            "lambda_dice": initial_lambda_dice,
            "focal_gamma": initial_focal_gamma
        }
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, self.current_params)
        
        # Add epochs_since_update attribute
        self.epochs_since_update = 0
        
        # Log initialization
        logging.info(f"Initialized with loss_params={self.current_params}")
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        The state includes:
        - Enhanced metrics features (if enabled)
        - Recent history of tracked metrics
        - Current loss function parameters (normalized)
        - Component-wise loss values if available
        - Epochs since last update
        - Training progress (current epoch / total epochs)
        - Overfitting signals
        
        Returns:
            State representation as numpy array
        """
        state_components = []
        
        # Use enhanced state representation if available and enabled
        if self.use_enhanced_state:
            # Get enhanced state features from the metric processor
            enhanced_features = self.shared_state_manager.get_enhanced_state_vector(self.metrics_to_track)
            
            if len(enhanced_features) > 0:
                state_components.extend(enhanced_features)
                
                # Add overfitting signals
                overfitting_signals = self.shared_state_manager.get_overfitting_signals()
                if overfitting_signals:
                    state_components.extend(list(overfitting_signals.values()))
                
                # Add normalized current loss parameters
                normalized_lambda_ce = (self.current_params["lambda_ce"] - self.min_lambda_ce) / (self.max_lambda_ce - self.min_lambda_ce)
                state_components.append(normalized_lambda_ce)
                
                normalized_lambda_dice = (self.current_params["lambda_dice"] - self.min_lambda_dice) / (self.max_lambda_dice - self.min_lambda_dice)
                state_components.append(normalized_lambda_dice)
                
                normalized_gamma = (self.current_params["focal_gamma"] - self.min_focal_gamma) / (self.max_focal_gamma - self.min_focal_gamma)
                state_components.append(normalized_gamma)
                
                # Add component-wise loss ratio if available
                dice_loss = self.shared_state_manager.get_latest_metric("dice_loss")
                ce_loss = self.shared_state_manager.get_latest_metric("ce_loss")
                
                if dice_loss is not None and ce_loss is not None and (dice_loss + ce_loss) > 0:
                    dice_ratio = dice_loss / (dice_loss + ce_loss)
                    state_components.append(dice_ratio)
                else:
                    state_components.append(0.5)  # Default balanced ratio
                
                # Add epochs since last update
                state_components.append(self.epochs_since_update / 20.0)  # Normalize
                
                # Add training progress
                current_epoch = self.shared_state_manager.get_current_epoch()
                total_epochs = self.shared_state_manager.total_epochs
                if total_epochs is not None and total_epochs > 0:
                    progress = current_epoch / total_epochs
                else:
                    progress = 0.0
                state_components.append(progress)
                
                # Ensure state has correct dimension
                state = np.array(state_components, dtype=np.float32)
                
                # Pad or truncate to match state_dim
                if len(state) < self.state_dim:
                    state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
                elif len(state) > self.state_dim:
                    state = state[:self.state_dim]
                
                return state
        
        # Fall back to standard state representation if enhanced state is not available or disabled
        # Get metrics history
        metrics_history = {}
        for metric in self.metrics_to_track:
            metrics_history[metric] = self.shared_state_manager.get_metric_history(
                metric, window_size=5  # Smaller window to save space
            )
        
        # Calculate metrics statistics
        
        # Add recent metric values
        for metric in self.metrics_to_track:
            history = metrics_history[metric]
            if len(history) > 0:
                # Add latest value
                state_components.append(history[-1])
                
                # Add trend (difference between latest and oldest in window)
                if len(history) > 1:
                    state_components.append(history[-1] - history[0])
                else:
                    state_components.append(0.0)
            else:
                # If no history, pad with zeros
                state_components.extend([0.0, 0.0])
        
        # Add normalized current loss parameters
        normalized_lambda_ce = (self.current_params["lambda_ce"] - self.min_lambda_ce) / (self.max_lambda_ce - self.min_lambda_ce)
        state_components.append(normalized_lambda_ce)
        
        normalized_lambda_dice = (self.current_params["lambda_dice"] - self.min_lambda_dice) / (self.max_lambda_dice - self.min_lambda_dice)
        state_components.append(normalized_lambda_dice)
        
        normalized_gamma = (self.current_params["focal_gamma"] - self.min_focal_gamma) / (self.max_focal_gamma - self.min_focal_gamma)
        state_components.append(normalized_gamma)
        
        # Add component-wise loss ratio if available
        dice_loss = self.shared_state_manager.get_latest_metric("dice_loss")
        ce_loss = self.shared_state_manager.get_latest_metric("ce_loss")
        
        if dice_loss is not None and ce_loss is not None and (dice_loss + ce_loss) > 0:
            dice_ratio = dice_loss / (dice_loss + ce_loss)
            state_components.append(dice_ratio)
        else:
            state_components.append(0.5)  # Default balanced ratio
        
        # Add epochs since last update
        state_components.append(self.epochs_since_update / 20.0)  # Normalize
        
        # Add training progress
        current_epoch = self.shared_state_manager.get_current_epoch()
        total_epochs = self.shared_state_manager.total_epochs
        if total_epochs is not None and total_epochs > 0:
            progress = current_epoch / total_epochs
        else:
            progress = 0.0
        state_components.append(progress)
        
        # Ensure state has correct dimension
        state = np.array(state_components, dtype=np.float32)
        
        # Pad or truncate to match state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return state
    
    def action_to_hyperparameter(self, action):
        """
        Convert normalized actions to actual loss function parameters.
        
        Args:
            action: Normalized actions from SAC (-1.0 to 1.0) for each parameter
            
        Returns:
            Dictionary of new loss function parameters
        """
        new_params = {}
        
        # Handle both scalar and array/list actions
        if not isinstance(action, (list, np.ndarray)):
            # If action is a scalar, use default values with small adjustments
            factor_ce = 1.0
            factor_dice = 1.0
            factor_gamma = 1.0
        else:
            # If action is an array/list, use the values
            factor_ce = 2.0 ** action[0] if len(action) > 0 else 1.0
            factor_dice = 2.0 ** action[1] if len(action) > 1 else 1.0
            factor_gamma = 2.0 ** action[2] if len(action) > 2 else 1.0
        
        # Convert actions based on loss type
        new_lambda_ce = self.current_params["lambda_ce"] * factor_ce
        new_lambda_ce = np.clip(new_lambda_ce, self.min_lambda_ce, self.max_lambda_ce)
        new_params["lambda_ce"] = new_lambda_ce
        
        new_lambda_dice = self.current_params["lambda_dice"] * factor_dice
        new_lambda_dice = np.clip(new_lambda_dice, self.min_lambda_dice, self.max_lambda_dice)
        new_params["lambda_dice"] = new_lambda_dice
        
        if "focal_gamma" in self.current_params:
            new_focal_gamma = self.current_params["focal_gamma"] * factor_gamma
            new_focal_gamma = np.clip(new_focal_gamma, self.min_focal_gamma, self.max_focal_gamma)
            new_params["focal_gamma"] = new_focal_gamma
        
        return new_params
    
    def _process_action(self, action):
        """
        Process the action from SAC to get the new loss function parameters.
        
        Args:
            action: Action from SAC
            
        Returns:
            New loss function parameters
        """
        # Handle both scalar and array/list actions
        return self.action_to_hyperparameter(action)
    
    def _apply_action(self, new_params: Dict[str, float]) -> None:
        """
        Apply the new loss function parameters.
        
        Args:
            new_params: New loss function parameters
        """
        # Update current value
        old_params = self.current_params.copy()
        self.current_params = new_params
        
        # Update shared state
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, self.current_params)
        
        # Log update
        logging.info(f"Updated loss_params: {old_params} -> {self.current_params}")
    
    def select_action(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        Select an action based on the current state.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Action value (loss function parameters) or None if no action should be taken
        """
        if not self.should_update(epoch):
            return None
            
        state = self.get_state_representation()
        action = self.sac.select_action(state)
        processed_action = self._process_action(action)
        
        return processed_action
    
    def _get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        Returns:
            State representation as numpy array

        return self.get_state_representation()
    
    def get_current_params(self) -> Dict[str, float]:
        """
        Get the current loss function parameters.
        
        Returns:
            Current loss function parameters
        """
        return self.current_params.copy()
