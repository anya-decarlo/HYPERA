#!/usr/bin/env python
# Weight Decay Agent for Multi-Agent Hyperparameter Optimization

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

from .base_agent import BaseHyperparameterAgent
from .shared_state import SharedStateManager

class WeightDecayAgent(BaseHyperparameterAgent):
    """
    Specialized agent for optimizing the weight decay hyperparameter.
    
    This agent uses SAC to learn the optimal weight decay adjustment policy
    based on observed training metrics. It can dynamically adjust the weight
    decay during training to improve generalization and reduce overfitting.
    """
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        initial_weight_decay: float = 1e-5,
        min_weight_decay: float = 1e-8,
        max_weight_decay: float = 1e-2,
        update_frequency: int = 5,  # Less frequent updates than learning rate
        patience: int = 5,
        cooldown: int = 10,
        log_dir: str = "results",
        verbose: bool = True,
        metrics_to_track: List[str] = ["loss", "val_loss", "dice_score", "train_val_gap"],
        state_dim: int = 20,
        hidden_dim: int = 256,
        use_enhanced_state: bool = True,
        eligibility_trace_length: int = 10,
        n_step: int = 3,
        stability_weight: float = 0.3,
        generalization_weight: float = 0.4,
        efficiency_weight: float = 0.3
    ):
        """
        Initialize the weight decay agent.
        
        Args:
            shared_state_manager: Manager for shared state across agents
            initial_weight_decay: Initial weight decay value
            min_weight_decay: Minimum allowed weight decay
            max_weight_decay: Maximum allowed weight decay
            update_frequency: How often to consider updates (in epochs)
            patience: Epochs to wait before considering action
            cooldown: Epochs to wait after an action
            log_dir: Directory for saving logs
            verbose: Whether to print verbose output
            metrics_to_track: List of metrics to include in state representation
            state_dim: Dimension of state representation
            hidden_dim: Hidden dimension for SAC networks
            use_enhanced_state: Whether to use enhanced state representation
            eligibility_trace_length: Length of eligibility traces for reward calculation
            n_step: Number of steps for n-step returns
            stability_weight: Weight for stability component in reward
            generalization_weight: Weight for generalization component in reward
            efficiency_weight: Weight for efficiency component in reward
        """
        super().__init__(
            name="weight_decay",
            hyperparameter_key="weight_decay",
            shared_state_manager=shared_state_manager,
            state_dim=state_dim,
            action_dim=1,  # Weight decay adjustment is a 1D action
            action_space=(-1.0, 1.0),  # Normalized action space
            hidden_dim=hidden_dim,
            update_frequency=update_frequency,
            patience=patience,
            cooldown=cooldown,
            log_dir=log_dir,
            verbose=verbose,
            eligibility_trace_length=eligibility_trace_length,
            n_step=n_step,
            stability_weight=stability_weight,
            generalization_weight=generalization_weight,
            efficiency_weight=efficiency_weight
        )
        
        # Weight decay specific parameters
        self.min_weight_decay = min_weight_decay
        self.max_weight_decay = max_weight_decay
        self.metrics_to_track = metrics_to_track
        self.use_enhanced_state = use_enhanced_state
        
        # Initialize weight decay
        self.current_weight_decay = initial_weight_decay
        self.shared_state_manager.update_hyperparameter(self.hyperparameter_key, self.current_weight_decay)
        
        # Log initialization
        self.log(f"Initialized with weight_decay={self.current_weight_decay}")
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        The state includes:
        - Enhanced metrics features (if enabled)
        - Recent history of tracked metrics
        - Current weight decay (normalized)
        - Train-validation gap (to detect overfitting)
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
                
                # Add overfitting signals - especially important for weight decay agent
                overfitting_signals = self.shared_state_manager.get_overfitting_signals()
                if overfitting_signals:
                    state_components.extend(list(overfitting_signals.values()))
                
                # Add normalized current weight decay
                normalized_wd = (np.log10(self.current_weight_decay) - np.log10(self.min_weight_decay)) / (np.log10(self.max_weight_decay) - np.log10(self.min_weight_decay))
                state_components.append(normalized_wd)
                
                # Add epochs since last update
                state_components.append(self.epochs_since_update / 10.0)  # Normalize
                
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
            if metric == "train_val_gap":
                # Calculate train-val gap if not directly provided
                loss_history = self.shared_state_manager.get_metric_history(
                    "loss", window_size=self.state_dim // len(self.metrics_to_track)
                )
                val_loss_history = self.shared_state_manager.get_metric_history(
                    "val_loss", window_size=self.state_dim // len(self.metrics_to_track)
                )
                
                # Ensure both histories have the same length
                min_len = min(len(loss_history), len(val_loss_history))
                if min_len > 0:
                    gap_history = [loss_history[i] - val_loss_history[i] for i in range(min_len)]
                else:
                    gap_history = []
                
                metrics_history[metric] = gap_history
            else:
                metrics_history[metric] = self.shared_state_manager.get_metric_history(
                    metric, window_size=self.state_dim // len(self.metrics_to_track)
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
        
        # Add normalized current weight decay
        normalized_wd = (np.log10(self.current_weight_decay) - np.log10(self.min_weight_decay)) / (np.log10(self.max_weight_decay) - np.log10(self.min_weight_decay))
        state_components.append(normalized_wd)
        
        # Add epochs since last update
        state_components.append(self.epochs_since_update / 10.0)  # Normalize
        
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
    
    def action_to_hyperparameter(self, action: float) -> float:
        """
        Convert normalized action to actual weight decay value.
        
        Args:
            action: Normalized action from SAC (-1.0 to 1.0)
            
        Returns:
            New weight decay value
        """
        # Convert action to multiplicative factor
        # Action -1.0 -> divide by 5
        # Action 0.0 -> no change
        # Action 1.0 -> multiply by 5
        factor = 5.0 ** action
        
        # Apply factor to current weight decay
        new_weight_decay = self.current_weight_decay * factor
        
        # Clip to valid range
        new_weight_decay = np.clip(new_weight_decay, self.min_weight_decay, self.max_weight_decay)
        
        return new_weight_decay
    
    def update_hyperparameter(self, action: float) -> Dict[str, Any]:
        """
        Update the weight decay based on the agent's action.
        
        Args:
            action: Normalized action from SAC (-1.0 to 1.0)
            
        Returns:
            Dictionary with update information
        """
        # Convert action to weight decay
        new_weight_decay = self.action_to_hyperparameter(action)
        
        # Calculate relative change
        relative_change = new_weight_decay / self.current_weight_decay
        
        # Update current value
        old_weight_decay = self.current_weight_decay
        self.current_weight_decay = new_weight_decay
        
        # Update shared state
        self.shared_state_manager.update_hyperparameter(self.hyperparameter_key, self.current_weight_decay)
        
        # Log update
        self.log(f"Updated weight_decay: {old_weight_decay:.8f} -> {self.current_weight_decay:.8f} (factor: {relative_change:.2f})")
        
        # Return update info
        return {
            "old_value": old_weight_decay,
            "new_value": self.current_weight_decay,
            "relative_change": relative_change,
            "hyperparameter": self.hyperparameter_key
        }
    
    def _process_action(self, action: np.ndarray) -> float:
        """
        Process the action from SAC to get the new weight decay.
        
        Args:
            action: Action from SAC
            
        Returns:
            New weight decay value
        """
        return self.action_to_hyperparameter(action[0])
    
    def _apply_action(self, new_weight_decay: float) -> None:
        """
        Apply the new weight decay.
        
        Args:
            new_weight_decay: New weight decay value
        """
        # Update current value
        old_weight_decay = self.current_weight_decay
        self.current_weight_decay = new_weight_decay
        
        # Update shared state
        self.shared_state_manager.update_hyperparameter(self.hyperparameter_key, self.current_weight_decay)
        
        # Log update
        self.log(f"Updated weight_decay: {old_weight_decay:.8f} -> {self.current_weight_decay:.8f}")
    
    def _get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        Returns:
            State representation as numpy array
        """
        return self.get_state_representation()
