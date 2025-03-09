#!/usr/bin/env python
# Normalization Agent for Multi-Agent Hyperparameter Optimization

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

from .base_agent import BaseHyperparameterAgent
from .shared_state import SharedStateManager

class NormalizationAgent(BaseHyperparameterAgent):
    """
    Specialized agent for selecting the optimal normalization type.
    
    This agent uses SAC to learn which normalization method works best
    for the current training stage. It can switch between different
    normalization types like batch norm, instance norm, and layer norm.
    """
    
    # Define normalization types and their corresponding indices
    NORM_TYPES = {
        0: "batch",
        1: "instance",
        2: "layer",
        3: "group"
    }
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        initial_norm_type: str = "instance",
        update_frequency: int = 20,  # Very infrequent updates
        patience: int = 10,
        cooldown: int = 30,
        log_dir: str = "results",
        verbose: bool = True,
        metrics_to_track: List[str] = ["loss", "val_loss", "dice_score", "gradient_norm"],
        state_dim: int = 20,
        hidden_dim: int = 256,
        use_enhanced_state: bool = True,
        eligibility_trace_length: int = 10,
        n_step: int = 3,
        stability_weight: float = 0.4,  # Higher stability weight for normalization
        generalization_weight: float = 0.4,
        efficiency_weight: float = 0.2,
        use_adaptive_scaling: bool = True,
        use_phase_aware_scaling: bool = True,
        auto_balance_components: bool = True,
        reward_clip_range: Optional[Tuple[float, float]] = None,
        reward_scaling_window: int = 100,
        device: Optional[torch.device] = None,
        name: str = "normalization_agent",
        priority: int = 0
    ):
        """
        Initialize the normalization agent.
        
        Args:
            shared_state_manager: Manager for shared state across agents
            initial_norm_type: Initial normalization type ('batch', 'instance', 'layer', or 'group')
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
            use_adaptive_scaling: Whether to use adaptive scaling
            use_phase_aware_scaling: Whether to use phase-aware scaling
            auto_balance_components: Whether to auto-balance components
            reward_clip_range: Range for clipping rewards
            reward_scaling_window: Window size for reward scaling
            device: Device for computations
            name: Name of the agent
            priority: Priority of the agent (higher means more important)
        """
        # Validate initial norm type
        if initial_norm_type not in self.NORM_TYPES.values():
            raise ValueError(f"Invalid normalization type: {initial_norm_type}. "
                             f"Must be one of {list(self.NORM_TYPES.values())}")
        
        # Map norm types to indices for action space
        self.norm_type_to_idx = {norm_type: idx for idx, norm_type in self.NORM_TYPES.items()}
        
        super().__init__(
            name=name,
            hyperparameter_key="norm_type",
            shared_state_manager=shared_state_manager,
            state_dim=state_dim,
            action_dim=1,  # Discrete action space mapped to continuous
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
            efficiency_weight=efficiency_weight,
            use_adaptive_scaling=use_adaptive_scaling,
            use_phase_aware_scaling=use_phase_aware_scaling,
            auto_balance_components=auto_balance_components,
            reward_clip_range=reward_clip_range,
            reward_scaling_window=reward_scaling_window,
            device=device,
            priority=priority
        )
        
        # Normalization specific parameters
        self.metrics_to_track = metrics_to_track
        self.use_enhanced_state = use_enhanced_state
        
        # Initialize normalization type
        self.current_norm_type = initial_norm_type
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, self.current_norm_type)
        
        # Add epochs_since_update attribute
        self.epochs_since_update = 0
        
        # Log initialization
        logging.info(f"Initialized with norm_type={self.current_norm_type}")
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        The state includes:
        - Enhanced metrics features (if enabled)
        - Recent history of tracked metrics
        - Current normalization type (one-hot encoded)
        - Batch size (affects batch norm behavior)
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
                
                # Add one-hot encoding of current normalization type
                for norm_type in self.NORM_TYPES.values():
                    state_components.append(1.0 if self.current_norm_type == norm_type else 0.0)
                
                # Add batch size (normalized) if available
                batch_size = self.shared_state_manager.get_hyperparameter("batch_size")
                if batch_size is not None:
                    # Normalize batch size to [0, 1] range assuming max batch size of 32
                    state_components.append(min(batch_size / 32.0, 1.0))
                else:
                    state_components.append(0.5)  # Default normalized batch size
                
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
        
        # Add one-hot encoding of current normalization type
        for norm_type in self.NORM_TYPES.values():
            state_components.append(1.0 if self.current_norm_type == norm_type else 0.0)
        
        # Add batch size (normalized) if available
        batch_size = self.shared_state_manager.get_hyperparameter("batch_size")
        if batch_size is not None:
            # Normalize batch size to [0, 1] range assuming max batch size of 32
            state_components.append(min(batch_size / 32.0, 1.0))
        else:
            state_components.append(0.5)  # Default normalized batch size
        
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
        Convert normalized action to normalization type.
        
        Args:
            action: Normalized action from SAC (-1.0 to 1.0)
            
        Returns:
            New normalization type
        """
        # Handle both scalar and array/list actions
        if isinstance(action, (list, np.ndarray)) and len(action) > 0:
            action_value = action[0]
        else:
            action_value = action
            
        # Map continuous action to discrete norm type index
        # Scale from [-1, 1] to [0, len(NORM_TYPES)-1]
        scaled_action = (action_value + 1.0) / 2.0 * (len(self.NORM_TYPES) - 1)
        norm_idx = int(np.round(scaled_action))
        
        # Clip to valid range
        norm_idx = np.clip(norm_idx, 0, len(self.NORM_TYPES) - 1)
        
        # Convert index to norm type
        return self.NORM_TYPES[norm_idx]
    
    def update_hyperparameter(self, action: float) -> Dict[str, Any]:
        """
        Update the normalization type based on the agent's action.
        
        Args:
            action: Normalized action from SAC (-1.0 to 1.0)
            
        Returns:
            Dictionary with update information
        """
        # Convert action to normalization type
        new_norm_type = self.action_to_hyperparameter(action)
        
        # Check if there's an actual change
        if new_norm_type == self.current_norm_type:
            self.log(f"No change in norm_type: keeping {self.current_norm_type}")
            return {
                "old_value": self.current_norm_type,
                "new_value": self.current_norm_type,
                "relative_change": 0.0,
                "hyperparameter": self.hyperparameter_key,
                "changed": False
            }
        
        # Update current value
        old_norm_type = self.current_norm_type
        self.current_norm_type = new_norm_type
        
        # Update shared state
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, self.current_norm_type)
        
        # Log update
        logging.info(f"Updated norm_type: {old_norm_type} -> {self.current_norm_type}")
        
        # Return update info
        return {
            "old_value": old_norm_type,
            "new_value": self.current_norm_type,
            "relative_change": 1.0,  # Binary change indicator for categorical parameter
            "hyperparameter": self.hyperparameter_key,
            "changed": True
        }
    
    def _get_state_representation(self) -> np.ndarray:
        """
        Get the state representation for the agent.
        
        Returns:
            State representation as numpy array
        """
        return self.get_state_representation()
        
    def _process_action(self, action):
        """
        Process the continuous action from the agent to get a discrete action.
        
        Args:
            action: Continuous action from the agent
            
        Returns:
            Discrete action index
        """
        # Map continuous action to discrete action space
        # Scale from [-1, 1] to [0, 3]
        # Handle both scalar and array/list actions
        if isinstance(action, (list, np.ndarray)):
            if len(action) > 0:
                # Ensure action[0] is a number
                action_value = float(action[0]) if isinstance(action[0], (int, float, np.number)) else 0.0
            else:
                action_value = 0.0
        else:
            # Ensure action is a number
            action_value = float(action) if isinstance(action, (int, float, np.number)) else 0.0
            
        scaled_action = (action_value + 1.0) / 2.0 * (len(self.NORM_TYPES) - 1)
        discrete_action = int(np.round(scaled_action))
        discrete_action = np.clip(discrete_action, 0, len(self.NORM_TYPES) - 1)
        return discrete_action
        
    def _apply_action(self, new_norm_type: str) -> None:
        """
        Apply the new normalization type.
        
        Args:
            new_norm_type: New normalization type
        """
        self.current_norm_type = new_norm_type
        self.shared_state_manager.set_hyperparameter(self.hyperparameter_key, new_norm_type)
        
        if self.verbose:
            logging.info(f"Normalization type updated to {new_norm_type}")
    
    def select_action(self, epoch: int) -> Optional[str]:
        """
        Select an action based on the current state.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Action value (normalization type) or None if no action should be taken
        """
        if not self.should_update(epoch):
            return None
            
        state = self.get_state_representation()
        action = self.sac.select_action(state)
        processed_action = self._process_action(action)
        
        return self.NORM_TYPES[processed_action]
    
    def get_current_norm_type(self) -> str:
        """

        
        Returns:
            Current normalization type
        """
        return self.current_norm_type
