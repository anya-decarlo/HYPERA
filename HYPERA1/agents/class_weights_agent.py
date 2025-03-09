#!/usr/bin/env python
# Class Weights Agent for Multi-Agent Hyperparameter Optimization

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

from .base_agent import BaseHyperparameterAgent
from .shared_state import SharedStateManager

class ClassWeightsAgent(BaseHyperparameterAgent):
    """
    Specialized agent for optimizing class weights in segmentation loss functions.
    
    This agent uses SAC to learn the optimal class weight adjustment policy
    based on observed training metrics. It can dynamically adjust the weights
    for different classes to address class imbalance and improve performance
    on minority classes.
    """
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        initial_class_weights: Optional[Dict[int, float]] = None,
        min_weight: float = 0.5,
        max_weight: float = 5.0,
        num_classes: int = 2,
        update_frequency: int = 10,  # Less frequent updates
        patience: int = 8,
        cooldown: int = 15,
        log_dir: str = "results",
        verbose: bool = True,
        metrics_to_track: List[str] = ["loss", "val_loss", "dice_score", "class_dice"],
        state_dim: int = 24,
        hidden_dim: int = 256,
        use_enhanced_state: bool = True,
        eligibility_trace_length: int = 10,
        n_step: int = 3,
        stability_weight: float = 0.3,
        generalization_weight: float = 0.4,
        efficiency_weight: float = 0.3
    ):
        """
        Initialize the class weights agent.
        
        Args:
            shared_state_manager: Manager for shared state across agents
            initial_class_weights: Initial class weights dictionary {class_id: weight}
            min_weight: Minimum allowed class weight
            max_weight: Maximum allowed class weight
            num_classes: Number of classes in the segmentation task
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
        # Initialize default class weights if not provided
        if initial_class_weights is None:
            initial_class_weights = {i: 1.0 for i in range(num_classes)}
        
        super().__init__(
            name="class_weights",
            hyperparameter_key="class_weights",
            shared_state_manager=shared_state_manager,
            state_dim=state_dim,
            action_dim=num_classes,  # One action dimension per class
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
        
        # Class weights specific parameters
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_classes = num_classes
        self.metrics_to_track = metrics_to_track
        self.use_enhanced_state = use_enhanced_state
        
        # Initialize class weights
        self.current_class_weights = initial_class_weights.copy()
        self.shared_state_manager.update_hyperparameter(self.hyperparameter_key, self.current_class_weights)
        
        # Log initialization
        self.log(f"Initialized with class_weights={self.current_class_weights}")
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        The state includes:
        - Enhanced metrics features (if enabled)
        - Recent history of tracked metrics
        - Current class weights (normalized)
        - Per-class dice scores (if available)
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
                
                # Add normalized current class weights
                for weight in self.current_class_weights.values():
                    normalized_weight = (weight - self.min_weight) / (self.max_weight - self.min_weight)
                    state_components.append(normalized_weight)
                
                # Add per-class dice scores if available
                for class_idx in range(self.num_classes):
                    class_dice_key = f"dice_class_{class_idx}"
                    class_dice = self.shared_state_manager.get_latest_metric(class_dice_key)
                    if class_dice is not None:
                        state_components.append(class_dice)
                    else:
                        # If not available, use overall dice
                        overall_dice = self.shared_state_manager.get_latest_metric("dice_score")
                        state_components.append(overall_dice if overall_dice is not None else 0.0)
                
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
            if metric == "class_dice":
                # For class-specific dice scores, we'll handle them separately
                continue
            else:
                metrics_history[metric] = self.shared_state_manager.get_metric_history(
                    metric, window_size=5  # Smaller window to save space
                )
        
        # Calculate metrics statistics
        
        # Add recent metric values
        for metric in self.metrics_to_track:
            if metric == "class_dice":
                continue
                
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
        
        # Add normalized current class weights
        for weight in self.current_class_weights.values():
            normalized_weight = (weight - self.min_weight) / (self.max_weight - self.min_weight)
            state_components.append(normalized_weight)
        
        # Add per-class dice scores if available
        for class_idx in range(self.num_classes):
            class_dice_key = f"dice_class_{class_idx}"
            class_dice = self.shared_state_manager.get_latest_metric(class_dice_key)
            if class_dice is not None:
                state_components.append(class_dice)
            else:
                # If not available, use overall dice
                overall_dice = self.shared_state_manager.get_latest_metric("dice_score")
                state_components.append(overall_dice if overall_dice is not None else 0.0)
        
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
    
    def action_to_hyperparameter(self, action: np.ndarray) -> np.ndarray:
        """
        Convert normalized actions to actual class weight values.
        
        Args:
            action: Normalized actions from SAC (-1.0 to 1.0) for each class
            
        Returns:
            New class weight values
        """
        # Convert actions to multiplicative factors
        # Action -1.0 -> divide by 2
        # Action 0.0 -> no change
        # Action 1.0 -> multiply by 2
        factors = 2.0 ** action
        
        # Apply factors to current class weights
        new_weights = np.array(list(self.current_class_weights.values())) * factors
        
        # Clip to valid range
        new_weights = np.clip(new_weights, self.min_weight, self.max_weight)
        
        # Normalize weights to have mean of 1.0
        # This ensures the overall loss scale doesn't change dramatically
        if np.mean(new_weights) > 0:
            new_weights = new_weights * (self.num_classes / np.sum(new_weights))
        
        return new_weights
    
    def update_hyperparameter(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Update the class weights based on the agent's actions.
        
        Args:
            action: Normalized actions from SAC (-1.0 to 1.0) for each class
            
        Returns:
            Dictionary with update information
        """
        # Convert action to class weights
        new_weights = self.action_to_hyperparameter(action)
        
        # Calculate relative changes
        relative_changes = new_weights / np.array(list(self.current_class_weights.values()))
        
        # Update current values
        old_weights = self.current_class_weights.copy()
        self.current_class_weights = {i: weight for i, weight in enumerate(new_weights)}
        
        # Update shared state
        self.shared_state_manager.update_hyperparameter(self.hyperparameter_key, self.current_class_weights)
        
        # Log update
        self.log(f"Updated class_weights: {old_weights} -> {self.current_class_weights}")
        self.log(f"Relative changes: {relative_changes.tolist()}")
        
        # Return update info
        return {
            "old_value": old_weights,
            "new_value": self.current_class_weights,
            "relative_change": relative_changes.tolist(),
            "hyperparameter": self.hyperparameter_key
        }
    
    def _process_action(self, action: np.ndarray) -> Dict[int, float]:
        """
        Process the action from SAC to get the new class weights.
        
        Args:
            action: Action from SAC
            
        Returns:
            New class weights dictionary
        """
        return self.action_to_hyperparameter(action)
    
    def _apply_action(self, new_class_weights: Dict[int, float]) -> None:
        """
        Apply the new class weights.
        
        Args:
            new_class_weights: New class weights dictionary
        """
        # Update current value
        old_class_weights = self.current_class_weights.copy()
        self.current_class_weights = new_class_weights
        
        # Update shared state
        self.shared_state_manager.update_hyperparameter(self.hyperparameter_key, self.current_class_weights)
        
        # Log update
        self.log(f"Updated class_weights: {old_class_weights} -> {self.current_class_weights}")
    
    def _get_state_representation(self) -> np.ndarray:
        """
        Get the current state representation for the agent.
        
        Returns:
            State representation as numpy array
        """
        return self.get_state_representation()
