#!/usr/bin/env python
# Adaptive Weight Manager - Dynamically adjusts weights for the multi-objective reward function

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

class AdaptiveWeightManager:
    """
    Manages the weights for different components of the multi-objective reward function.
    
    This class dynamically adjusts the weights of different reward components based on
    the current phase of training (exploration, exploitation, fine-tuning) and other factors.
    
    Attributes:
        initial_weights (Dict[str, float]): Initial weights for each reward component
        current_weights (Dict[str, float]): Current weights for each reward component
        phase_detection_enabled (bool): Whether to automatically detect training phase
        current_phase (str): Current training phase ('exploration', 'exploitation', 'fine_tuning')
        max_epochs (int): Maximum number of epochs for training
        exploration_ratio (float): Ratio of epochs for exploration phase
        exploitation_ratio (float): Ratio of epochs for exploitation phase
        logger (logging.Logger): Logger for the weight manager
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        max_epochs: int = 100,
        exploration_ratio: float = 0.3,
        exploitation_ratio: float = 0.5,
        phase_detection_enabled: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the adaptive weight manager.
        
        Args:
            initial_weights: Initial weights for each reward component
            max_epochs: Maximum number of epochs for training
            exploration_ratio: Ratio of epochs for exploration phase
            exploitation_ratio: Ratio of epochs for exploitation phase
            phase_detection_enabled: Whether to automatically detect training phase
            verbose: Whether to print verbose output
        """
        # Set default initial weights if not provided
        if initial_weights is None:
            self.initial_weights = {
                "dice": 1.0,           # Regional overlap (Dice score)
                "boundary": 0.2,       # Boundary accuracy (Hausdorff distance)
                "obj_f1": 0.8,         # Precision-recall balance (F1-score for cell detection)
                "shape": 0.1,          # Compactness & shape regularization
                "fg_bg_balance": 0.3   # Foreground-background balance
            }
        else:
            self.initial_weights = initial_weights
        
        # Initialize current weights with initial weights
        self.current_weights = self.initial_weights.copy()
        
        # Set training parameters
        self.max_epochs = max_epochs
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.phase_detection_enabled = phase_detection_enabled
        
        # Initialize training phase
        self.current_phase = "exploration"
        
        # Set up logging
        self.logger = logging.getLogger("AdaptiveWeightManager")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Initialize phase transition epochs
        self.exploration_end = int(self.max_epochs * self.exploration_ratio)
        self.exploitation_end = int(self.max_epochs * (self.exploration_ratio + self.exploitation_ratio))
        
        # Initialize metrics history for adaptive adjustment
        self.metrics_history = {
            "dice": [],
            "boundary": [],
            "obj_f1": [],
            "shape": [],
            "fg_bg_balance": []
        }
        
        # Log initialization
        self.logger.info(f"Initialized AdaptiveWeightManager with initial weights: {self.initial_weights}")
        self.logger.info(f"Phase transitions: Exploration end: {self.exploration_end}, Exploitation end: {self.exploitation_end}")
    
    def update_weights(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update the weights based on the current epoch and metrics.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of current metrics
            
        Returns:
            Updated weights
        """
        # Update metrics history
        for metric_name, metric_value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(metric_value)
        
        # Detect training phase if enabled
        if self.phase_detection_enabled:
            self._detect_phase(epoch)
        
        # Update weights based on current phase
        if self.current_phase == "exploration":
            self._update_weights_exploration()
        elif self.current_phase == "exploitation":
            self._update_weights_exploitation()
        elif self.current_phase == "fine_tuning":
            self._update_weights_fine_tuning()
        
        # Log weight update
        self.logger.info(f"Updated weights for phase {self.current_phase}: {self.current_weights}")
        
        return self.current_weights
    
    def _detect_phase(self, epoch: int) -> None:
        """
        Detect the current training phase based on epoch and metrics.
        
        Args:
            epoch: Current epoch
        """
        # Simple phase detection based on epoch
        if epoch < self.exploration_end:
            self.current_phase = "exploration"
        elif epoch < self.exploitation_end:
            self.current_phase = "exploitation"
        else:
            self.current_phase = "fine_tuning"
        
        # Advanced phase detection based on metrics (if available)
        if len(self.metrics_history["dice"]) > 10:
            # Calculate improvement rate for Dice score
            recent_dice = self.metrics_history["dice"][-10:]
            improvement_rate = (recent_dice[-1] - recent_dice[0]) / max(1e-6, recent_dice[0])
            
            # Override phase based on improvement rate
            if improvement_rate > 0.1:
                # Still in rapid improvement phase
                self.current_phase = "exploration"
            elif improvement_rate > 0.01:
                # Moderate improvement phase
                self.current_phase = "exploitation"
            else:
                # Slow improvement phase
                self.current_phase = "fine_tuning"
    
    def _update_weights_exploration(self) -> None:
        """
        Update weights for the exploration phase.
        
        In the exploration phase, we prioritize regional overlap (Dice score) and
        precision-recall balance (F1-score) to get a good initial segmentation.
        """
        # Prioritize Dice score and F1-score
        self.current_weights["dice"] = self.initial_weights["dice"] * 1.5
        self.current_weights["obj_f1"] = self.initial_weights["obj_f1"] * 1.2
        
        # Reduce weights for boundary accuracy and shape regularization
        self.current_weights["boundary"] = self.initial_weights["boundary"] * 0.5
        self.current_weights["shape"] = self.initial_weights["shape"] * 0.5
        
        # Keep foreground-background balance weight as is
        self.current_weights["fg_bg_balance"] = self.initial_weights["fg_bg_balance"]
    
    def _update_weights_exploitation(self) -> None:
        """
        Update weights for the exploitation phase.
        
        In the exploitation phase, we start to increase the importance of boundary
        accuracy and shape regularization to refine the segmentation.
        """
        # Gradually reduce Dice score weight
        self.current_weights["dice"] = self.initial_weights["dice"] * 1.0
        
        # Keep F1-score weight as is
        self.current_weights["obj_f1"] = self.initial_weights["obj_f1"]
        
        # Increase boundary accuracy weight
        self.current_weights["boundary"] = self.initial_weights["boundary"] * 1.5
        
        # Slightly increase shape regularization weight
        self.current_weights["shape"] = self.initial_weights["shape"] * 1.2
        
        # Keep foreground-background balance weight as is
        self.current_weights["fg_bg_balance"] = self.initial_weights["fg_bg_balance"]
    
    def _update_weights_fine_tuning(self) -> None:
        """
        Update weights for the fine-tuning phase.
        
        In the fine-tuning phase, we prioritize boundary accuracy and shape
        regularization to get a refined segmentation with realistic shapes.
        """
        # Reduce Dice score weight
        self.current_weights["dice"] = self.initial_weights["dice"] * 0.8
        
        # Keep F1-score weight as is
        self.current_weights["obj_f1"] = self.initial_weights["obj_f1"]
        
        # Significantly increase boundary accuracy weight
        self.current_weights["boundary"] = self.initial_weights["boundary"] * 2.0
        
        # Significantly increase shape regularization weight
        self.current_weights["shape"] = self.initial_weights["shape"] * 2.0
        
        # Keep foreground-background balance weight as is
        self.current_weights["fg_bg_balance"] = self.initial_weights["fg_bg_balance"]
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Get the current weights.
        
        Returns:
            Current weights
        """
        return self.current_weights
    
    def set_phase(self, phase: str) -> None:
        """
        Manually set the training phase.
        
        Args:
            phase: Training phase ('exploration', 'exploitation', 'fine_tuning')
        """
        if phase not in ["exploration", "exploitation", "fine_tuning"]:
            raise ValueError(f"Invalid phase: {phase}")
        
        self.current_phase = phase
        self.phase_detection_enabled = False
        
        # Update weights based on new phase
        if phase == "exploration":
            self._update_weights_exploration()
        elif phase == "exploitation":
            self._update_weights_exploitation()
        elif phase == "fine_tuning":
            self._update_weights_fine_tuning()
        
        self.logger.info(f"Manually set phase to {phase} with weights: {self.current_weights}")
    
    def enable_phase_detection(self) -> None:
        """
        Enable automatic phase detection.
        """
        self.phase_detection_enabled = True
        self.logger.info("Enabled automatic phase detection")
    
    def disable_phase_detection(self) -> None:
        """
        Disable automatic phase detection.
        """
        self.phase_detection_enabled = False
        self.logger.info("Disabled automatic phase detection")
    
    def set_max_epochs(self, max_epochs: int) -> None:
        """
        Set the maximum number of epochs.
        
        Args:
            max_epochs: Maximum number of epochs
        """
        self.max_epochs = max_epochs
        
        # Recalculate phase transition epochs
        self.exploration_end = int(self.max_epochs * self.exploration_ratio)
        self.exploitation_end = int(self.max_epochs * (self.exploration_ratio + self.exploitation_ratio))
        
        self.logger.info(f"Updated max epochs to {max_epochs}")
        self.logger.info(f"New phase transitions: Exploration end: {self.exploration_end}, Exploitation end: {self.exploitation_end}")
    
    def set_phase_ratios(self, exploration_ratio: float, exploitation_ratio: float) -> None:
        """
        Set the ratios for exploration and exploitation phases.
        
        Args:
            exploration_ratio: Ratio of epochs for exploration phase
            exploitation_ratio: Ratio of epochs for exploitation phase
        """
        if exploration_ratio + exploitation_ratio > 1.0:
            raise ValueError(f"Sum of ratios must be <= 1.0, got {exploration_ratio + exploitation_ratio}")
        
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        
        # Recalculate phase transition epochs
        self.exploration_end = int(self.max_epochs * self.exploration_ratio)
        self.exploitation_end = int(self.max_epochs * (self.exploration_ratio + self.exploitation_ratio))
        
        self.logger.info(f"Updated phase ratios: Exploration {exploration_ratio}, Exploitation {exploitation_ratio}")
        self.logger.info(f"New phase transitions: Exploration end: {self.exploration_end}, Exploitation end: {self.exploitation_end}")
    
    def reset_weights(self) -> None:
        """
        Reset weights to initial values.
        """
        self.current_weights = self.initial_weights.copy()
        self.logger.info(f"Reset weights to initial values: {self.current_weights}")
    
    def set_initial_weights(self, weights: Dict[str, float]) -> None:
        """
        Set new initial weights.
        
        Args:
            weights: New initial weights
        """
        self.initial_weights = weights
        self.current_weights = weights.copy()
        self.logger.info(f"Set new initial weights: {self.initial_weights}")
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """
        Get the history of weights for each component.
        
        Returns:
            Dictionary of weight histories
        """
        # Not implemented yet, would require storing weight history
        return {}
