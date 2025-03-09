#!/usr/bin/env python
# Shared State Manager for Multi-Agent Hyperparameter Optimization

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from collections import deque, defaultdict
import time

from .metric_processor import MetricProcessor

class SharedStateManager:
    """
    Manages shared state for multi-agent hyperparameter optimization.
    
    This class is responsible for:
    1. Tracking training metrics over time
    2. Storing current hyperparameter values
    3. Providing access to metrics and hyperparameters for agents
    4. Logging and visualizing metrics and hyperparameter changes
    5. Processing metrics to extract meaningful signals for agent state representation
    """
    
    def __init__(
        self,
        history_size: int = 100,
        log_dir: Optional[str] = None,
        verbose: bool = False,
        total_epochs: Optional[int] = None,
        enable_enhanced_metrics: bool = True,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50
    ):
        """
        Initialize the shared state manager.
        
        Args:
            history_size: Number of epochs to keep in history
            log_dir: Directory for saving logs and visualizations
            verbose: Whether to print verbose output
            total_epochs: Total number of epochs for training (for visualization)
            enable_enhanced_metrics: Whether to enable enhanced metric processing
            short_window: Size of short-term window for trend analysis
            medium_window: Size of medium-term window for trend analysis
            long_window: Size of long-term window for trend analysis
        """
        self.history_size = history_size
        self.log_dir = log_dir
        self.verbose = verbose
        self.total_epochs = total_epochs
        self.enable_enhanced_metrics = enable_enhanced_metrics
        
        # Initialize metrics history
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.epoch_history = deque(maxlen=history_size)
        
        # Initialize hyperparameter state
        self.hyperparameters = {}
        self.hyperparameter_history = defaultdict(lambda: deque(maxlen=history_size))
        
        # Initialize agent action history
        self.agent_actions = defaultdict(lambda: deque(maxlen=history_size))
        
        # Initialize metric processor for enhanced state representation
        if enable_enhanced_metrics:
            self.metric_processor = MetricProcessor(
                short_window=short_window,
                medium_window=medium_window,
                long_window=long_window,
                max_history_size=history_size,
                verbose=verbose
            )
        else:
            self.metric_processor = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        
        # Create log directory if needed
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
            # Set up file handler for logging
            log_file = os.path.join(log_dir, "shared_state.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def record_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Record training metrics for the current epoch.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metric names and values
        """
        self.epoch_history.append(epoch)
        
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
        
        # Update enhanced metrics if enabled
        if self.enable_enhanced_metrics and self.metric_processor:
            self.metric_processor.update(metrics)
        
        if self.verbose:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Epoch {epoch}: {metrics_str}")
    
    def get_metrics_history(self, metric_name: str, window: Optional[int] = None) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            window: Number of most recent epochs to include (None for all)
            
        Returns:
            List of metric values
        """
        if metric_name not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[metric_name])
        if window is not None:
            history = history[-window:]
        
        return history
    
    def get_metric_value(self, metric_name: str, default: float = 0.0) -> float:
        """
        Get the most recent value of a metric.
        
        Args:
            metric_name: Name of the metric
            default: Default value if metric not found
            
        Returns:
            Most recent value of the metric
        """
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return default
        
        return self.metrics_history[metric_name][-1]
    
    def get_latest_metric(self, metric_name: str, default: float = 0.0) -> float:
        """
        Alias for get_metric_value for backward compatibility.
        
        Args:
            metric_name: Name of the metric
            default: Default value if metric not found
            
        Returns:
            Most recent value of the metric
        """
        return self.get_metric_value(metric_name, default)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get the most recent values of all metrics.
        
        Returns:
            Dictionary of metric names and their most recent values
        """
        return {metric_name: self.get_metric_value(metric_name) 
                for metric_name in self.metrics_history.keys()}
    
    def set_hyperparameter(self, name: str, value: Any) -> None:
        """
        Set a hyperparameter value.
        
        Args:
            name: Name of the hyperparameter
            value: Value to set
        """
        self.hyperparameters[name] = value
        
        # Record in history if epoch is available
        if self.epoch_history:
            current_epoch = self.epoch_history[-1]
            self.hyperparameter_history[name].append((current_epoch, value))
            
            if self.verbose:
                self.logger.info(f"Set hyperparameter {name} = {value} at epoch {current_epoch}")
    
    def get_hyperparameter(self, name: str, default: Any = None) -> Any:
        """
        Get the current value of a hyperparameter.
        
        Args:
            name: Name of the hyperparameter
            default: Default value if hyperparameter not found
            
        Returns:
            Current value of the hyperparameter
        """
        return self.hyperparameters.get(name, default)
    
    def get_hyperparameter_history(self, name: str) -> List[tuple]:
        """
        Get history of a specific hyperparameter.
        
        Args:
            name: Name of the hyperparameter
            
        Returns:
            List of (epoch, value) tuples
        """
        if name not in self.hyperparameter_history:
            return []
        
        return list(self.hyperparameter_history[name])
    
    def record_agent_action(self, agent_name: str, action_info: Dict[str, Any]) -> None:
        """
        Record an action taken by an agent.
        
        Args:
            agent_name: Name of the agent
            action_info: Dictionary with action information
        """
        # Add timestamp and current epoch
        action_info["timestamp"] = time.time()
        if self.epoch_history:
            action_info["epoch"] = self.epoch_history[-1]
        
        self.agent_actions[agent_name].append(action_info)
        
        if self.verbose:
            action_str = ", ".join([f"{k}: {v}" for k, v in action_info.items() 
                                   if k not in ["timestamp", "state", "next_state"]])
            self.logger.info(f"Agent {agent_name} action: {action_str}")
    
    def get_agent_actions(self, agent_name: str, window: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of actions for a specific agent.
        
        Args:
            agent_name: Name of the agent
            window: Number of most recent actions to include (None for all)
            
        Returns:
            List of action information dictionaries
        """
        if agent_name not in self.agent_actions:
            return []
        
        actions = list(self.agent_actions[agent_name])
        if window is not None:
            actions = actions[-window:]
        
        return actions
    
    def save_state(self, filename: Optional[str] = None) -> None:
        """
        Save the current state to a file.
        
        Args:
            filename: Name of the file to save to (default: "shared_state.json")
        """
        if not self.log_dir:
            self.logger.warning("No log directory specified, state will not be saved")
            return
        
        filename = filename or "shared_state.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Prepare data for serialization
        state = {
            "epochs": list(self.epoch_history),
            "metrics": {k: list(v) for k, v in self.metrics_history.items()},
            "hyperparameters": self.hyperparameters,
            "hyperparameter_history": {k: list(v) for k, v in self.hyperparameter_history.items()},
            "agent_actions": {k: list(v) for k, v in self.agent_actions.items()}
        }
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        if self.verbose:
            self.logger.info(f"Saved state to {filepath}")
    
    def load_state(self, filename: Optional[str] = None) -> bool:
        """
        Load state from a file.
        
        Args:
            filename: Name of the file to load from (default: "shared_state.json")
            
        Returns:
            True if successful, False otherwise
        """
        if not self.log_dir:
            self.logger.warning("No log directory specified, state will not be loaded")
            return False
        
        filename = filename or "shared_state.json"
        filepath = os.path.join(self.log_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"State file {filepath} not found")
            return False
        
        try:
            with open(filepath, "r") as f:
                state = json.load(f)
            
            # Restore state
            self.epoch_history = deque(state["epochs"], maxlen=self.history_size)
            
            for metric_name, values in state["metrics"].items():
                self.metrics_history[metric_name] = deque(values, maxlen=self.history_size)
            
            self.hyperparameters = state["hyperparameters"]
            
            for param_name, values in state["hyperparameter_history"].items():
                self.hyperparameter_history[param_name] = deque(values, maxlen=self.history_size)
            
            for agent_name, actions in state["agent_actions"].items():
                self.agent_actions[agent_name] = deque(actions, maxlen=self.history_size)
            
            if self.verbose:
                self.logger.info(f"Loaded state from {filepath}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load state from {filepath}: {e}")
            return False
    
    def visualize_metrics(self, metrics: Optional[List[str]] = None, save: bool = True) -> None:
        """
        Visualize training metrics.
        
        Args:
            metrics: List of metrics to visualize (None for all)
            save: Whether to save the visualization to a file
        """
        if not self.epoch_history:
            self.logger.warning("No metrics to visualize")
            return
        
        if metrics is None:
            metrics = list(self.metrics_history.keys())
        
        # Create figure
        fig, ax = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            ax = [ax]
        
        epochs = list(self.epoch_history)
        
        for i, metric_name in enumerate(metrics):
            if metric_name not in self.metrics_history:
                continue
            
            values = list(self.metrics_history[metric_name])
            ax[i].plot(epochs, values, "b-", label=metric_name)
            ax[i].set_ylabel(metric_name)
            ax[i].set_title(f"{metric_name} over time")
            ax[i].grid(True)
            
            # Add hyperparameter change markers
            for param_name, history in self.hyperparameter_history.items():
                for epoch, value in history:
                    if epoch in epochs:
                        idx = epochs.index(epoch)
                        if idx < len(values):
                            ax[i].axvline(x=epoch, color="r", linestyle="--", alpha=0.3)
                            ax[i].text(epoch, values[idx], f"{param_name}={value:.4f}", 
                                     rotation=90, verticalalignment="bottom")
        
        ax[-1].set_xlabel("Epoch")
        
        # Set x-axis limits
        if self.total_epochs:
            ax[-1].set_xlim(0, self.total_epochs)
        
        plt.tight_layout()
        
        # Save figure
        if save and self.log_dir:
            filepath = os.path.join(self.log_dir, "metrics.png")
            plt.savefig(filepath)
            if self.verbose:
                self.logger.info(f"Saved metrics visualization to {filepath}")
        
        plt.close(fig)
    
    def visualize_hyperparameters(self, params: Optional[List[str]] = None, save: bool = True) -> None:
        """
        Visualize hyperparameter changes over time.
        
        Args:
            params: List of hyperparameters to visualize (None for all)
            save: Whether to save the visualization to a file
        """
        if not self.hyperparameter_history:
            self.logger.warning("No hyperparameter changes to visualize")
            return
        
        if params is None:
            params = list(self.hyperparameter_history.keys())
        
        # Create figure
        fig, ax = plt.subplots(len(params), 1, figsize=(10, 4 * len(params)), sharex=True)
        if len(params) == 1:
            ax = [ax]
        
        for i, param_name in enumerate(params):
            if param_name not in self.hyperparameter_history:
                continue
            
            history = list(self.hyperparameter_history[param_name])
            epochs = [h[0] for h in history]
            values = [h[1] for h in history]
            
            ax[i].plot(epochs, values, "g-o", label=param_name)
            ax[i].set_ylabel(param_name)
            ax[i].set_title(f"{param_name} over time")
            ax[i].grid(True)
        
        ax[-1].set_xlabel("Epoch")
        
        # Set x-axis limits
        if self.total_epochs:
            ax[-1].set_xlim(0, self.total_epochs)
        
        plt.tight_layout()
        
        # Save figure
        if save and self.log_dir:
            filepath = os.path.join(self.log_dir, "hyperparameters.png")
            plt.savefig(filepath)
            if self.verbose:
                self.logger.info(f"Saved hyperparameter visualization to {filepath}")
        
        plt.close(fig)

    def get_current_epoch(self) -> Optional[int]:
        """
        Get the current epoch.
        
        Returns:
            Current epoch or None if no epochs recorded
        """
        if not self.epoch_history:
            return None
        
        return self.epoch_history[-1]
    
    def get_enhanced_state_features(self, metric_names: List[str]) -> Dict[str, float]:
        """
        Get enhanced state features for agent state representation.
        
        Args:
            metric_names: List of metric names to include
            
        Returns:
            Dictionary of state features or empty dict if enhanced metrics disabled
        """
        if not self.enable_enhanced_metrics or not self.metric_processor:
            return {}
        
        return self.metric_processor.get_enhanced_state_features(metric_names)
    
    def get_enhanced_state_vector(self, metric_names: List[str], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get enhanced state vector for agent state representation.
        
        Args:
            metric_names: List of metric names to include
            feature_names: List of specific features to include (if None, include all)
            
        Returns:
            Numpy array of state features or empty array if enhanced metrics disabled
        """
        if not self.enable_enhanced_metrics or not self.metric_processor:
            return np.array([], dtype=np.float32)
        
        return self.metric_processor.get_enhanced_state_vector(metric_names, feature_names)
    
    def get_overfitting_signals(self) -> Dict[str, float]:
        """
        Get current overfitting signals.
        
        Returns:
            Dictionary of overfitting signals or empty dict if enhanced metrics disabled
        """
        if not self.enable_enhanced_metrics or not self.metric_processor:
            return {}
        
        return self.metric_processor.get_overfitting_signals()
