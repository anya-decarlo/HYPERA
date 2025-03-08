#!/usr/bin/env python
# Reward Statistics Tracker - Monitors and analyzes reward components over time

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from collections import deque

class RewardStatisticsTracker:
    """
    Tracks statistics for reward components over time.
    
    This class maintains a history of reward components and provides methods for
    analyzing trends, calculating statistics, and normalizing rewards.
    
    Attributes:
        window_size (int): Size of the sliding window for statistics
        reward_history (Dict[str, deque]): History of rewards for each component
        normalized_history (Dict[str, deque]): History of normalized rewards
        component_means (Dict[str, float]): Running means for each component
        component_stds (Dict[str, float]): Running standard deviations
        reward_clip_range (Tuple[float, float]): Range for clipping rewards
        use_z_score_normalization (bool): Whether to use z-score normalization
        logger (logging.Logger): Logger for the statistics tracker
    """
    
    def __init__(
        self,
        window_size: int = 100,
        reward_clip_range: Tuple[float, float] = (-10.0, 10.0),
        use_z_score_normalization: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the reward statistics tracker.
        
        Args:
            window_size: Size of the sliding window for statistics
            reward_clip_range: Range for clipping rewards
            use_z_score_normalization: Whether to use z-score normalization
            verbose: Whether to print verbose output
        """
        self.window_size = window_size
        self.reward_clip_range = reward_clip_range
        self.use_z_score_normalization = use_z_score_normalization
        
        # Initialize reward history
        self.reward_history = {
            "dice": deque(maxlen=window_size),
            "boundary": deque(maxlen=window_size),
            "obj_f1": deque(maxlen=window_size),
            "shape": deque(maxlen=window_size),
            "fg_bg_balance": deque(maxlen=window_size),
            "total": deque(maxlen=window_size)
        }
        
        # Initialize normalized history
        self.normalized_history = {
            "dice": deque(maxlen=window_size),
            "boundary": deque(maxlen=window_size),
            "obj_f1": deque(maxlen=window_size),
            "shape": deque(maxlen=window_size),
            "fg_bg_balance": deque(maxlen=window_size),
            "total": deque(maxlen=window_size)
        }
        
        # Initialize component statistics
        self.component_means = {
            "dice": 0.0,
            "boundary": 0.0,
            "obj_f1": 0.0,
            "shape": 0.0,
            "fg_bg_balance": 0.0,
            "total": 0.0
        }
        
        self.component_stds = {
            "dice": 1.0,
            "boundary": 1.0,
            "obj_f1": 1.0,
            "shape": 1.0,
            "fg_bg_balance": 1.0,
            "total": 1.0
        }
        
        # Set up logging
        self.logger = logging.getLogger("RewardStatisticsTracker")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initialized RewardStatisticsTracker with window size {window_size}")
        self.logger.info(f"Reward clip range: {reward_clip_range}")
        self.logger.info(f"Using z-score normalization: {use_z_score_normalization}")
    
    def update(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Update statistics with new reward values.
        
        Args:
            rewards: Dictionary of reward components
            
        Returns:
            Dictionary of normalized rewards
        """
        # Update reward history
        for component, value in rewards.items():
            if component in self.reward_history:
                self.reward_history[component].append(value)
        
        # Update total reward history if not provided
        if "total" not in rewards and len(rewards) > 0:
            total_reward = sum(rewards.values())
            self.reward_history["total"].append(total_reward)
        
        # Update component statistics
        self._update_statistics()
        
        # Normalize rewards
        normalized_rewards = self._normalize_rewards(rewards)
        
        # Update normalized history
        for component, value in normalized_rewards.items():
            if component in self.normalized_history:
                self.normalized_history[component].append(value)
        
        return normalized_rewards
    
    def _update_statistics(self) -> None:
        """
        Update component statistics based on reward history.
        """
        for component, history in self.reward_history.items():
            if len(history) > 0:
                # Update mean
                self.component_means[component] = np.mean(history)
                
                # Update standard deviation (with minimum value to avoid division by zero)
                self.component_stds[component] = max(np.std(history), 1e-6)
    
    def _normalize_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize rewards using z-score normalization and clipping.
        
        Args:
            rewards: Dictionary of reward components
            
        Returns:
            Dictionary of normalized rewards
        """
        normalized_rewards = {}
        
        for component, value in rewards.items():
            if component in self.component_means and component in self.component_stds:
                # Apply z-score normalization if enabled
                if self.use_z_score_normalization:
                    normalized_value = (value - self.component_means[component]) / self.component_stds[component]
                else:
                    normalized_value = value
                
                # Clip normalized value
                normalized_value = np.clip(
                    normalized_value,
                    self.reward_clip_range[0],
                    self.reward_clip_range[1]
                )
                
                normalized_rewards[component] = normalized_value
            else:
                # If component is not tracked, pass through unchanged
                normalized_rewards[component] = value
        
        return normalized_rewards
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each reward component.
        
        Returns:
            Dictionary of component statistics
        """
        statistics = {}
        
        for component in self.reward_history.keys():
            history = list(self.reward_history[component])
            
            if len(history) > 0:
                statistics[component] = {
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "median": np.median(history),
                    "count": len(history)
                }
            else:
                statistics[component] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "count": 0
                }
        
        return statistics
    
    def get_reward_trends(self) -> Dict[str, Dict[str, float]]:
        """
        Get trend information for each reward component.
        
        Returns:
            Dictionary of component trends
        """
        trends = {}
        
        for component in self.reward_history.keys():
            history = list(self.reward_history[component])
            
            if len(history) > 10:
                # Calculate trend over last 10 values
                recent_history = history[-10:]
                
                # Calculate slope using linear regression
                x = np.arange(len(recent_history))
                slope, _ = np.polyfit(x, recent_history, 1)
                
                # Calculate improvement percentage
                start_value = recent_history[0]
                end_value = recent_history[-1]
                
                if abs(start_value) > 1e-6:
                    improvement_pct = (end_value - start_value) / abs(start_value) * 100.0
                else:
                    improvement_pct = 0.0
                
                trends[component] = {
                    "slope": slope,
                    "improvement_pct": improvement_pct,
                    "is_improving": slope > 0
                }
            else:
                trends[component] = {
                    "slope": 0.0,
                    "improvement_pct": 0.0,
                    "is_improving": False
                }
        
        return trends
    
    def get_component_correlations(self) -> Dict[Tuple[str, str], float]:
        """
        Get correlations between reward components.
        
        Returns:
            Dictionary of component correlations
        """
        correlations = {}
        
        components = list(self.reward_history.keys())
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                history1 = list(self.reward_history[comp1])
                history2 = list(self.reward_history[comp2])
                
                if len(history1) > 5 and len(history2) > 5:
                    # Ensure same length
                    min_len = min(len(history1), len(history2))
                    history1 = history1[-min_len:]
                    history2 = history2[-min_len:]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(history1, history2)[0, 1]
                    
                    correlations[(comp1, comp2)] = correlation
                else:
                    correlations[(comp1, comp2)] = 0.0
        
        return correlations
    
    def set_window_size(self, window_size: int) -> None:
        """
        Set the window size for statistics.
        
        Args:
            window_size: New window size
        """
        self.window_size = window_size
        
        # Create new deques with new maxlen
        for component in self.reward_history.keys():
            old_history = list(self.reward_history[component])
            self.reward_history[component] = deque(old_history[-window_size:], maxlen=window_size)
            
            old_normalized = list(self.normalized_history[component])
            self.normalized_history[component] = deque(old_normalized[-window_size:], maxlen=window_size)
        
        self.logger.info(f"Updated window size to {window_size}")
    
    def set_reward_clip_range(self, clip_range: Tuple[float, float]) -> None:
        """
        Set the range for clipping rewards.
        
        Args:
            clip_range: New clip range
        """
        self.reward_clip_range = clip_range
        self.logger.info(f"Updated reward clip range to {clip_range}")
    
    def enable_z_score_normalization(self) -> None:
        """
        Enable z-score normalization.
        """
        self.use_z_score_normalization = True
        self.logger.info("Enabled z-score normalization")
    
    def disable_z_score_normalization(self) -> None:
        """
        Disable z-score normalization.
        """
        self.use_z_score_normalization = False
        self.logger.info("Disabled z-score normalization")
    
    def reset(self) -> None:
        """
        Reset all statistics.
        """
        # Clear reward history
        for component in self.reward_history.keys():
            self.reward_history[component].clear()
            self.normalized_history[component].clear()
        
        # Reset component statistics
        for component in self.component_means.keys():
            self.component_means[component] = 0.0
            self.component_stds[component] = 1.0
        
        self.logger.info("Reset all statistics")
