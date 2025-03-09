#!/usr/bin/env python
# Foreground-Background Balance Agent - Specialized agent for optimizing class balance

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

from ..base_segmentation_agent import BaseSegmentationAgent
from ..segmentation_state_manager import SegmentationStateManager

class FGBalanceAgent(BaseSegmentationAgent):
    """
    Specialized agent for optimizing foreground-background balance in segmentation.
    
    This agent focuses on improving the foreground-background balance component of the reward function,
    which prevents over-segmentation (predicting too much foreground) or under-segmentation
    (predicting too little foreground) by encouraging a balance similar to the ground truth.
    
    Attributes:
        name (str): Name of the agent
        state_manager (SegmentationStateManager): Manager for shared state
        device (torch.device): Device to use for computation
        feature_extractor (nn.Module): Neural network for extracting features
        policy_network (nn.Module): Neural network for decision making
        optimizer (torch.optim.Optimizer): Optimizer for policy network
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor for future rewards
        update_frequency (int): Frequency of agent updates
        last_update_step (int): Last step when agent was updated
        action_history (List): History of actions taken by agent
        reward_history (List): History of rewards received by agent
        observation_history (List): History of observations
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        state_manager: SegmentationStateManager,
        device: torch.device = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 5,
        verbose: bool = False
    ):
        """
        Initialize the foreground-background balance agent.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            verbose: Whether to print verbose output
        """
        super().__init__(
            name="FGBalanceAgent",
            state_manager=state_manager,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=verbose
        )
        
        # Initialize balance-specific components
        self._init_balance_components()
        
        if self.verbose:
            self.logger.info("Initialized FGBalanceAgent")
    
    def _init_balance_components(self):
        """
        Initialize balance-specific components.
        """
        # Feature extractor for balance features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        ).to(self.device)
        
        # Policy network for balance-specific actions
        self.policy_network = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 2, 256),  # +2 for current fg ratio and target fg ratio
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 actions: increase, decrease, or maintain threshold
        ).to(self.device)
        
        # Optimizer for policy network
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate
        )
        
        # Balance-specific parameters
        self.segmentation_threshold = 0.5  # Initial segmentation threshold
        self.min_threshold = 0.1
        self.max_threshold = 0.9
        self.threshold_step = 0.05
        
        # Target foreground ratio (will be updated based on ground truth)
        self.target_fg_ratio = 0.5
        
        # Moving average of foreground ratio
        self.avg_fg_ratio = 0.5
        self.avg_decay = 0.9  # Decay factor for moving average
    
    def observe(self) -> Dict[str, Any]:
        """
        Observe the current state.
        
        Returns:
            Dictionary of observations
        """
        # Get current state from state manager
        current_image = self.state_manager.get_current_image()
        current_mask = self.state_manager.get_current_mask()
        current_prediction = self.state_manager.get_current_prediction()
        
        if current_image is None or current_mask is None or current_prediction is None:
            # If any required state is missing, return empty observation
            return {}
        
        # Ensure tensors are on the correct device
        current_image = current_image.to(self.device)
        current_mask = current_mask.to(self.device)
        current_prediction = current_prediction.to(self.device)
        
        # Calculate foreground ratios
        gt_fg_ratio = torch.mean((current_mask > 0.5).float()).item()
        pred_fg_ratio = torch.mean((current_prediction > self.segmentation_threshold).float()).item()
        
        # Update target foreground ratio based on ground truth
        self.target_fg_ratio = gt_fg_ratio
        
        # Update moving average of foreground ratio
        self.avg_fg_ratio = self.avg_decay * self.avg_fg_ratio + (1 - self.avg_decay) * pred_fg_ratio
        
        # Calculate foreground-background balance metrics
        fg_balance_metrics = self._calculate_fg_balance_metrics(current_prediction, current_mask)
        
        # Get recent metrics from state manager
        recent_metrics = self.state_manager.get_recent_metrics()
        
        # Create observation dictionary
        observation = {
            "current_image": current_image,
            "current_mask": current_mask,
            "current_prediction": current_prediction,
            "gt_fg_ratio": gt_fg_ratio,
            "pred_fg_ratio": pred_fg_ratio,
            "avg_fg_ratio": self.avg_fg_ratio,
            "target_fg_ratio": self.target_fg_ratio,
            "segmentation_threshold": self.segmentation_threshold,
            "fg_balance_metrics": fg_balance_metrics,
            "recent_metrics": recent_metrics
        }
        
        # Store observation in history
        self.observation_history.append(observation)
        
        return observation
    
    def _calculate_fg_balance_metrics(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        Calculate foreground-background balance metrics.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dictionary of foreground-background balance metrics
        """
        # Ensure tensors are on the correct device
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Calculate foreground ratios
        pred_fg_ratio = torch.mean((prediction > self.segmentation_threshold).float()).item()
        gt_fg_ratio = torch.mean((ground_truth > 0.5).float()).item()
        
        # Calculate absolute difference in foreground ratios
        fg_ratio_diff = abs(pred_fg_ratio - gt_fg_ratio)
        
        # Calculate class-wise metrics
        pred_binary = (prediction > self.segmentation_threshold).float()
        gt_binary = (ground_truth > 0.5).float()
        
        # True positives, false positives, false negatives
        tp = torch.sum(pred_binary * gt_binary).item()
        fp = torch.sum(pred_binary * (1 - gt_binary)).item()
        fn = torch.sum((1 - pred_binary) * gt_binary).item()
        tn = torch.sum((1 - pred_binary) * (1 - gt_binary)).item()
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate class imbalance metrics
        fg_pixels = torch.sum(gt_binary).item()
        bg_pixels = torch.sum(1 - gt_binary).item()
        total_pixels = fg_pixels + bg_pixels
        
        fg_weight = bg_pixels / total_pixels if total_pixels > 0 else 0.5
        bg_weight = fg_pixels / total_pixels if total_pixels > 0 else 0.5
        
        # Create metrics dictionary
        fg_balance_metrics = {
            "pred_fg_ratio": pred_fg_ratio,
            "gt_fg_ratio": gt_fg_ratio,
            "fg_ratio_diff": fg_ratio_diff,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "fg_weight": fg_weight,
            "bg_weight": bg_weight
        }
        
        return fg_balance_metrics
    
    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on an action based on the observation.
        
        Args:
            observation: Dictionary of observations
            
        Returns:
            Dictionary of actions
        """
        if not observation:
            # If observation is empty, return no action
            return {}
        
        # Extract features from current image
        current_image = observation["current_image"]
        
        # Ensure image has channel dimension
        if len(current_image.shape) == 3:
            current_image = current_image.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(current_image)
        features = features.view(features.size(0), -1)
        
        # Add current foreground ratio and target foreground ratio to features
        pred_fg_ratio = observation["pred_fg_ratio"]
        target_fg_ratio = observation["target_fg_ratio"]
        
        # Concatenate features with foreground ratios
        extended_features = torch.cat([
            features,
            torch.tensor([[pred_fg_ratio, target_fg_ratio]], device=self.device)
        ], dim=1)
        
        # Get action logits from policy network
        action_logits = self.policy_network(extended_features)
        
        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Sample action from probabilities
        if self.training:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = torch.argmax(action_probs, dim=1).item()
        
        # Map action to segmentation threshold adjustment
        if action == 0:
            # Increase threshold (decreases foreground)
            new_threshold = min(self.segmentation_threshold + self.threshold_step, self.max_threshold)
        elif action == 1:
            # Decrease threshold (increases foreground)
            new_threshold = max(self.segmentation_threshold - self.threshold_step, self.min_threshold)
        else:
            # Maintain current threshold
            new_threshold = self.segmentation_threshold
        
        # Create action dictionary
        action_dict = {
            "segmentation_threshold": new_threshold,
            "action_type": action,
            "action_probs": action_probs.detach().cpu().numpy(),
            "pred_fg_ratio": pred_fg_ratio,
            "target_fg_ratio": target_fg_ratio
        }
        
        # Store action in history
        self.action_history.append(action_dict)
        
        # Update segmentation threshold
        self.segmentation_threshold = new_threshold
        
        return action_dict
    
    def learn(self, reward: float) -> Dict[str, float]:
        """
        Learn from the reward.
        
        Args:
            reward: Reward value
            
        Returns:
            Dictionary of learning metrics
        """
        # Store reward in history
        self.reward_history.append(reward)
        
        # Check if it's time to update
        current_step = self.state_manager.get_current_step()
        if current_step - self.last_update_step < self.update_frequency:
            return {}
        
        # Update last update step
        self.last_update_step = current_step
        
        # Check if we have enough history for learning
        if len(self.action_history) < 2 or len(self.reward_history) < 2:
            return {}
        
        # Get the most recent observation, action, and reward
        observation = self.observation_history[-1]
        action = self.action_history[-1]
        
        # Extract features from current image
        current_image = observation["current_image"]
        
        # Ensure image has channel dimension
        if len(current_image.shape) == 3:
            current_image = current_image.unsqueeze(1)
        
        # Extract features
        features = self.feature_extractor(current_image)
        features = features.view(features.size(0), -1)
        
        # Add current foreground ratio and target foreground ratio to features
        pred_fg_ratio = observation["pred_fg_ratio"]
        target_fg_ratio = observation["target_fg_ratio"]
        
        # Concatenate features with foreground ratios
        extended_features = torch.cat([
            features,
            torch.tensor([[pred_fg_ratio, target_fg_ratio]], device=self.device)
        ], dim=1)
        
        # Get action logits from policy network
        action_logits = self.policy_network(extended_features)
        
        # Calculate policy loss using REINFORCE algorithm
        action_type = action["action_type"]
        policy_loss = -action_logits[0, action_type] * reward
        
        # Backpropagate and optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Create learning metrics dictionary
        metrics = {
            "policy_loss": policy_loss.item(),
            "reward": reward,
            "segmentation_threshold": self.segmentation_threshold,
            "pred_fg_ratio": pred_fg_ratio,
            "target_fg_ratio": target_fg_ratio,
            "fg_ratio_diff": abs(pred_fg_ratio - target_fg_ratio)
        }
        
        return metrics
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary of agent state
        """
        return {
            "name": self.name,
            "segmentation_threshold": self.segmentation_threshold,
            "target_fg_ratio": self.target_fg_ratio,
            "avg_fg_ratio": self.avg_fg_ratio,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "update_frequency": self.update_frequency,
            "last_update_step": self.last_update_step,
            "action_history_length": len(self.action_history),
            "reward_history_length": len(self.reward_history),
            "observation_history_length": len(self.observation_history),
            "training": self.training
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the agent.
        
        Args:
            state: Dictionary of agent state
        """
        if "segmentation_threshold" in state:
            self.segmentation_threshold = state["segmentation_threshold"]
        
        if "target_fg_ratio" in state:
            self.target_fg_ratio = state["target_fg_ratio"]
        
        if "avg_fg_ratio" in state:
            self.avg_fg_ratio = state["avg_fg_ratio"]
        
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
            # Update optimizer with new learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        
        if "gamma" in state:
            self.gamma = state["gamma"]
        
        if "update_frequency" in state:
            self.update_frequency = state["update_frequency"]
        
        if "training" in state:
            self.training = state["training"]
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        # Create state dictionary
        state_dict = {
            "feature_extractor": self.feature_extractor.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "segmentation_threshold": self.segmentation_threshold,
            "target_fg_ratio": self.target_fg_ratio,
            "avg_fg_ratio": self.avg_fg_ratio,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "update_frequency": self.update_frequency,
            "last_update_step": self.last_update_step,
            "action_history": self.action_history,
            "reward_history": self.reward_history,
            "training": self.training
        }
        
        # Save state dictionary
        torch.save(state_dict, path)
        
        if self.verbose:
            self.logger.info(f"Saved FGBalanceAgent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        # Load state dictionary
        state_dict = torch.load(path, map_location=self.device)
        
        # Load model parameters
        self.feature_extractor.load_state_dict(state_dict["feature_extractor"])
        self.policy_network.load_state_dict(state_dict["policy_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        
        # Load agent parameters
        self.segmentation_threshold = state_dict["segmentation_threshold"]
        self.target_fg_ratio = state_dict["target_fg_ratio"]
        self.avg_fg_ratio = state_dict["avg_fg_ratio"]
        self.learning_rate = state_dict["learning_rate"]
        self.gamma = state_dict["gamma"]
        self.update_frequency = state_dict["update_frequency"]
        self.last_update_step = state_dict["last_update_step"]
        self.action_history = state_dict["action_history"]
        self.reward_history = state_dict["reward_history"]
        self.training = state_dict["training"]
        
        if self.verbose:
            self.logger.info(f"Loaded FGBalanceAgent from {path}")
