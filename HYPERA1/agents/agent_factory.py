#!/usr/bin/env python
# Agent Factory for Multi-Agent Hyperparameter Optimization

from typing import Dict, List, Any, Optional, Tuple
import logging
import torch

from .shared_state import SharedStateManager
from .base_agent import BaseHyperparameterAgent
from .learning_rate_agent import LearningRateAgent
from .weight_decay_agent import WeightDecayAgent
from .class_weights_agent import ClassWeightsAgent
from .normalization_agent import NormalizationAgent
from .loss_function_agent import LossFunctionAgent

class AgentFactory:
    """Factory class for creating hyperparameter optimization agents."""
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        log_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        """
        Initialize the agent factory.
        
        Args:
            shared_state_manager: Manager for shared state between agents
            log_dir: Directory for logging
            device: Device to use for training
            verbose: Whether to print verbose output
        """
        self.shared_state_manager = shared_state_manager
        self.log_dir = log_dir
        self.device = device
        self.verbose = verbose
        
    def create_learning_rate_agent(
        self,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        update_frequency: int = 1,
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
        name: str = "learning_rate_agent",
        priority: int = 0
    ) -> LearningRateAgent:
        """
        Create a learning rate optimization agent.
        
        Args:
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            update_frequency: How often to update the learning rate
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
            name: Name of agent
            priority: Priority of the agent (higher means more important)
            
        Returns:
            Learning rate optimization agent
        """
        return LearningRateAgent(
            shared_state_manager=self.shared_state_manager,
            min_lr=min_lr,
            max_lr=max_lr,
            update_frequency=update_frequency,
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
            log_dir=self.log_dir,
            device=self.device,
            verbose=self.verbose,
            name=name,
            priority=priority
        )
        
    def create_weight_decay_agent(
        self,
        min_wd: float = 1e-8,
        max_wd: float = 1e-3,
        update_frequency: int = 5,
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
        name: str = "weight_decay_agent",
        priority: int = 0
    ) -> WeightDecayAgent:
        """
        Create a weight decay optimization agent.
        
        Args:
            min_wd: Minimum weight decay
            max_wd: Maximum weight decay
            update_frequency: How often to update the weight decay
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
            name: Name of agent
            priority: Priority of the agent (higher means more important)
            
        Returns:
            Weight decay optimization agent
        """
        return WeightDecayAgent(
            shared_state_manager=self.shared_state_manager,
            min_wd=min_wd,
            max_wd=max_wd,
            update_frequency=update_frequency,
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
            log_dir=self.log_dir,
            device=self.device,
            verbose=self.verbose,
            name=name,
            priority=priority
        )
        
    def create_class_weights_agent(
        self,
        num_classes: int,
        min_weight: float = 0.5,
        max_weight: float = 5.0,
        update_frequency: int = 10,
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
        name: str = "class_weights_agent",
        priority: int = 0
    ) -> ClassWeightsAgent:
        """
        Create a class weights optimization agent.
        
        Args:
            num_classes: Number of classes
            min_weight: Minimum class weight
            max_weight: Maximum class weight
            update_frequency: How often to update class weights
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
            name: Name of agent
            priority: Priority of the agent (higher means more important)
            
        Returns:
            Class weights optimization agent
        """
        return ClassWeightsAgent(
            shared_state_manager=self.shared_state_manager,
            num_classes=num_classes,
            min_weight=min_weight,
            max_weight=max_weight,
            update_frequency=update_frequency,
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
            log_dir=self.log_dir,
            device=self.device,
            verbose=self.verbose,
            name=name,
            priority=priority
        )
        
    def create_loss_function_agent(
        self,
        min_lambda_ce: float = 0.1,
        max_lambda_ce: float = 2.0,
        min_lambda_dice: float = 0.1,
        max_lambda_dice: float = 2.0,
        min_focal_gamma: float = 0.5,
        max_focal_gamma: float = 5.0,
        update_frequency: int = 15,
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
        name: str = "loss_function_agent",
        priority: int = 0
    ) -> LossFunctionAgent:
        """
        Create a loss function optimization agent.
        
        Args:
            min_lambda_ce: Minimum weight for cross-entropy loss
            max_lambda_ce: Maximum weight for cross-entropy loss
            min_lambda_dice: Minimum weight for Dice loss
            max_lambda_dice: Maximum weight for Dice loss
            min_focal_gamma: Minimum focal loss gamma parameter
            max_focal_gamma: Maximum focal loss gamma parameter
            update_frequency: How often to update loss function parameters
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
            name: Name of agent
            priority: Priority of the agent (higher means more important)
            
        Returns:
            Loss function optimization agent
        """
        return LossFunctionAgent(
            shared_state_manager=self.shared_state_manager,
            min_lambda_ce=min_lambda_ce,
            max_lambda_ce=max_lambda_ce,
            min_lambda_dice=min_lambda_dice,
            max_lambda_dice=max_lambda_dice,
            min_focal_gamma=min_focal_gamma,
            max_focal_gamma=max_focal_gamma,
            update_frequency=update_frequency,
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
            log_dir=self.log_dir,
            device=self.device,
            verbose=self.verbose,
            name=name,
            priority=priority
        )
        
    def create_normalization_agent(
        self,
        update_frequency: int = 20,
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
        name: str = "normalization_agent",
        priority: int = 0
    ) -> NormalizationAgent:
        """
        Create a normalization optimization agent.
        
        Args:
            update_frequency: How often to update normalization type
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
            name: Name of agent
            priority: Priority of the agent (higher means more important)
            
        Returns:
            Normalization optimization agent
        """
        return NormalizationAgent(
            shared_state_manager=self.shared_state_manager,
            update_frequency=update_frequency,
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
            log_dir=self.log_dir,
            device=self.device,
            verbose=self.verbose,
            name=name,
            priority=priority
        )
        
    def create_all_agents(
        self,
        num_classes: int,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        min_wd: float = 1e-8,
        max_wd: float = 1e-2,
        min_class_weight: float = 0.5,
        max_class_weight: float = 5.0,
        min_lambda_ce: float = 0.0,
        max_lambda_ce: float = 1.0,
        min_lambda_dice: float = 0.0,
        max_lambda_dice: float = 1.0,
        min_focal_gamma: float = 0.0,
        max_focal_gamma: float = 5.0,
        use_adaptive_scaling: bool = True,
        use_phase_aware_scaling: bool = True,
        auto_balance_components: bool = True,
        reward_clip_range: Tuple[float, float] = (-10.0, 10.0),
        reward_scaling_window: int = 100
    ) -> Dict[str, BaseHyperparameterAgent]:
        """
        Create all hyperparameter optimization agents.
        
        Args:
            num_classes: Number of classes
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            min_wd: Minimum weight decay
            max_wd: Maximum weight decay
            min_class_weight: Minimum class weight
            max_class_weight: Maximum class weight
            min_lambda_ce: Minimum weight for cross-entropy loss
            max_lambda_ce: Maximum weight for cross-entropy loss
            min_lambda_dice: Minimum weight for dice loss
            max_lambda_dice: Maximum weight for dice loss
            min_focal_gamma: Minimum gamma for focal loss
            max_focal_gamma: Maximum gamma for focal loss
            use_adaptive_scaling: Whether to use adaptive reward scaling
            use_phase_aware_scaling: Whether to use phase-aware scaling
            auto_balance_components: Whether to auto-balance reward components
            reward_clip_range: Range for clipping rewards
            reward_scaling_window: Window size for reward statistics
            
        Returns:
            Dictionary of all hyperparameter optimization agents
        """
        return {
            "learning_rate": self.create_learning_rate_agent(
                min_lr=min_lr,
                max_lr=max_lr,
                update_frequency=1,
                eligibility_trace_length=10,
                n_step=3,
                stability_weight=0.3,
                generalization_weight=0.4,
                efficiency_weight=0.3,
                use_adaptive_scaling=use_adaptive_scaling,
                use_phase_aware_scaling=use_phase_aware_scaling,
                auto_balance_components=auto_balance_components,
                reward_clip_range=reward_clip_range,
                reward_scaling_window=reward_scaling_window,
                priority=100  # Highest priority
            ),
            "weight_decay": self.create_weight_decay_agent(
                min_wd=min_wd,
                max_wd=max_wd,
                update_frequency=5,
                eligibility_trace_length=12,
                n_step=3,
                stability_weight=0.3,
                generalization_weight=0.4,
                efficiency_weight=0.3,
                use_adaptive_scaling=use_adaptive_scaling,
                use_phase_aware_scaling=use_phase_aware_scaling,
                auto_balance_components=auto_balance_components,
                reward_clip_range=reward_clip_range,
                reward_scaling_window=reward_scaling_window,
                priority=80  # High priority
            ),
            "class_weights": self.create_class_weights_agent(
                num_classes=num_classes,
                min_weight=min_class_weight,
                max_weight=max_class_weight,
                update_frequency=10,
                eligibility_trace_length=15,
                n_step=4,
                stability_weight=0.3,
                generalization_weight=0.4,
                efficiency_weight=0.3,
                use_adaptive_scaling=use_adaptive_scaling,
                use_phase_aware_scaling=use_phase_aware_scaling,
                auto_balance_components=auto_balance_components,
                reward_clip_range=reward_clip_range,
                reward_scaling_window=reward_scaling_window,
                priority=60  # Medium priority
            ),
            "normalization": self.create_normalization_agent(
                update_frequency=20,
                eligibility_trace_length=20,
                n_step=5,
                stability_weight=0.4,
                generalization_weight=0.4,
                efficiency_weight=0.2,
                use_adaptive_scaling=use_adaptive_scaling,
                use_phase_aware_scaling=use_phase_aware_scaling,
                auto_balance_components=auto_balance_components,
                reward_clip_range=reward_clip_range,
                reward_scaling_window=reward_scaling_window,
                priority=40  # Lower priority
            ),
            "loss_function": self.create_loss_function_agent(
                min_lambda_ce=min_lambda_ce,
                max_lambda_ce=max_lambda_ce,
                min_lambda_dice=min_lambda_dice,
                max_lambda_dice=max_lambda_dice,
                min_focal_gamma=min_focal_gamma,
                max_focal_gamma=max_focal_gamma,
                update_frequency=15,
                eligibility_trace_length=15,
                n_step=4,
                stability_weight=0.3,
                generalization_weight=0.4,
                efficiency_weight=0.3,
                use_adaptive_scaling=use_adaptive_scaling,
                use_phase_aware_scaling=use_phase_aware_scaling,
                auto_balance_components=auto_balance_components,
                reward_clip_range=reward_clip_range,
                reward_scaling_window=reward_scaling_window,
                priority=50  # Medium-low priority
            )
        }
