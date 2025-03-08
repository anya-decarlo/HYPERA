#!/usr/bin/env python
# Enhanced Reward System for HYPERA Agents

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import deque

class EligibilityTrace:
    """
    Implements eligibility traces for delayed credit assignment.
    
    Eligibility traces provide a way to assign credit to actions that occurred
    in the past, allowing for more effective learning of long-term dependencies.
    """
    
    def __init__(
        self,
        capacity: int = 10,
        decay_factor: float = 0.9,
        discount_factor: float = 0.99
    ):
        """
        Initialize the eligibility trace.
        
        Args:
            capacity: Maximum number of steps to track
            decay_factor: Decay factor for eligibility (lambda in TD(λ))
            discount_factor: Discount factor for future rewards (gamma)
        """
        self.capacity = capacity
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor
        self.traces = deque(maxlen=capacity)
    
    def add_trace(
        self,
        state: np.ndarray,
        action: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
        """
        Add a new state-action pair to the trace.
        
        Args:
            state: State representation
            action: Action taken
            metrics: Current metrics at this step
        """
        self.traces.append({
            'state': state.copy(),
            'action': action.copy(),
            'metrics': metrics.copy(),
            'processed': False,
            'reward': 0.0
        })
    
    def update_traces(self, reward_function: Callable) -> List[Dict[str, Any]]:
        """
        Update all traces with rewards and return processed traces.
        
        Args:
            reward_function: Function to calculate rewards between consecutive metrics
            
        Returns:
            List of processed traces with assigned rewards
        """
        if len(self.traces) < 2:
            return []
        
        processed_traces = []
        
        # Calculate immediate rewards for each transition
        for i in range(len(self.traces) - 1):
            if self.traces[i]['processed']:
                continue
                
            current_metrics = self.traces[i]['metrics']
            next_metrics = self.traces[i + 1]['metrics']
            
            # Calculate immediate reward
            immediate_reward = reward_function(current_metrics, next_metrics)
            
            # Store the reward
            self.traces[i]['reward'] = immediate_reward
            self.traces[i]['processed'] = True
            
            processed_traces.append(self.traces[i])
        
        return processed_traces
    
    def calculate_n_step_returns(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Calculate n-step returns for all processed traces.
        
        Args:
            n: Number of steps to look ahead
            
        Returns:
            List of traces with n-step returns
        """
        if len(self.traces) < 2:
            return []
        
        traces_with_returns = []
        
        for i in range(len(self.traces) - 1):
            if not self.traces[i]['processed']:
                continue
            
            # Calculate n-step return
            n_step_return = self.traces[i]['reward']
            
            # Add future rewards with discount
            for j in range(1, min(n, len(self.traces) - i - 1)):
                if i + j < len(self.traces) and self.traces[i + j]['processed']:
                    n_step_return += (self.discount_factor ** j) * self.traces[i + j]['reward']
            
            # Create a copy of the trace with the n-step return
            trace_with_return = self.traces[i].copy()
            trace_with_return['n_step_return'] = n_step_return
            
            traces_with_returns.append(trace_with_return)
        
        return traces_with_returns


class EnhancedRewardSystem:
    """
    Enhanced reward system for HYPERA agents.
    
    This system implements:
    1. Delayed rewards with eligibility traces
    2. N-step returns
    3. Reward shaping for intermediate rewards
    4. Stability, generalization, and efficiency components
    5. Adaptive reward scaling with normalization and clipping
    """
    
    def __init__(
        self,
        eligibility_trace_length: int = 10,
        n_step: int = 3,
        stability_weight: float = 0.3,
        generalization_weight: float = 0.4,
        efficiency_weight: float = 0.3,
        decay_factor: float = 0.9,
        discount_factor: float = 0.99,
        reward_scaling_window: int = 100,  # Window size for reward statistics
        reward_clip_range: Tuple[float, float] = (-10.0, 10.0),  # Clipping range
        use_adaptive_scaling: bool = True,  # Whether to use adaptive scaling
        use_phase_aware_scaling: bool = True,  # Whether to use phase-aware scaling
        auto_balance_components: bool = True  # Whether to auto-balance reward components
    ):
        """
        Initialize the enhanced reward system.
        
        Args:
            eligibility_trace_length: Length of eligibility traces
            n_step: Number of steps for n-step returns
            stability_weight: Weight for stability component
            generalization_weight: Weight for generalization component
            efficiency_weight: Weight for efficiency component
            decay_factor: Decay factor for eligibility traces
            discount_factor: Discount factor for future rewards
            reward_scaling_window: Window size for calculating reward statistics
            reward_clip_range: Range for clipping rewards (min, max)
            use_adaptive_scaling: Whether to use adaptive reward scaling
            use_phase_aware_scaling: Whether to use phase-aware scaling based on training phase
            auto_balance_components: Whether to automatically balance reward components
        """
        self.eligibility_trace = EligibilityTrace(
            capacity=eligibility_trace_length,
            decay_factor=decay_factor,
            discount_factor=discount_factor
        )
        self.n_step = n_step
        self.stability_weight = stability_weight
        self.generalization_weight = generalization_weight
        self.efficiency_weight = efficiency_weight
        
        # Metrics history for calculating stability
        self.metrics_history = {}
        self.max_history_length = 20
        
        # Adaptive reward scaling parameters
        self.reward_scaling_window = reward_scaling_window
        self.reward_clip_range = reward_clip_range
        self.use_adaptive_scaling = use_adaptive_scaling
        self.use_phase_aware_scaling = use_phase_aware_scaling
        self.auto_balance_components = auto_balance_components
        
        # Reward statistics
        self.reward_history = {
            'stability': deque(maxlen=reward_scaling_window),
            'generalization': deque(maxlen=reward_scaling_window),
            'efficiency': deque(maxlen=reward_scaling_window),
            'total': deque(maxlen=reward_scaling_window)
        }
        
        # Training phase detection
        self.training_phase = 'exploration'  # 'exploration', 'exploitation', or 'fine_tuning'
        self.phase_epoch_counter = 0
        self.phase_detection_metrics = deque(maxlen=20)  # Recent metrics for phase detection
        
        # Component scaling factors (initialized to 1.0)
        self.component_scaling_factors = {
            'stability': 1.0,
            'generalization': 1.0,
            'efficiency': 1.0
        }
    
    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
        """
        Add a new experience to the reward system.
        
        Args:
            state: Current state
            action: Action taken
            metrics: Current metrics
        """
        # Add to eligibility trace
        self.eligibility_trace.add_trace(state, action, metrics)
        
        # Update metrics history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = deque(maxlen=self.max_history_length)
            self.metrics_history[key].append(value)
        
        # Update training phase
        if self.use_phase_aware_scaling and 'dice_score' in metrics:
            self.phase_detection_metrics.append(metrics['dice_score'])
            self._update_training_phase(metrics)
    
    def calculate_stability_component(self, metrics: Dict[str, float]) -> float:
        """
        Calculate stability component of the reward.
        
        Rewards low variance in key metrics, indicating stable training.
        
        Args:
            metrics: Current metrics
            
        Returns:
            Stability component of reward
        """
        stability_reward = 0.0
        
        # Calculate variance of recent metrics
        for key in ['loss', 'val_loss']:
            if key in self.metrics_history and len(self.metrics_history[key]) > 5:
                recent_values = list(self.metrics_history[key])[-5:]
                variance = np.var(recent_values)
                
                # Lower variance is better for stability
                # Normalize and invert so that lower variance gives higher reward
                max_acceptable_variance = 0.1
                normalized_variance = min(variance / max_acceptable_variance, 1.0)
                stability_reward += (1.0 - normalized_variance)
        
        # Normalize by number of metrics considered
        if 'loss' in self.metrics_history and 'val_loss' in self.metrics_history:
            stability_reward /= 2.0
        elif 'loss' in self.metrics_history or 'val_loss' in self.metrics_history:
            stability_reward /= 1.0
        else:
            stability_reward = 0.0
        
        # Apply adaptive scaling if enabled
        if self.use_adaptive_scaling:
            stability_reward *= self.component_scaling_factors['stability']
            
            # Record raw reward for statistics
            self.reward_history['stability'].append(stability_reward)
        
        return stability_reward
    
    def calculate_generalization_component(self, metrics: Dict[str, float]) -> float:
        """
        Calculate generalization component of the reward.
        
        Rewards good performance on validation data, indicating good generalization.
        
        Args:
            metrics: Current metrics
            
        Returns:
            Generalization component of reward
        """
        generalization_reward = 0.0
        
        # Check for validation metrics
        if 'val_dice_score' in metrics:
            # Higher dice score is better
            generalization_reward += metrics['val_dice_score']
        
        if 'dice_score' in metrics and 'val_dice_score' in metrics:
            # Small gap between training and validation is good
            gap = abs(metrics['dice_score'] - metrics['val_dice_score'])
            max_acceptable_gap = 0.2
            normalized_gap = min(gap / max_acceptable_gap, 1.0)
            generalization_reward += (1.0 - normalized_gap)
            
            # Normalize
            generalization_reward /= 2.0
        
        # Apply adaptive scaling if enabled
        if self.use_adaptive_scaling:
            generalization_reward *= self.component_scaling_factors['generalization']
            
            # Record raw reward for statistics
            self.reward_history['generalization'].append(generalization_reward)
        
        return generalization_reward
    
    def calculate_efficiency_component(
        self,
        metrics: Dict[str, float],
        next_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate efficiency component of the reward.
        
        Rewards rapid improvement in key metrics, indicating efficient training.
        
        Args:
            metrics: Current metrics
            next_metrics: Next metrics
            
        Returns:
            Efficiency component of reward
        """
        efficiency_reward = 0.0
        
        # Reward improvement in dice score
        if 'dice_score' in metrics and 'dice_score' in next_metrics:
            improvement = next_metrics['dice_score'] - metrics['dice_score']
            
            # Scale improvement based on training phase if phase-aware scaling is enabled
            if self.use_phase_aware_scaling:
                if self.training_phase == 'exploration':
                    # In early training, larger improvements are expected
                    normalized_improvement = min(improvement / 0.02, 1.0)
                elif self.training_phase == 'exploitation':
                    # In mid training, moderate improvements are expected
                    normalized_improvement = min(improvement / 0.01, 1.0)
                else:  # fine_tuning
                    # In late training, smaller improvements are expected but more valuable
                    normalized_improvement = min(improvement / 0.005, 1.0) * 1.5
            else:
                # Default scaling if phase-aware scaling is disabled
                normalized_improvement = min(improvement / 0.01, 1.0)
                
            efficiency_reward += max(0.0, normalized_improvement)
        
        # Reward reduction in loss
        if 'loss' in metrics and 'loss' in next_metrics:
            reduction = metrics['loss'] - next_metrics['loss']
            
            # Scale reduction based on training phase if phase-aware scaling is enabled
            if self.use_phase_aware_scaling:
                if self.training_phase == 'exploration':
                    # In early training, larger reductions are expected
                    normalized_reduction = min(reduction / 0.1, 1.0)
                elif self.training_phase == 'exploitation':
                    # In mid training, moderate reductions are expected
                    normalized_reduction = min(reduction / 0.05, 1.0)
                else:  # fine_tuning
                    # In late training, smaller reductions are expected but more valuable
                    normalized_reduction = min(reduction / 0.02, 1.0) * 1.5
            else:
                # Default scaling if phase-aware scaling is disabled
                normalized_reduction = min(reduction / 0.05, 1.0)
                
            efficiency_reward += max(0.0, normalized_reduction)
            
            # Normalize
            efficiency_reward /= 2.0
        
        # Apply adaptive scaling if enabled
        if self.use_adaptive_scaling:
            efficiency_reward *= self.component_scaling_factors['efficiency']
            
            # Record raw reward for statistics
            self.reward_history['efficiency'].append(efficiency_reward)
        
        return efficiency_reward
    
    def calculate_reward(
        self,
        current_metrics: Dict[str, float],
        next_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate the combined reward between two consecutive steps.
        
        Args:
            current_metrics: Metrics at current step
            next_metrics: Metrics at next step
            
        Returns:
            Combined reward value
        """
        # Update component scaling factors if auto-balancing is enabled
        if self.auto_balance_components and self.use_adaptive_scaling:
            self._update_component_scaling_factors()
        
        # Calculate individual components
        stability = self.calculate_stability_component(next_metrics)
        generalization = self.calculate_generalization_component(next_metrics)
        efficiency = self.calculate_efficiency_component(current_metrics, next_metrics)
        
        # Combine components with weights
        combined_reward = (
            self.stability_weight * stability +
            self.generalization_weight * generalization +
            self.efficiency_weight * efficiency
        )
        
        # Apply adaptive scaling to total reward if enabled
        if self.use_adaptive_scaling:
            # Record raw reward for statistics
            self.reward_history['total'].append(combined_reward)
            
            # Apply z-score normalization if we have enough history
            if len(self.reward_history['total']) > 10:
                combined_reward = self._normalize_reward(combined_reward)
        
        # Apply reward clipping
        combined_reward = np.clip(
            combined_reward,
            self.reward_clip_range[0],
            self.reward_clip_range[1]
        )
        
        return combined_reward
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using z-score normalization based on recent history.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward value
        """
        history = list(self.reward_history['total'])
        
        if len(history) < 2:
            return reward
        
        mean = np.mean(history)
        std = np.std(history)
        
        # Avoid division by zero
        if std < 1e-8:
            return reward
        
        # Z-score normalization
        normalized_reward = (reward - mean) / (std + 1e-8)
        
        return normalized_reward
    
    def _update_component_scaling_factors(self) -> None:
        """
        Update scaling factors for reward components to balance their contributions.
        """
        # Only update if we have enough history
        min_history = 10
        if (len(self.reward_history['stability']) < min_history or
            len(self.reward_history['generalization']) < min_history or
            len(self.reward_history['efficiency']) < min_history):
            return
        
        # Calculate statistics for each component
        stats = {}
        for component in ['stability', 'generalization', 'efficiency']:
            values = list(self.reward_history[component])
            stats[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
        
        # Calculate target mean (average of all component means)
        target_mean = np.mean([stats[c]['mean'] for c in stats])
        
        # Update scaling factors to balance means
        for component in stats:
            if stats[component]['mean'] > 1e-8:  # Avoid division by zero
                # Gradually adjust scaling factor towards target
                target_factor = target_mean / stats[component]['mean']
                # Use exponential moving average for smooth transitions
                self.component_scaling_factors[component] = (
                    0.9 * self.component_scaling_factors[component] +
                    0.1 * target_factor
                )
    
    def _update_training_phase(self, metrics: Dict[str, float]) -> None:
        """
        Update the detected training phase based on recent metrics.
        
        Args:
            metrics: Current metrics
        """
        self.phase_epoch_counter += 1
        
        # Need enough history to detect phase
        if len(self.phase_detection_metrics) < 10:
            self.training_phase = 'exploration'
            return
        
        # Calculate improvement rate over recent epochs
        recent_metrics = list(self.phase_detection_metrics)
        if len(recent_metrics) >= 10:
            early_avg = np.mean(recent_metrics[:5])
            late_avg = np.mean(recent_metrics[-5:])
            improvement_rate = (late_avg - early_avg) / max(early_avg, 1e-8)
            
            # Determine phase based on improvement rate and absolute performance
            current_performance = recent_metrics[-1]
            
            if current_performance < 0.5:
                # Early training with low performance
                self.training_phase = 'exploration'
            elif improvement_rate > 0.05:
                # Significant improvement rate
                self.training_phase = 'exploitation'
            elif current_performance > 0.8:
                # High performance with slower improvement
                self.training_phase = 'fine_tuning'
            else:
                # Default to exploitation for mid-range performance
                self.training_phase = 'exploitation'
    
    def get_processed_experiences(self) -> List[Dict[str, Any]]:
        """
        Process experiences and return them with calculated rewards.
        
        Returns:
            List of processed experiences with n-step returns
        """
        # Update traces with rewards
        self.eligibility_trace.update_traces(self.calculate_reward)
        
        # Calculate n-step returns
        return self.eligibility_trace.calculate_n_step_returns(self.n_step)
    
    def get_latest_reward_components(self) -> Dict[str, float]:
        """
        Get the latest calculated reward components.
        
        Returns:
            Dictionary of reward components
        """
        if len(self.eligibility_trace.traces) < 2:
            return {
                'stability': 0.0,
                'generalization': 0.0,
                'efficiency': 0.0,
                'total': 0.0,
                'training_phase': self.training_phase
            }
        
        current_metrics = self.eligibility_trace.traces[-2]['metrics']
        next_metrics = self.eligibility_trace.traces[-1]['metrics']
        
        stability = self.calculate_stability_component(next_metrics)
        generalization = self.calculate_generalization_component(next_metrics)
        efficiency = self.calculate_efficiency_component(current_metrics, next_metrics)
        
        total = (
            self.stability_weight * stability +
            self.generalization_weight * generalization +
            self.efficiency_weight * efficiency
        )
        
        # Apply adaptive scaling to total reward if enabled
        if self.use_adaptive_scaling and len(self.reward_history['total']) > 10:
            normalized_total = self._normalize_reward(total)
            clipped_total = np.clip(
                normalized_total,
                self.reward_clip_range[0],
                self.reward_clip_range[1]
            )
        else:
            clipped_total = np.clip(
                total,
                self.reward_clip_range[0],
                self.reward_clip_range[1]
            )
        
        return {
            'stability': stability,
            'generalization': generalization,
            'efficiency': efficiency,
            'total': total,
            'normalized_total': clipped_total if self.use_adaptive_scaling else total,
            'training_phase': self.training_phase
        }
