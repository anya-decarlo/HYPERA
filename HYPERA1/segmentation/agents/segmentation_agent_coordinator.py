#!/usr/bin/env python
# Segmentation Agent Coordinator - Manages and coordinates specialized segmentation agents

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time

from .segmentation_agent_factory import SegmentationAgentFactory
from ..segmentation_state_manager import SegmentationStateManager
from ..rewards.multi_objective_reward import MultiObjectiveRewardCalculator

class SegmentationAgentCoordinator:
    """
    Coordinates multiple specialized segmentation agents.
    
    This class manages the interactions between specialized segmentation agents,
    resolves conflicts, and ensures that agents work together effectively to
    optimize segmentation performance.
    
    Attributes:
        state_manager (SegmentationStateManager): Manager for shared state
        agents (Dict[str, Any]): Dictionary of specialized agents
        reward_calculator (MultiObjectiveRewardCalculator): Calculator for rewards
        device (torch.device): Device to use for computation
        update_priorities (Dict[str, int]): Priorities for agent updates
        conflict_resolution_strategy (str): Strategy for resolving conflicts
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        state_manager: SegmentationStateManager,
        reward_calculator: MultiObjectiveRewardCalculator,
        device: torch.device = None,
        conflict_resolution_strategy: str = "weighted_average",
        verbose: bool = False
    ):
        """
        Initialize the segmentation agent coordinator.
        
        Args:
            state_manager: Manager for shared state
            reward_calculator: Calculator for rewards
            device: Device to use for computation
            conflict_resolution_strategy: Strategy for resolving conflicts
                Options: "weighted_average", "priority_based", "consensus"
            verbose: Whether to print verbose output
        """
        self.state_manager = state_manager
        self.reward_calculator = reward_calculator
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.verbose = verbose
        
        # Initialize agent factory
        self.agent_factory = SegmentationAgentFactory(
            state_manager=state_manager,
            device=self.device,
            verbose=verbose
        )
        
        # Initialize agents dictionary
        self.agents = {}
        
        # Set default update priorities
        self.update_priorities = {
            "region": 1,  # Highest priority
            "boundary": 2,
            "fg_balance": 3,
            "object_detection": 4,
            "shape": 5  # Lowest priority
        }
        
        # Initialize action history
        self.action_history = []
        
        # Set up logging
        self.logger = logging.getLogger("SegmentationAgentCoordinator")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        if self.verbose:
            self.logger.info("Initialized SegmentationAgentCoordinator")
            self.logger.info(f"Conflict resolution strategy: {conflict_resolution_strategy}")
    
    def create_agents(self, config: Dict[str, Dict[str, Any]] = None) -> None:
        """
        Create all segmentation agents with optional custom configurations.
        
        Args:
            config: Dictionary of agent configurations
        """
        self.agents = self.agent_factory.create_all_agents(config)
        
        if self.verbose:
            self.logger.info(f"Created {len(self.agents)} agents: {', '.join(self.agents.keys())}")
    
    def load_agents(self, directory: str) -> None:
        """
        Load agents from files.
        
        Args:
            directory: Directory to load agents from
        """
        self.agents = self.agent_factory.load_agents(directory)
        
        if self.verbose:
            self.logger.info(f"Loaded {len(self.agents)} agents from {directory}")
    
    def save_agents(self, directory: str) -> None:
        """
        Save agents to files.
        
        Args:
            directory: Directory to save agents to
        """
        self.agent_factory.save_agents(self.agents, directory)
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.agents)} agents to {directory}")
    
    def update_agents(self) -> Dict[str, Any]:
        """
        Update all agents based on current state.
        
        Returns:
            Dictionary of agent updates and metrics
        """
        # Get current state
        current_image = self.state_manager.get_current_image()
        current_mask = self.state_manager.get_current_mask()
        current_prediction = self.state_manager.get_current_prediction()
        
        if current_image is None or current_mask is None or current_prediction is None:
            # If any required state is missing, return empty updates
            if self.verbose:
                self.logger.warning("Missing required state for agent updates")
            return {}
        
        # Calculate reward
        reward_dict = self.reward_calculator.calculate_reward(
            prediction=current_prediction,
            ground_truth=current_mask,
            include_detailed_metrics=True
        )
        
        # Update each agent
        agent_observations = {}
        agent_actions = {}
        agent_metrics = {}
        
        # Sort agents by update priority
        agent_items = sorted(
            self.agents.items(),
            key=lambda x: self.update_priorities.get(x[0], 999)
        )
        
        for agent_name, agent in agent_items:
            # Get agent observation
            observation = agent.observe()
            agent_observations[agent_name] = observation
            
            # Get agent action
            action = agent.decide(observation)
            agent_actions[agent_name] = action
            
            # Calculate agent-specific reward
            agent_reward = self._calculate_agent_specific_reward(agent_name, reward_dict)
            
            # Let agent learn from reward
            metrics = agent.learn(agent_reward)
            agent_metrics[agent_name] = metrics
        
        # Resolve conflicts between agent actions
        resolved_actions = self._resolve_conflicts(agent_actions)
        
        # Apply resolved actions to state
        self._apply_actions(resolved_actions)
        
        # Store action history
        self.action_history.append({
            "agent_actions": agent_actions,
            "resolved_actions": resolved_actions,
            "reward": reward_dict["total"],
            "timestamp": time.time()
        })
        
        # Create update dictionary
        updates = {
            "observations": agent_observations,
            "actions": agent_actions,
            "resolved_actions": resolved_actions,
            "metrics": agent_metrics,
            "reward": reward_dict
        }
        
        return updates
    
    def _calculate_agent_specific_reward(
        self,
        agent_name: str,
        reward_dict: Dict[str, Any]
    ) -> float:
        """
        Calculate agent-specific reward based on the agent's specialty.
        
        Args:
            agent_name: Name of the agent
            reward_dict: Dictionary of reward components
            
        Returns:
            Agent-specific reward
        """
        # Get normalized rewards if available
        if "normalized" in reward_dict:
            rewards = reward_dict["normalized"]
        else:
            rewards = {
                "dice": reward_dict["dice"],
                "boundary": reward_dict["boundary"],
                "object_f1": reward_dict["object_f1"],
                "shape": reward_dict["shape"],
                "fg_balance": reward_dict["fg_balance"]
            }
        
        # Calculate agent-specific reward based on agent type
        if agent_name == "region":
            # Region agent focuses on Dice score and object F1
            agent_reward = 0.7 * rewards["dice"] + 0.3 * rewards["object_f1"]
        elif agent_name == "boundary":
            # Boundary agent focuses on boundary accuracy
            agent_reward = rewards["boundary"]
            
            # Add detailed boundary metrics if available
            if "detailed_metrics" in reward_dict and "boundary_details" in reward_dict["detailed_metrics"]:
                boundary_details = reward_dict["detailed_metrics"]["boundary_details"]
                # Add boundary Dice coefficient to reward
                if "boundary_dice" in boundary_details:
                    agent_reward = 0.6 * agent_reward + 0.4 * boundary_details["boundary_dice"]
        elif agent_name == "shape":
            # Shape agent focuses on shape regularization
            agent_reward = rewards["shape"]
        elif agent_name == "fg_balance":
            # Foreground-background balance agent focuses on balance
            agent_reward = rewards["fg_balance"]
            
            # Add detailed foreground-background metrics if available
            if "detailed_metrics" in reward_dict and "fg_balance_details" in reward_dict["detailed_metrics"]:
                fg_details = reward_dict["detailed_metrics"]["fg_balance_details"]
                # Add weighted F1 score to reward
                if "weighted_f1" in fg_details:
                    agent_reward = 0.6 * agent_reward + 0.4 * fg_details["weighted_f1"]
        elif agent_name == "object_detection":
            # Object detection agent focuses on object detection accuracy
            agent_reward = rewards["object_f1"]
        else:
            # Default to total reward
            agent_reward = reward_dict["total"]
        
        return agent_reward
    
    def _resolve_conflicts(self, agent_actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts between agent actions.
        
        Args:
            agent_actions: Dictionary of agent actions
            
        Returns:
            Dictionary of resolved actions
        """
        if not agent_actions:
            return {}
        
        # Initialize resolved actions
        resolved_actions = {}
        
        # Resolve conflicts based on strategy
        if self.conflict_resolution_strategy == "weighted_average":
            resolved_actions = self._resolve_by_weighted_average(agent_actions)
        elif self.conflict_resolution_strategy == "priority_based":
            resolved_actions = self._resolve_by_priority(agent_actions)
        elif self.conflict_resolution_strategy == "consensus":
            resolved_actions = self._resolve_by_consensus(agent_actions)
        else:
            # Default to weighted average
            resolved_actions = self._resolve_by_weighted_average(agent_actions)
        
        if self.verbose:
            self.logger.debug(f"Resolved actions: {resolved_actions}")
        
        return resolved_actions
    
    def _resolve_by_weighted_average(self, agent_actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by weighted average of agent actions.
        
        Args:
            agent_actions: Dictionary of agent actions
            
        Returns:
            Dictionary of resolved actions
        """
        # Initialize resolved actions
        resolved_actions = {}
        
        # Get action weights based on agent priorities
        action_weights = {}
        for agent_name in agent_actions:
            # Higher priority = lower weight value = higher weight
            priority = self.update_priorities.get(agent_name, 999)
            # Convert priority to weight (inverse relationship)
            weight = 1.0 / (priority + 1)
            action_weights[agent_name] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(action_weights.values())
        if total_weight > 0:
            for agent_name in action_weights:
                action_weights[agent_name] /= total_weight
        
        # Process segmentation threshold
        thresholds = []
        threshold_weights = []
        
        for agent_name, action in agent_actions.items():
            if "segmentation_threshold" in action:
                thresholds.append(action["segmentation_threshold"])
                threshold_weights.append(action_weights[agent_name])
        
        if thresholds:
            # Calculate weighted average threshold
            resolved_actions["segmentation_threshold"] = sum(t * w for t, w in zip(thresholds, threshold_weights))
        
        return resolved_actions
    
    def _resolve_by_priority(self, agent_actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by priority-based selection of agent actions.
        
        Args:
            agent_actions: Dictionary of agent actions
            
        Returns:
            Dictionary of resolved actions
        """
        # Initialize resolved actions
        resolved_actions = {}
        
        # Sort agents by priority
        sorted_agents = sorted(
            agent_actions.keys(),
            key=lambda x: self.update_priorities.get(x, 999)
        )
        
        # Process segmentation threshold
        for agent_name in sorted_agents:
            action = agent_actions[agent_name]
            if "segmentation_threshold" in action:
                resolved_actions["segmentation_threshold"] = action["segmentation_threshold"]
                break
        
        return resolved_actions
    
    def _resolve_by_consensus(self, agent_actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by consensus of agent actions.
        
        Args:
            agent_actions: Dictionary of agent actions
            
        Returns:
            Dictionary of resolved actions
        """
        # Initialize resolved actions
        resolved_actions = {}
        
        # Process segmentation threshold
        thresholds = []
        
        for agent_name, action in agent_actions.items():
            if "segmentation_threshold" in action:
                thresholds.append(action["segmentation_threshold"])
        
        if thresholds:
            # Calculate median threshold
            resolved_actions["segmentation_threshold"] = np.median(thresholds)
        
        return resolved_actions
    
    def _apply_actions(self, actions: Dict[str, Any]) -> None:
        """
        Apply resolved actions to state.
        
        Args:
            actions: Dictionary of resolved actions
        """
        if "segmentation_threshold" in actions:
            self.state_manager.set_segmentation_threshold(actions["segmentation_threshold"])
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all agents.
        
        Returns:
            Dictionary of agent states
        """
        agent_states = {}
        
        for agent_name, agent in self.agents.items():
            agent_states[agent_name] = agent.get_state()
        
        return agent_states
    
    def set_agent_states(self, agent_states: Dict[str, Dict[str, Any]]) -> None:
        """
        Set the state of all agents.
        
        Args:
            agent_states: Dictionary of agent states
        """
        for agent_name, state in agent_states.items():
            if agent_name in self.agents:
                self.agents[agent_name].set_state(state)
    
    def set_update_priorities(self, priorities: Dict[str, int]) -> None:
        """
        Set the update priorities for agents.
        
        Args:
            priorities: Dictionary of agent priorities
        """
        self.update_priorities.update(priorities)
        
        if self.verbose:
            self.logger.info(f"Updated agent priorities: {self.update_priorities}")
    
    def set_conflict_resolution_strategy(self, strategy: str) -> None:
        """
        Set the conflict resolution strategy.
        
        Args:
            strategy: Conflict resolution strategy
                Options: "weighted_average", "priority_based", "consensus"
        """
        valid_strategies = ["weighted_average", "priority_based", "consensus"]
        
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid conflict resolution strategy: {strategy}. Must be one of {valid_strategies}")
        
        self.conflict_resolution_strategy = strategy
        
        if self.verbose:
            self.logger.info(f"Set conflict resolution strategy to: {strategy}")
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get the action history.
        
        Returns:
            List of action history entries
        """
        return self.action_history
    
    def clear_action_history(self) -> None:
        """
        Clear the action history.
        """
        self.action_history = []
        
        if self.verbose:
            self.logger.info("Cleared action history")
