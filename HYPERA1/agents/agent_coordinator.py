#!/usr/bin/env python
# Agent Coordinator for Multi-Agent Hyperparameter Optimization

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

from .shared_state import SharedStateManager
from .base_agent import BaseHyperparameterAgent

class AgentCoordinator:
    """
    Coordinates multiple hyperparameter agents, manages updates, and resolves conflicts.
    
    This class is responsible for:
    1. Managing the collection of hyperparameter agents
    2. Coordinating agent updates based on their update frequencies
    3. Resolving conflicts when multiple agents want to update the same hyperparameter
    4. Logging agent activities and decisions
    """
    
    def __init__(
        self,
        shared_state_manager: SharedStateManager,
        agents: Optional[List[BaseHyperparameterAgent]] = None,
        update_frequency: int = 1,
        conflict_resolution_strategy: str = "priority",
        log_dir: Optional[str] = None,
        verbose: bool = False,
        change_callback: Optional[callable] = None
    ):
        """
        Initialize the agent coordinator.
        
        Args:
            shared_state_manager: Shared state manager for metrics and hyperparameters
            agents: List of hyperparameter agents (optional)
            update_frequency: Global update frequency for all agents
            conflict_resolution_strategy: Strategy for resolving conflicts ("priority", "voting", "consensus")
            log_dir: Directory for saving logs and agent states
            verbose: Whether to print verbose output
            change_callback: Optional callback function to be called when an agent's action is applied
        """
        self.shared_state_manager = shared_state_manager
        self.agents = {} if agents is None else {agent.name: agent for agent in agents}
        self.update_frequency = update_frequency
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.log_dir = log_dir
        self.verbose = verbose
        self.change_callback = change_callback
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def get_agent(self, agent_name: str) -> Optional[BaseHyperparameterAgent]:
        """
        Get an agent by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            The agent if found, None otherwise
        """
        return self.agents.get(agent_name)
    
    def add_agent(self, agent: BaseHyperparameterAgent, priority: int = 0) -> None:
        """
        Add an agent to the coordinator.
        
        Args:
            agent: Hyperparameter agent to add
            priority: Priority of the agent (higher means more important)
        """
        agent.priority = priority
        self.agents[agent.name] = agent
        if self.verbose:
            self.logger.info(f"Added agent: {agent.name} with priority {priority}")
    
    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the coordinator.
        
        Args:
            agent_name: Name of the agent to remove
            
        Returns:
            True if the agent was removed, False otherwise
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            if self.verbose:
                self.logger.info(f"Removed agent: {agent_name}")
            return True
        return False
    
    def register_agent(self, agent: BaseHyperparameterAgent):
        """
        Register a new agent with the coordinator.
        
        Args:
            agent: The agent to register
        """
        if agent.name in self.agents:
            logging.warning(f"Agent with name {agent.name} already registered. Overwriting.")
        
        self.agents[agent.name] = agent
        if self.verbose:
            logging.info(f"Registered agent: {agent.name}")
            
    def register_agents(self, agents: List[BaseHyperparameterAgent]):
        """
        Register multiple agents with the coordinator.
        
        Args:
            agents: List of agents to register
        """
        for agent in agents:
            self.register_agent(agent)
    
    def update(self, epoch: int) -> Dict[str, Any]:
        """
        Update all agents and resolve conflicts.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary with update information
        """
        if not self.agents:
            return {}
        
        # Check if any agent should update in this epoch
        agents_to_update = []
        for agent_name, agent in self.agents.items():
            if epoch % agent.update_frequency == 0:
                agents_to_update.append(agent)
        
        if not agents_to_update:
            return {}
        
        # Collect proposed actions from each agent
        proposed_actions = {}
        for agent in agents_to_update:
            action = agent.select_action(epoch)
            if action is not None:
                proposed_actions[agent.name] = {
                    "agent": agent,
                    "action": action,
                    "priority": agent.priority
                }
        
        if not proposed_actions:
            return {}
        
        # Resolve conflicts and apply actions
        resolved_actions = self._resolve_conflicts(proposed_actions)
        
        # Apply resolved actions
        applied_actions = {}
        for param_name, action_info in resolved_actions.items():
            agent = action_info["agent"]
            action = action_info["action"]
            
            # Apply the action
            result = agent.apply_action(action, epoch)
            
            if result:
                applied_actions[param_name] = result
                if self.verbose:
                    self.logger.info(f"Applied action from {agent.name}: {param_name} = {result}")
                
                # Call change callback if provided
                if self.change_callback:
                    self.change_callback(param_name, result)
        
        return applied_actions
    
    def _resolve_conflicts(self, proposed_actions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Resolve conflicts between proposed actions.
        
        Args:
            proposed_actions: Dictionary of proposed actions from each agent
            
        Returns:
            Dictionary of resolved actions
        """
        # Group proposed actions by hyperparameter
        param_actions = defaultdict(list)
        for agent_name, action_info in proposed_actions.items():
            agent = action_info["agent"]
            action = action_info["action"]
            priority = action_info["priority"]
            
            param_name = agent.get_param_name()
            param_actions[param_name].append({
                "agent": agent,
                "action": action,
                "priority": priority
            })
        
        # Resolve conflicts for each hyperparameter
        resolved_actions = {}
        for param_name, actions in param_actions.items():
            if len(actions) == 1:
                # No conflict
                resolved_actions[param_name] = actions[0]
            else:
                # Conflict detected
                if self.verbose:
                    self.logger.info(f"Conflict detected for {param_name} with {len(actions)} agents")
                
                # Resolve based on strategy
                if self.conflict_resolution_strategy == "priority":
                    # Choose action from agent with highest priority
                    resolved_action = max(actions, key=lambda x: x["priority"])
                    resolved_actions[param_name] = resolved_action
                    
                    if self.verbose:
                        self.logger.info(f"Resolved conflict for {param_name} using priority strategy: "
                                         f"Selected {resolved_action['agent'].name}")
                
                elif self.conflict_resolution_strategy == "voting":
                    # Implement voting strategy (e.g., majority vote)
                    # This is a simplified implementation
                    action_values = [a["action"] for a in actions]
                    unique_values, counts = np.unique(action_values, return_counts=True)
                    winner_idx = np.argmax(counts)
                    winner_value = unique_values[winner_idx]
                    
                    # Find an agent that proposed this value
                    for action_info in actions:
                        if action_info["action"] == winner_value:
                            resolved_actions[param_name] = action_info
                            break
                    
                    if self.verbose:
                        self.logger.info(f"Resolved conflict for {param_name} using voting strategy: "
                                         f"Selected value {winner_value}")
                
                elif self.conflict_resolution_strategy == "consensus":
                    # Implement consensus strategy (e.g., average of proposed values)
                    # This assumes actions are numeric values
                    try:
                        avg_value = np.mean([a["action"] for a in actions])
                        # Assign to the highest priority agent
                        best_agent = max(actions, key=lambda x: x["priority"])
                        resolved_actions[param_name] = {
                            "agent": best_agent["agent"],
                            "action": avg_value,
                            "priority": best_agent["priority"]
                        }
                        
                        if self.verbose:
                            self.logger.info(f"Resolved conflict for {param_name} using consensus strategy: "
                                             f"Selected average value {avg_value}")
                    except:
                        # Fallback to priority if consensus fails
                        resolved_action = max(actions, key=lambda x: x["priority"])
                        resolved_actions[param_name] = resolved_action
                        
                        if self.verbose:
                            self.logger.info(f"Consensus strategy failed for {param_name}, "
                                             f"falling back to priority: Selected {resolved_action['agent'].name}")
                else:
                    # Unknown strategy, fall back to priority
                    resolved_action = max(actions, key=lambda x: x["priority"])
                    resolved_actions[param_name] = resolved_action
                    
                    if self.verbose:
                        self.logger.info(f"Unknown conflict resolution strategy, "
                                         f"falling back to priority for {param_name}: "
                                         f"Selected {resolved_action['agent'].name}")
        
        return resolved_actions
    
    def save_agents(self, save_dir: Optional[str] = None) -> None:
        """
        Save all agents to disk.
        
        Args:
            save_dir: Directory to save agents to (defaults to self.log_dir)
        """
        save_dir = save_dir or self.log_dir
        if save_dir is None:
            self.logger.warning("No save directory specified, agents will not be saved")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            agent_save_path = os.path.join(save_dir, f"{agent_name}.pt")
            try:
                agent.save(agent_save_path)
                if self.verbose:
                    self.logger.info(f"Saved agent {agent_name} to {agent_save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save agent {agent_name}: {e}")
    
    def load_agents(self, load_dir: Optional[str] = None) -> None:
        """
        Load all agents from disk.
        
        Args:
            load_dir: Directory to load agents from (defaults to self.log_dir)
        """
        load_dir = load_dir or self.log_dir
        if load_dir is None:
            self.logger.warning("No load directory specified, agents will not be loaded")
            return
        
        for agent_name, agent in self.agents.items():
            agent_load_path = os.path.join(load_dir, f"{agent_name}.pt")
            if os.path.exists(agent_load_path):
                try:
                    agent.load(agent_load_path)
                    if self.verbose:
                        self.logger.info(f"Loaded agent {agent_name} from {agent_load_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load agent {agent_name}: {e}")
            else:
                self.logger.warning(f"Agent file not found for {agent_name} at {agent_load_path}")
