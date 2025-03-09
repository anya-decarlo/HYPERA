#!/usr/bin/env python
# Segmentation Agent Factory - Creates and manages segmentation agents

import os
import sys
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HYPERA1.segmentation.agents.region_agent import RegionAgent
from HYPERA1.segmentation.agents.boundary_agent import BoundaryAgent
from HYPERA1.segmentation.agents.shape_agent import ShapeAgent
from HYPERA1.segmentation.agents.fg_balance_agent import FGBalanceAgent
from HYPERA1.segmentation.agents.object_detection_agent import ObjectDetectionAgent
from HYPERA1.segmentation.segmentation_state_manager import SegmentationStateManager

class SegmentationAgentFactory:
    """
    Factory class for creating and managing segmentation agents.
    
    This class provides methods for creating individual agents or sets of agents
    with custom configurations. It also handles agent initialization, configuration,
    and provides a unified interface for agent creation.
    
    Attributes:
        state_manager (SegmentationStateManager): Manager for shared state
        device (torch.device): Device to use for computation
        verbose (bool): Whether to print verbose output
        logger (logging.Logger): Logger for the agent factory
    """
    
    def __init__(
        self,
        state_manager: SegmentationStateManager,
        device: torch.device = None,
        verbose: bool = False
    ):
        """
        Initialize the segmentation agent factory.
        
        Args:
            state_manager: Manager for shared state
            device: Device to use for computation
            verbose: Whether to print verbose output
        """
        self.state_manager = state_manager
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Set up logging
        self.logger = logging.getLogger("SegmentationAgentFactory")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        if self.verbose:
            self.logger.info("Initialized SegmentationAgentFactory")
    
    def create_region_agent(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        update_frequency: int = 1,
        state_dim: int = 128,
        action_dim: int = 5,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        log_dir: str = "logs"
    ) -> RegionAgent:
        """
        Create a region agent.
        
        Args:
            lr: Learning rate
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            state_dim: Dimension of state representation
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            automatic_entropy_tuning: Whether to automatically tune entropy
            log_dir: Directory for saving logs and checkpoints
            
        Returns:
            Initialized region agent
        """
        agent = RegionAgent(
            state_manager=self.state_manager,
            device=self.device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            automatic_entropy_tuning=automatic_entropy_tuning,
            update_frequency=update_frequency,
            log_dir=log_dir,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created RegionAgent with lr={lr}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_boundary_agent(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        update_frequency: int = 3,
        state_dim: int = 10,
        action_dim: int = 1,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        log_dir: str = "logs"
    ):
        """
        Create a boundary agent.
        
        Args:
            lr: Learning rate
            gamma: Discount factor
            update_frequency: Frequency of agent updates
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            automatic_entropy_tuning: Whether to automatically tune entropy
            log_dir: Directory for saving logs and checkpoints
            
        Returns:
            Initialized boundary agent
        """
        agent = BoundaryAgent(
            state_manager=self.state_manager,
            device=self.device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            automatic_entropy_tuning=automatic_entropy_tuning,
            update_frequency=update_frequency,
            log_dir=log_dir,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created BoundaryAgent with lr={lr}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_shape_agent(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        update_frequency: int = 5,
        state_dim: int = 10,
        action_dim: int = 1,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        log_dir: str = "logs"
    ) -> ShapeAgent:
        """
        Create a shape agent.
        
        Args:
            lr: Learning rate
            gamma: Discount factor
            update_frequency: Frequency of agent updates
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            automatic_entropy_tuning: Whether to automatically tune entropy
            log_dir: Directory for saving logs and checkpoints
            
        Returns:
            Initialized shape agent
        """
        agent = ShapeAgent(
            state_manager=self.state_manager,
            device=self.device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            automatic_entropy_tuning=automatic_entropy_tuning,
            update_frequency=update_frequency,
            log_dir=log_dir,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created ShapeAgent with lr={lr}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_fg_balance_agent(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        update_frequency: int = 2,
        state_dim: int = 10,
        action_dim: int = 1,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        log_dir: str = "logs"
    ) -> FGBalanceAgent:
        """
        Create a foreground-background balance agent.
        
        Args:
            lr: Learning rate
            gamma: Discount factor
            update_frequency: Frequency of agent updates
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            automatic_entropy_tuning: Whether to automatically tune entropy
            log_dir: Directory for saving logs and checkpoints
            
        Returns:
            Initialized foreground-background balance agent
        """
        agent = FGBalanceAgent(
            state_manager=self.state_manager,
            device=self.device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            automatic_entropy_tuning=automatic_entropy_tuning,
            update_frequency=update_frequency,
            log_dir=log_dir,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created FGBalanceAgent with lr={lr}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_object_detection_agent(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        update_frequency: int = 1,
        state_dim: int = 10,
        action_dim: int = 1,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        log_dir: str = "logs"
    ) -> ObjectDetectionAgent:
        """
        Create an object detection agent.
        
        Args:
            lr: Learning rate
            gamma: Discount factor
            update_frequency: Frequency of agent updates
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            automatic_entropy_tuning: Whether to automatically tune entropy
            log_dir: Directory for saving logs and checkpoints
            
        Returns:
            Initialized object detection agent
        """
        agent = ObjectDetectionAgent(
            state_manager=self.state_manager,
            device=self.device,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            automatic_entropy_tuning=automatic_entropy_tuning,
            update_frequency=update_frequency,
            log_dir=log_dir,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created ObjectDetectionAgent with lr={lr}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_all_agents(
        self,
        config: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create all segmentation agents with optional custom configurations.
        
        Args:
            config: Dictionary of agent configurations
                Example:
                {
                    "region": {
                        "learning_rate": 0.001,
                        "gamma": 0.99,
                        "update_frequency": 1
                    },
                    "boundary": {
                        "learning_rate": 0.001,
                        "gamma": 0.99,
                        "update_frequency": 3
                    },
                    ...
                }
            
        Returns:
            Dictionary of initialized agents
        """
        # Set default config if not provided
        if config is None:
            config = {
                "region": {},
                "boundary": {},
                "shape": {},
                "fg_balance": {},
                "object_detection": {}
            }
        
        # Create agents with custom configurations
        agents = {}
        
        # Create region agent if in config
        if "region" in config:
            # Convert learning_rate to lr if present
            if "learning_rate" in config["region"]:
                config["region"]["lr"] = config["region"].pop("learning_rate")
            agents["region"] = self.create_region_agent(**config["region"])
        
        # Create boundary agent if in config
        if "boundary" in config:
            # Convert learning_rate to lr if present
            if "learning_rate" in config["boundary"]:
                config["boundary"]["lr"] = config["boundary"].pop("learning_rate")
            agents["boundary"] = self.create_boundary_agent(**config["boundary"])
        
        # Create shape agent if in config
        if "shape" in config:
            # Convert learning_rate to lr if present
            if "learning_rate" in config["shape"]:
                config["shape"]["lr"] = config["shape"].pop("learning_rate")
            agents["shape"] = self.create_shape_agent(**config["shape"])
        
        # Create foreground-background balance agent if in config
        if "fg_balance" in config:
            # Convert learning_rate to lr if present
            if "learning_rate" in config["fg_balance"]:
                config["fg_balance"]["lr"] = config["fg_balance"].pop("learning_rate")
            agents["fg_balance"] = self.create_fg_balance_agent(**config["fg_balance"])
        
        # Create object detection agent if in config
        if "object_detection" in config:
            # Convert learning_rate to lr if present
            if "learning_rate" in config["object_detection"]:
                config["object_detection"]["lr"] = config["object_detection"].pop("learning_rate")
            agents["object_detection"] = self.create_object_detection_agent(**config["object_detection"])
        
        if self.verbose:
            self.logger.info(f"Created {len(agents)} agents: {', '.join(agents.keys())}")
        
        return agents
    
    def load_agent(self, agent_type: str, path: str) -> Any:
        """
        Load an agent from a file.
        
        Args:
            agent_type: Type of agent to load (region, boundary, shape, fg_balance, object_detection)
            path: Path to load the agent from
            
        Returns:
            Loaded agent
        """
        # Create empty agent of the specified type
        if agent_type == "region":
            agent = self.create_region_agent()
        elif agent_type == "boundary":
            agent = self.create_boundary_agent()
        elif agent_type == "shape":
            agent = self.create_shape_agent()
        elif agent_type == "fg_balance":
            agent = self.create_fg_balance_agent()
        elif agent_type == "object_detection":
            agent = self.create_object_detection_agent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Load agent from file
        agent.load(path)
        
        if self.verbose:
            self.logger.info(f"Loaded {agent_type} agent from {path}")
        
        return agent
    
    def save_agents(self, agents: Dict[str, Any], directory: str) -> None:
        """
        Save agents to files.
        
        Args:
            agents: Dictionary of agents to save
            directory: Directory to save agents to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each agent
        for agent_type, agent in agents.items():
            path = os.path.join(directory, f"{agent_type}_agent.pt")
            agent.save(path)
            
            if self.verbose:
                self.logger.info(f"Saved {agent_type} agent to {path}")
    
    def load_agents(self, directory: str) -> Dict[str, Any]:
        """
        Load agents from files.
        
        Args:
            directory: Directory to load agents from
            
        Returns:
            Dictionary of loaded agents
        """
        agents = {}
        
        # Load each agent if file exists
        agent_types = ["region", "boundary", "shape", "fg_balance", "object_detection"]
        
        for agent_type in agent_types:
            path = os.path.join(directory, f"{agent_type}_agent.pt")
            
            if os.path.exists(path):
                agents[agent_type] = self.load_agent(agent_type, path)
        
        if self.verbose:
            self.logger.info(f"Loaded {len(agents)} agents from {directory}")
        
        return agents
