#!/usr/bin/env python
# Segmentation Agent Factory - Creates and manages segmentation agents

import os
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from .region_agent import RegionAgent
from .boundary_agent import BoundaryAgent
from .shape_agent import ShapeAgent
from .fg_balance_agent import FGBalanceAgent
from .object_detection_agent import ObjectDetectionAgent
from ..segmentation_state_manager import SegmentationStateManager

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
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 1
    ) -> RegionAgent:
        """
        Create a region agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            
        Returns:
            Initialized region agent
        """
        agent = RegionAgent(
            state_manager=self.state_manager,
            device=self.device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created RegionAgent with learning_rate={learning_rate}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_boundary_agent(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 3
    ) -> BoundaryAgent:
        """
        Create a boundary agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            
        Returns:
            Initialized boundary agent
        """
        agent = BoundaryAgent(
            state_manager=self.state_manager,
            device=self.device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created BoundaryAgent with learning_rate={learning_rate}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_shape_agent(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 5
    ) -> ShapeAgent:
        """
        Create a shape agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            
        Returns:
            Initialized shape agent
        """
        agent = ShapeAgent(
            state_manager=self.state_manager,
            device=self.device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created ShapeAgent with learning_rate={learning_rate}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_fg_balance_agent(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 2
    ) -> FGBalanceAgent:
        """
        Create a foreground-background balance agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            
        Returns:
            Initialized foreground-background balance agent
        """
        agent = FGBalanceAgent(
            state_manager=self.state_manager,
            device=self.device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created FGBalanceAgent with learning_rate={learning_rate}, gamma={gamma}, update_frequency={update_frequency}")
        
        return agent
    
    def create_object_detection_agent(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        update_frequency: int = 1
    ) -> ObjectDetectionAgent:
        """
        Create an object detection agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            update_frequency: Frequency of agent updates
            
        Returns:
            Initialized object detection agent
        """
        agent = ObjectDetectionAgent(
            state_manager=self.state_manager,
            device=self.device,
            learning_rate=learning_rate,
            gamma=gamma,
            update_frequency=update_frequency,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Created ObjectDetectionAgent with learning_rate={learning_rate}, gamma={gamma}, update_frequency={update_frequency}")
        
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
            agents["region"] = self.create_region_agent(**config["region"])
        
        # Create boundary agent if in config
        if "boundary" in config:
            agents["boundary"] = self.create_boundary_agent(**config["boundary"])
        
        # Create shape agent if in config
        if "shape" in config:
            agents["shape"] = self.create_shape_agent(**config["shape"])
        
        # Create foreground-background balance agent if in config
        if "fg_balance" in config:
            agents["fg_balance"] = self.create_fg_balance_agent(**config["fg_balance"])
        
        # Create object detection agent if in config
        if "object_detection" in config:
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
