#!/usr/bin/env python
# Replay Buffer for SAC Implementation in Segmentation Agents

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from collections import deque

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience tuples for SAC.
    
    This buffer stores transitions (state, action, reward, next_state, done)
    and provides functionality to sample random batches for training.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 10,
        action_dim: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to store tensors on (cpu or cuda)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize buffer
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Ensure state and action are numpy arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        if not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.float32)
        
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        # Add transition to buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Ensure we have enough samples
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.float32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def to_torch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions and convert to PyTorch tensors.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors
        """
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
