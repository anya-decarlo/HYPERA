#!/usr/bin/env python
# Neural Network Models for SAC Implementation in Segmentation Agents

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class ValueNetwork(nn.Module):
    """
    Value Network for SAC.
    
    Estimates the value of a state.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Dimension of hidden layers
        """
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Value estimate
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

class QNetwork(nn.Module):
    """
    Q-Network for SAC.
    
    Estimates the Q-value (expected return) of a state-action pair.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (for min double-Q learning)
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of Q-values from both networks
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        
        # Q2 forward pass
        q2 = F.relu(self.linear4(x))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through just the first Q-network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value from the first network
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        
        return q1

class GaussianPolicy(nn.Module):
    """
    Gaussian Policy Network for SAC.
    
    Outputs a Gaussian distribution over actions given a state.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_space: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            action_space: Tuple of (min_action, max_action)
        """
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        # Action space bounds
        self.action_dim = action_dim
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            # Ensure action_space is properly handled as a tuple of two values
            min_action, max_action = action_space
            self.action_scale = torch.tensor((max_action - min_action) / 2.0)
            self.action_bias = torch.tensor((max_action + min_action) / 2.0)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (mean, log_std) of the Gaussian distribution
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        
        # Scale and shift action
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh squashing correction
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean
    
    def to(self, device):
        """Move the model to the specified device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        
        # Move all nn.Module components to the device
        for attr_str in dir(self):
            attr = getattr(self, attr_str)
            if isinstance(attr, nn.Module):
                attr.to(device)
                
        return self
