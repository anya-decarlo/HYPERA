#!/usr/bin/env python
# SAC (Soft Actor-Critic) Implementation for Segmentation Agents

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json
import logging

from .networks import ValueNetwork, QNetwork, GaussianPolicy
from ..replay_buffer import ReplayBuffer

class SAC:
    """
    Soft Actor-Critic (SAC) algorithm implementation for segmentation agents.
    
    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It combines off-policy updates with a stable
    stochastic actor-critic formulation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space: Tuple[float, float] = (-1.0, 1.0),
        hidden_dim: int = 256,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        automatic_entropy_tuning: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "logs",
        name: str = "sac"
    ):
        """
        Initialize the SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Tuple of (min_action, max_action)
            hidden_dim: Dimension of hidden layers in networks
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Target network update rate
            alpha: Temperature parameter for entropy
            lr: Learning rate
            automatic_entropy_tuning: Whether to automatically tune entropy
            device: Device to use for tensors
            log_dir: Directory to save logs
            name: Name of the agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device
        self.log_dir = log_dir
        self.name = name
        
        # Initialize critic networks
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy critic parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Initialize actor network
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim, action_space).to(device)
        
        # Initialize optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize target entropy and alpha optimizer if automatic tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Training metrics
        self.critic_loss_history = []
        self.policy_loss_history = []
        self.alpha_loss_history = []
        self.entropy_history = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging for the SAC agent."""
        log_path = os.path.join(self.log_dir, f"{self.name}_sac")
        os.makedirs(log_path, exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join(log_path, f"{self.name}_sac_log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[file_handler, console_handler]
        )
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action using the policy.
        
        Args:
            state: Current state
            evaluate: Whether to evaluate (deterministic) or explore
            
        Returns:
            Selected action
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            # Use mean action (deterministic)
            with torch.no_grad():
                _, _, action = self.policy.sample(state)
                return action.cpu().numpy().flatten()
        else:
            # Sample action (stochastic)
            with torch.no_grad():
                action, _, _ = self.policy.sample(state)
                return action.cpu().numpy().flatten()
    
    def update_parameters(self, updates: int = 1) -> Dict[str, float]:
        """
        Update the network parameters using SAC.
        
        Args:
            updates: Number of updates to perform
            
        Returns:
            Dictionary of losses
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {
                'critic_loss': 0.0,
                'policy_loss': 0.0,
                'alpha_loss': 0.0,
                'entropy': 0.0
            }
        
        # Track losses
        critic_losses = []
        policy_losses = []
        alpha_losses = []
        entropies = []
        
        for _ in range(updates):
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.to_torch(self.batch_size)
            
            # Get current alpha
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha
            
            # ---------- Update critic ----------
            with torch.no_grad():
                # Sample next actions and log probs from current policy
                next_actions, next_log_probs, _ = self.policy.sample(next_states)
                
                # Compute target Q values
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - dones) * self.gamma * (target_q - alpha * next_log_probs)
            
            # Compute current Q values
            current_q1, current_q2 = self.critic(states, actions)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            critic_losses.append(critic_loss.item())
            
            # ---------- Update policy ----------
            # Sample actions and log probs from current policy
            pi_actions, log_probs, _ = self.policy.sample(states)
            
            # Compute Q values for policy actions
            q1_pi, q2_pi = self.critic(states, pi_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # Compute policy loss
            policy_loss = (alpha * log_probs - min_q_pi).mean()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            entropies.append(-log_probs.mean().item())
            
            # ---------- Update alpha ----------
            if self.automatic_entropy_tuning:
                # Compute alpha loss
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                # Update alpha
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                alpha_losses.append(alpha_loss.item())
            else:
                alpha_loss = torch.tensor(0.0)
                alpha_losses.append(0.0)
            
            # ---------- Update target networks ----------
            # Soft update of the target networks
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # Update history
        self.critic_loss_history.extend(critic_losses)
        self.policy_loss_history.extend(policy_losses)
        self.alpha_loss_history.extend(alpha_losses)
        self.entropy_history.extend(entropies)
        
        # Return average losses
        return {
            'critic_loss': np.mean(critic_losses),
            'policy_loss': np.mean(policy_losses),
            'alpha_loss': np.mean(alpha_losses),
            'entropy': np.mean(entropies)
        }
    
    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def save_models(self, path: str = None) -> None:
        """
        Save the model parameters.
        
        Args:
            path: Path to save models to (default: log_dir/name_sac)
        """
        if path is None:
            path = os.path.join(self.log_dir, f"{self.name}_sac")
        
        os.makedirs(path, exist_ok=True)
        
        # Save models
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        
        # Save alpha if automatic tuning
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, os.path.join(path, "log_alpha.pth"))
        
        # Save hyperparameters
        hyperparams = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_space': self.action_space,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'lr': self.lr,
            'automatic_entropy_tuning': self.automatic_entropy_tuning
        }
        
        with open(os.path.join(path, "hyperparams.json"), 'w') as f:
            json.dump(hyperparams, f, indent=4)
        
        # Save training history
        history = {
            'critic_loss': self.critic_loss_history,
            'policy_loss': self.policy_loss_history,
            'alpha_loss': self.alpha_loss_history,
            'entropy': self.entropy_history
        }
        
        with open(os.path.join(path, "history.json"), 'w') as f:
            json.dump(history, f, indent=4)
        
        logging.info(f"Saved SAC models to {path}")
    
    def load_models(self, path: str) -> None:
        """
        Load the model parameters.
        
        Args:
            path: Path to load models from
        """
        # Load models
        self.policy.load_state_dict(torch.load(os.path.join(path, "policy.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))
        
        # Load alpha if automatic tuning
        if self.automatic_entropy_tuning and os.path.exists(os.path.join(path, "log_alpha.pth")):
            self.log_alpha = torch.load(os.path.join(path, "log_alpha.pth"))
        
        # Load hyperparameters
        if os.path.exists(os.path.join(path, "hyperparams.json")):
            with open(os.path.join(path, "hyperparams.json"), 'r') as f:
                hyperparams = json.load(f)
                
                # Update hyperparameters
                for key, value in hyperparams.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        
        # Load training history
        if os.path.exists(os.path.join(path, "history.json")):
            with open(os.path.join(path, "history.json"), 'r') as f:
                history = json.load(f)
                
                # Update history
                self.critic_loss_history = history.get('critic_loss', [])
                self.policy_loss_history = history.get('policy_loss', [])
                self.alpha_loss_history = history.get('alpha_loss', [])
                self.entropy_history = history.get('entropy', [])
        
        logging.info(f"Loaded SAC models from {path}")
        
    def get_current_alpha(self) -> float:
        """
        Get the current alpha value.
        
        Returns:
            Current alpha value
        """
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp().item()
        else:
            return self.alpha
