# SAC Implementation for Segmentation Agents

This directory contains the implementation of Soft Actor-Critic (SAC) for the segmentation agents in the HYPERA system.

## Overview

Soft Actor-Critic (SAC) is an off-policy actor-critic deep reinforcement learning algorithm based on the maximum entropy reinforcement learning framework. It combines off-policy updates with a stable stochastic actor-critic formulation, making it particularly suitable for continuous action spaces.

## Components

### 1. Networks (`networks.py`)

Contains the neural network architectures used in SAC:

- **ValueNetwork**: Estimates the value of a state.
- **QNetwork**: Estimates the Q-value (expected return) of a state-action pair.
- **GaussianPolicy**: Outputs a Gaussian distribution over actions given a state.

### 2. Replay Buffer (`replay_buffer.py`)

Implements experience replay for storing and sampling transitions:

- Stores transitions (state, action, reward, next_state, done)
- Provides functionality to sample random batches for training
- Supports conversion to PyTorch tensors

### 3. SAC Algorithm (`sac.py`)

Implements the SAC algorithm with:

- Off-policy learning with experience replay
- Separate actor and critic networks
- Entropy regularization for exploration
- Automatic temperature tuning
- Target networks for stable learning
- Soft updates for critic target networks

## Integration with Segmentation Agents

The SAC implementation is integrated into the segmentation agents through the `BaseSegmentationAgent` class, which provides:

1. A consistent interface for all segmentation agents
2. SAC-based reinforcement learning capabilities
3. Methods for saving and loading agent state

Each specialized agent (e.g., RegionAgent, BoundaryAgent) inherits from `BaseSegmentationAgent` and implements:

- `get_state_representation()`: Extracts a state representation from observations
- `apply_action()`: Applies SAC actions to produce segmentation decisions
- `_extract_features()`: Extracts features from observations

## Usage

The SAC components are automatically used when a segmentation agent is created. The agent's `act()` and `learn()` methods use SAC's `select_action()` and `update_parameters()` methods internally.

## Parameters

The SAC implementation can be configured with the following parameters:

- `state_dim`: Dimension of state space
- `action_dim`: Dimension of action space
- `action_space`: Tuple of (min_action, max_action)
- `hidden_dim`: Dimension of hidden layers in networks
- `replay_buffer_size`: Size of replay buffer
- `batch_size`: Batch size for training
- `gamma`: Discount factor
- `tau`: Target network update rate
- `alpha`: Temperature parameter for entropy
- `lr`: Learning rate
- `automatic_entropy_tuning`: Whether to automatically tune entropy

## References

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290.
2. Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.
