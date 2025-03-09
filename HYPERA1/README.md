# HYPERA: Multi-Agent Hyperparameter Optimization System

HYPERA is an advanced hyperparameter optimization system for medical image segmentation models using a multi-agent reinforcement learning approach. Each hyperparameter is managed by a specialized agent that can dynamically adjust its value during training to optimize model performance.

## Key Features

- **Multi-Agent Architecture**: Each hyperparameter has its own dedicated agent
- **Asynchronous Updates**: Agents operate at different frequencies based on hyperparameter characteristics
- **Conflict Resolution**: Built-in mechanisms to resolve conflicts between agent decisions
- **Shared State Management**: Centralized repository for training metrics and hyperparameter values
- **Reinforcement Learning**: Agents learn optimal policies through Q-learning
- **Extensible Design**: Easy to add new specialized agents for different hyperparameters

## Directory Structure

```
/HYPERA1.0/
├── README.md                                # Project overview and documentation
├── requirements.txt                         # Dependencies
├── run_comparison.sh                        # Comparison script
├── rl_agent_technical_explanation.md        # Technical documentation
│
├── agents/                                  # Directory for all agent-related code
│   ├── __init__.py                          # Makes agents a proper package
│   ├── base_agent.py                        # Abstract base agent class
│   ├── shared_state.py                      # Shared state manager
│   ├── agent_coordinator.py                 # Coordinator for all agents
│   │
│   ├── specialized/                         # Specialized agent implementations
│   │   ├── __init__.py                      # Makes specialized a proper package
│   │   ├── learning_rate_agent.py           # Learning rate optimization agent
│   │   └── ...                              # Other specialized agents
│   │
│   └── utils/                               # Utilities for agents
│       ├── __init__.py                      # Makes utils a proper package
│       └── ...                              # Utility modules
│
├── training/                                # Training-related code
│   ├── __init__.py                          # Makes training a proper package
│   ├── train_with_agents.py                 # Main training script with multi-agent system
│   └── ...                                  # Other training modules
│
├── evaluation/                              # Evaluation and analysis
│   ├── __init__.py                          # Makes evaluation a proper package
│   └── ...                                  # Evaluation modules
│
└── legacy/                                  # Original implementation (for reference)
    ├── rl_hyperparameter_agent.py           # Original monolithic agent
    └── train_synthetic_with_hyperparameter_agent.py  # Original training script
```

## Multi-Agent System Architecture

The multi-agent system consists of several key components:

1. **BaseHyperparameterAgent**: Abstract base class that defines the common interface for all agents
2. **SharedStateManager**: Centralized repository for training metrics and hyperparameter values
3. **AgentCoordinator**: Manages interactions between agents and resolves conflicts
4. **Specialized Agents**: Individual agents for each hyperparameter (e.g., LearningRateAgent)

### Agent Communication Flow

```
Training Loop → AgentCoordinator → SharedStateManager ↔ Specialized Agents
```

### Conflict Resolution Strategies

The system supports multiple conflict resolution strategies:

- **Priority-based**: Agents with higher priority take precedence
- **Voting-based**: Agents vote on the best action to take
- **Consensus-based**: Agents reach a consensus through negotiation

## Tunable Hyperparameters

The system can optimize a wide range of hyperparameters, including:

### Model Architecture
- Spatial dimensions
- Input/output channels
- Network depth and width
- Kernel sizes
- Strides
- Normalization type

### Optimization
- Learning rate
- Weight decay
- Momentum
- Optimizer type
- Batch size

### Loss Function
- Loss function type
- Class weights
- Component weights (e.g., lambda_ce, lambda_dice)
- Focal gamma

### Data Processing
- Augmentation types and probabilities
- Normalization strategy
- Patch size

### Training Strategy
- Learning rate scheduler
- Early stopping criteria
- Validation frequency

## Usage

To train a model with the multi-agent system:

```bash
python training/train_with_agents.py --data_dir data/synthetic --results_dir results/multi_agent
```

To train with fixed hyperparameters (disable agents):

```bash
python training/train_with_agents.py --data_dir data/synthetic --results_dir results/baseline --disable_agents
```

## Extending the System

To add a new specialized agent:

1. Create a new file in `agents/specialized/` (e.g., `weight_decay_agent.py`)
2. Implement a class that inherits from `BaseHyperparameterAgent`
3. Override the abstract methods: `_initialize_states()`, `_initialize_actions()`, `_analyze_state()`, `_calculate_reward()`, and `_apply_action()`
4. Register the new agent in `initialize_agents()` in `train_with_agents.py`

## Advantages Over Traditional Hyperparameter Optimization

Compared to traditional methods like grid search, random search, or Bayesian optimization:

1. **Dynamic Adaptation**: Adjusts hyperparameters during training, not just at the start
2. **Parameter Interdependency**: Learns relationships between hyperparameters
3. **Computational Efficiency**: Single training run vs. multiple separate runs
4. **Domain Knowledge**: Incorporates medical imaging domain knowledge
5. **Continuous Learning**: Agents improve over time through reinforcement learning

## Requirements

- Python 3.8+
- PyTorch 1.9+
- MONAI 0.8+
- NumPy
- Matplotlib
- Pandas
