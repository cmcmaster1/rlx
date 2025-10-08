# Conservative Q-Learning (CQL) with MLX

This directory contains a complete implementation of Conservative Q-Learning (CQL) for offline reinforcement learning using Apple's MLX framework.

## Overview

Conservative Q-Learning (CQL) is an offline reinforcement learning algorithm that addresses the overestimation bias problem in Q-learning when learning from fixed datasets. CQL adds a conservative penalty to the Q-function that penalizes actions not well-represented in the training dataset.

## Key Features

- **Pure MLX Implementation**: Optimized for Apple Silicon MacBooks
- **Offline Learning**: Designed to work with pre-collected datasets
- **Conservative Penalty**: Prevents overestimation of Q-values for out-of-distribution actions
- **Flexible Architecture**: Configurable network architectures and hyperparameters
- **Complete Implementation**: Includes Q-networks, policy networks, and training loops

## Files

- `cql.py`: Main CQL algorithm implementation
- `networks.py`: Q-network and policy network architectures
- `main.py`: Training script for offline learning
- `hyperparameters.py`: Default hyperparameters and configuration
- `test_cql.py`: Unit tests for the implementation
- `example_usage.py`: Example usage and demonstrations
- `README.md`: This documentation

## Quick Start

### 1. Basic Usage

```python
import mlx.core as mx
import mlx.optimizers as optim
from cql import CQL
from networks import CQLNetworks

# Create networks
networks = CQLNetworks(state_dim=4, action_dim=2, hidden_dim=64)

# Create optimizers
q_optimizer = optim.Adam(learning_rate=3e-4)
policy_optimizer = optim.Adam(learning_rate=3e-4)

# Create CQL agent
agent = CQL(
    q_network=networks.q_network,
    q_target_network=networks.q_target_network,
    policy_network=networks.policy_network,
    q_optimizer=q_optimizer,
    policy_optimizer=policy_optimizer,
    cql_alpha=1.0,  # Conservative penalty weight
)
```

### 2. Training on a Dataset

```python
# Load your offline dataset
dataset = OfflineDataset("your_dataset.npz")

# Training loop
for step in range(num_steps):
    batch = dataset.sample(batch_size=256)
    
    # Update Q-network with CQL loss
    q_loss = agent.update_q_network(
        observations=batch['observations'],
        actions=batch['actions'],
        rewards=batch['rewards'],
        next_observations=batch['next_observations'],
        dones=batch['dones'],
        gamma=0.99
    )
    
    # Update policy
    if step % 2 == 0:
        policy_loss = agent.update_policy(batch['observations'])
    
    # Update target network
    if step % 10 == 0:
        agent.soft_update_target_network(tau=0.005)
```

### 3. Using the Trained Policy

```python
# Get action from trained policy
action, log_prob = agent.get_action(observation, deterministic=True)

# Get Q-value for state-action pair
q_value = networks.q_network(observation, action)
```

## Dataset Format

The implementation expects datasets in NumPy `.npz` format with the following keys:

```python
{
    'observations': np.array,      # Shape: (N, state_dim)
    'actions': np.array,          # Shape: (N, action_dim)
    'rewards': np.array,          # Shape: (N,)
    'next_observations': np.array, # Shape: (N, state_dim)
    'dones': np.array,            # Shape: (N,) - boolean or 0/1
}
```

## Hyperparameters

Key hyperparameters for CQL:

- `cql_alpha`: Conservative penalty weight (default: 1.0)
- `cql_temp`: Temperature for CQL loss (default: 1.0)
- `cql_lagrange`: Whether to use lagrange multiplier (default: False)
- `cql_target_action_gap`: Target gap for lagrange version (default: 1.0)
- `learning_rate`: Learning rate for optimizers (default: 3e-4)
- `batch_size`: Batch size for training (default: 256)
- `gamma`: Discount factor (default: 0.99)
- `tau`: Soft update parameter for target networks (default: 0.005)

## Running the Examples

### Test the Implementation

```bash
cd rlx/cql
python test_cql.py
```

### Run Example Training

```bash
cd rlx/cql
python example_usage.py
```

### Train on Your Dataset

```bash
cd rlx/cql
python main.py --dataset_path /path/to/your/dataset.npz
```

## Algorithm Details

### CQL Loss Function

The CQL loss combines:

1. **Standard TD Loss**: `MSE(Q(s,a), r + γ * Q_target(s',a'))`
2. **Conservative Penalty**: Penalizes high Q-values for actions not in the dataset

The conservative penalty encourages the Q-function to assign lower values to out-of-distribution actions, preventing overestimation bias.

### Network Architecture

- **Q-Network**: Takes state and action as input, outputs Q-value
- **Policy Network**: Takes state as input, outputs mean and log_std of action distribution
- **Target Q-Network**: Copy of Q-network for stable target computation

## Performance Notes

- **MLX Optimization**: Leverages Apple Silicon's unified memory architecture
- **Memory Efficient**: No GPU memory transfers needed
- **Fast Training**: Optimized for M-series chips

## Comparison with d3rlpy

| Feature | This Implementation | d3rlpy |
|---------|-------------------|---------|
| Framework | MLX (Apple Silicon) | PyTorch |
| Performance | Optimized for MacBooks | Cross-platform |
| CQL Algorithm | ✅ Implemented | ✅ Implemented |
| Offline Learning | ✅ Designed for | ✅ Designed for |
| Customization | High (MLX-based) | High (PyTorch-based) |

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Ensure your dataset has the correct dimensions
2. **Memory Issues**: Reduce batch size if running out of memory
3. **Training Instability**: Try reducing learning rate or CQL alpha

### Performance Tips

1. **Batch Size**: Use larger batch sizes (256-512) for better performance
2. **Learning Rate**: Start with 3e-4, adjust based on training stability
3. **CQL Alpha**: Higher values make the policy more conservative

## Contributing

This implementation is part of the RLX project. Contributions are welcome! Please ensure:

1. Code follows MLX best practices
2. Tests pass (`python test_cql.py`)
3. Examples work correctly (`python example_usage.py`)

## References

- [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [RLX Project](https://github.com/noahfarr/rlx)
