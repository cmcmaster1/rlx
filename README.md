# RLX: Reinforcement Learning with MLX

RLX is a collection of Reinforcement Learning algorithms implemented based on the implementations from CleanRL in MLX, Apple's new Machine Learning framework. This project aims to leverage the unified memory capabilities of Apple's M series chips to enhance the performance and efficiency of these algorithms.

## Prerequisites

- Python 3.11 or later
- uv for dependency management
- An Apple device with an M-series chip

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/noahfarr/rlx.git
cd rlx
```

Install dependencies using uv:
```bash
uv sync
```

## Structure

The project is organized into directories by algorithm. Each directory contains the implementation of a specific Reinforcement Learning algorithm, making the project modular and scalable. Here's an overview:

- **a2c/**: Advantage Actor-Critic algorithm
- **cql/**: Conservative Q-Learning for offline reinforcement learning
- **dqn/**: Deep Q-Network algorithm
- **ppo/**: Proximal Policy Optimization algorithm
- **reinforce/**: REINFORCE policy gradient algorithm
- **sac/**: Soft Actor-Critic algorithm
- **td3/**: Twin Delayed Deep Deterministic Policy Gradient algorithm

## Usage

To run a specific algorithm, navigate to its directory and execute the main script. For example:

```bash
# Run DQN
cd dqn
uv run python main.py

# Run CQL (Conservative Q-Learning) for offline RL
cd cql
uv run python main.py --dataset_path /path/to/your/dataset.npz

# Run PPO
cd ppo
uv run python main.py
```

### Offline Reinforcement Learning with CQL

The CQL implementation is specifically designed for offline reinforcement learning, where you train on pre-collected datasets rather than interacting with environments in real-time. This is perfect for scenarios where:

- You have historical data but can't collect new data
- Environment interaction is expensive or dangerous
- You want to leverage existing datasets for policy learning

See the [CQL documentation](rlx/cql/README.md) for detailed usage instructions.

## Contributing

Contributions to RLX are welcome. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (git checkout -b feature/AmazingFeature).
3. Commit your changes (git commit -m 'Add some AmazingFeature').
4. Push to the branch (git push origin feature/AmazingFeature).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to the MLX team for providing the framework.
This project is designed to run optimally on Apple's M series chips.
