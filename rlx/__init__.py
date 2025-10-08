"""RLX: A reinforcement learning framework based on MLX."""

__version__ = "0.1.0"

# Import main algorithms for easy access
from .a2c import A2C
from .cql import CQL
from .dqn import DQN
from .ppo import PPO
from .reinforce import REINFORCE
from .sac import SAC
from .td3 import TD3

__all__ = [
    "A2C",
    "CQL", 
    "DQN",
    "PPO",
    "REINFORCE",
    "SAC",
    "TD3",
]
