# CQL Hyperparameters
# Conservative Q-Learning algorithm configuration

# General Parameters
exp_name = "cql_offline_rl"
seed = 1
env_id = "HalfCheetah-v4"  # Example environment for testing

# Algorithm specific arguments
total_timesteps = 1_000_000
learning_rate = 3e-4
batch_size = 256
buffer_size = 1_000_000
gamma = 0.99
tau = 0.005  # Soft update parameter for target networks

# CQL specific parameters
cql_alpha = 1.0  # Conservative penalty weight
cql_lagrange = False  # Whether to use lagrange multiplier for CQL loss
cql_target_action_gap = 1.0  # Target gap between in-distribution and out-of-distribution actions
cql_temp = 1.0  # Temperature for CQL loss
cql_min_q_weight = 1.0  # Weight for minimum Q-value in CQL loss

# Network architecture
num_layers = 2
hidden_dim = 256
activations = ["relu", "relu"]

# Training parameters
learning_starts = 5000  # Number of steps before learning begins
target_network_frequency = 1  # How often to update target networks
policy_frequency = 2  # How often to update policy (delayed updates)

# Evaluation
eval_frequency = 10000  # How often to evaluate the policy
num_eval_episodes = 10  # Number of episodes for evaluation

# Logging
log_frequency = 1000  # How often to log training metrics
save_frequency = 50000  # How often to save model checkpoints
