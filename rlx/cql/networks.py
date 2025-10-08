import mlx.core as mx
import mlx.nn as nn


class QNetwork(nn.Module):
    """
    Q-Network for CQL algorithm.
    
    Takes state and action as input and outputs Q-value.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input layer (state + action)
        input_dim = state_dim + action_dim
        
        # Hidden layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer (single Q-value)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def __call__(self, states, actions):
        """
        Forward pass of Q-network.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Q-values for state-action pairs
        """
        # Concatenate states and actions
        inputs = mx.concatenate([states, actions], axis=-1)
        
        # Forward pass
        q_values = self.network(inputs)
        
        return q_values.squeeze(-1)  # Remove last dimension


class PolicyNetwork(nn.Module):
    """
    Policy Network for CQL algorithm.
    
    Takes state as input and outputs mean and log_std of action distribution.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared layers
        layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize log_std to small values
        self.log_std_layer.bias = mx.full((action_dim,), -0.5)
    
    def __call__(self, states):
        """
        Forward pass of policy network.
        
        Args:
            states: Batch of states
            
        Returns:
            mean: Mean of action distribution
            log_std: Log standard deviation of action distribution
        """
        # Shared feature extraction
        features = self.shared_layers(states)
        
        # Output mean and log_std
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = mx.clip(log_std, -20, 2)
        
        return mean, log_std


class CQLNetworks:
    """
    Container class for all networks used in CQL.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.q_target_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim, num_layers)
        
        # Initialize target network with same weights as main network
        self._copy_weights(self.q_network, self.q_target_network)
        
        # Initialize networks
        mx.eval(self.q_network.parameters())
        mx.eval(self.q_target_network.parameters())
        mx.eval(self.policy_network.parameters())
    
    def _copy_weights(self, source, target):
        """Copy weights from source network to target network."""
        source_params = source.parameters()
        target.update(source_params)
    
    def get_parameters(self):
        """Get all network parameters."""
        return {
            'q_network': self.q_network.parameters(),
            'q_target_network': self.q_target_network.parameters(),
            'policy_network': self.policy_network.parameters(),
        }
