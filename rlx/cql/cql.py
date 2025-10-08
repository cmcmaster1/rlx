import mlx.core as mx
import mlx.nn as nn
import numpy as np


class CQL:
    """
    Conservative Q-Learning (CQL) implementation for offline reinforcement learning.
    
    CQL addresses the overestimation bias in offline RL by adding a conservative penalty
    to the Q-function that penalizes actions not well-represented in the dataset.
    """
    
    def __init__(
        self,
        q_network,
        q_target_network,
        policy_network,
        q_optimizer,
        policy_optimizer,
        cql_alpha=1.0,
        cql_temp=1.0,
        cql_min_q_weight=1.0,
        cql_lagrange=False,
        cql_target_action_gap=1.0,
    ):
        self.q_network = q_network
        self.q_target_network = q_target_network
        self.policy_network = policy_network
        
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer
        
        # CQL hyperparameters
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        
        # Initialize lagrange multiplier if using lagrange version
        if self.cql_lagrange:
            self.log_alpha_prime = mx.zeros(1)
            self.alpha_prime_optimizer = mx.optimizers.Adam(learning_rate=3e-4)
        
        # Loss and gradient functions
        self.q_loss_and_grad_fn = nn.value_and_grad(q_network, self.q_loss_fn)
        self.policy_loss_and_grad_fn = nn.value_and_grad(policy_network, self.policy_loss_fn)
        
        if self.cql_lagrange:
            self.alpha_prime_loss_and_grad_fn = mx.value_and_grad(self.alpha_prime_loss_fn)
    
    def get_action(self, observations, deterministic=False):
        """
        Get action from policy network.
        
        Args:
            observations: Current observations
            deterministic: If True, return deterministic action (mean)
            
        Returns:
            actions: Actions sampled from policy
            log_probs: Log probabilities of actions
        """
        mean, log_std = self.policy_network(observations)
        std = mx.exp(log_std)
        
        if deterministic:
            actions = mean
            log_probs = self._get_log_prob(mean, mean, std)
        else:
            # Sample from normal distribution
            normal = mx.random.normal(mean.shape)
            actions = mean + std * normal
            log_probs = self._get_log_prob(actions, mean, std)
        
        # Apply tanh to bound actions
        actions = mx.tanh(actions)
        
        # Adjust log probabilities for tanh transformation
        log_probs -= mx.log(1 - actions**2 + 1e-6)
        log_probs = mx.sum(log_probs, axis=-1, keepdims=True)
        
        return actions, log_probs
    
    def _get_log_prob(self, actions, mean, std):
        """Calculate log probability of actions under normal distribution."""
        variance = std**2
        log_variance = mx.log(variance)
        return -0.5 * (
            log_variance + mx.log(2 * mx.pi) + (actions - mean)**2 / variance
        )
    
    def q_loss_fn(self, observations, actions, rewards, next_observations, dones, gamma):
        """
        CQL Q-function loss with conservative penalty.
        
        The loss consists of:
        1. Standard TD loss
        2. Conservative penalty (CQL loss)
        """
        # Standard TD loss
        # Get target Q-values (no gradient computation for target network)
        next_actions, next_log_probs = self.get_action(next_observations)
        target_q_values = self.q_target_network(next_observations, next_actions)
        target_q_values = target_q_values - next_log_probs.squeeze()  # SAC-style entropy bonus
        
        td_targets = rewards + gamma * (1 - dones.astype(mx.float32)) * target_q_values
        current_q_values = self.q_network(observations, actions)
        td_loss = nn.losses.mse_loss(current_q_values, td_targets)
        
        # CQL conservative penalty
        cql_loss = self._compute_cql_loss(observations, actions)
        
        # Combine losses
        total_loss = td_loss + self.cql_alpha * cql_loss
        
        return total_loss
    
    def _compute_cql_loss(self, observations, actions):
        """
        Compute the conservative Q-learning loss.
        
        This penalizes the Q-function for assigning high values to actions
        not well-represented in the dataset.
        """
        batch_size = observations.shape[0]
        
        # Q-values for dataset actions
        dataset_q_values = self.q_network(observations, actions)
        
        # Sample random actions for comparison
        random_actions = mx.random.uniform(
            low=-1.0, high=1.0, shape=(batch_size, actions.shape[-1])
        )
        random_q_values = self.q_network(observations, random_actions)
        
        # Q-values for policy actions
        policy_actions, _ = self.get_action(observations)
        policy_q_values = self.q_network(observations, policy_actions)
        
        # CQL loss: penalize high Q-values for non-dataset actions
        # We want: Q(s,a_dataset) > Q(s,a_random) and Q(s,a_dataset) > Q(s,a_policy)
        
        # Conservative penalty for random actions
        random_penalty = mx.mean(mx.exp(random_q_values / self.cql_temp))
        
        # Conservative penalty for policy actions
        policy_penalty = mx.mean(mx.exp(policy_q_values / self.cql_temp))
        
        # Dataset Q-values (we want these to be higher)
        dataset_penalty = mx.mean(mx.exp(dataset_q_values / self.cql_temp))
        
        # CQL loss: minimize Q-values for non-dataset actions, maximize for dataset actions
        cql_loss = mx.logsumexp(
            mx.concatenate([
                random_q_values / self.cql_temp,
                policy_q_values / self.cql_temp
            ], axis=0)
        ) - mx.mean(dataset_q_values / self.cql_temp)
        
        return cql_loss
    
    def policy_loss_fn(self, observations):
        """
        Policy loss for CQL.
        
        In CQL, the policy is trained to maximize Q-values while being regularized
        by the conservative Q-function.
        """
        actions, log_probs = self.get_action(observations)
        q_values = self.q_network(observations, actions)
        
        # Policy loss: maximize Q-values minus entropy regularization
        policy_loss = -mx.mean(q_values - log_probs)
        
        return policy_loss
    
    def alpha_prime_loss_fn(self, observations, actions):
        """
        Loss for the lagrange multiplier (if using lagrange version).
        """
        if not self.cql_lagrange:
            return mx.array(0.0)
        
        # Compute CQL loss
        cql_loss = self._compute_cql_loss(observations, actions)
        
        # Lagrange multiplier loss
        alpha_prime = mx.exp(self.log_alpha_prime)
        alpha_prime_loss = alpha_prime * (cql_loss - self.cql_target_action_gap)
        
        return alpha_prime_loss
    
    def update_q_network(self, observations, actions, rewards, next_observations, dones, gamma):
        """Update Q-network with CQL loss."""
        loss, grads = self.q_loss_and_grad_fn(
            observations, actions, rewards, next_observations, dones, gamma
        )
        
        self.q_optimizer.update(self.q_network, grads)
        mx.eval(self.q_network.parameters(), self.q_optimizer.state)
        
        return loss
    
    def update_policy(self, observations):
        """Update policy network."""
        loss, grads = self.policy_loss_and_grad_fn(observations)
        
        self.policy_optimizer.update(self.policy_network, grads)
        mx.eval(self.policy_network.parameters(), self.policy_optimizer.state)
        
        return loss
    
    def update_alpha_prime(self, observations, actions):
        """Update lagrange multiplier (if using lagrange version)."""
        if not self.cql_lagrange:
            return mx.array(0.0)
        
        loss, grads = self.alpha_prime_loss_and_grad_fn(observations, actions)
        
        self.alpha_prime_optimizer.update(self.log_alpha_prime, grads)
        mx.eval(self.log_alpha_prime, self.alpha_prime_optimizer.state)
        
        return loss
    
    def soft_update_target_network(self, tau):
        """Soft update target Q-network."""
        # For now, use a simple approach - copy all parameters
        # In a more sophisticated implementation, we would do element-wise updates
        source_params = self.q_network.parameters()
        self.q_target_network.update(source_params)
