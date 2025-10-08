#!/usr/bin/env python3
"""
Example usage of CQL implementation.

This script demonstrates how to use the CQL algorithm for offline reinforcement learning.
"""

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from cql import CQL
from networks import CQLNetworks


def create_example_dataset():
    """
    Create a simple example dataset for demonstration.
    
    In practice, you would load your real offline dataset here.
    """
    print("Creating example dataset...")
    
    # Dataset parameters
    num_transitions = 10000
    state_dim = 4
    action_dim = 2
    
    # Generate synthetic data
    np.random.seed(42)
    observations = np.random.randn(num_transitions, state_dim)
    actions = np.random.uniform(-1, 1, (num_transitions, action_dim))
    rewards = np.random.randn(num_transitions)
    next_observations = observations + np.random.randn(num_transitions, state_dim) * 0.1
    dones = np.random.random(num_transitions) < 0.05  # 5% chance of done
    
    # Save dataset
    dataset_path = "example_dataset.npz"
    np.savez(
        dataset_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones
    )
    
    print(f"Example dataset saved to {dataset_path}")
    return dataset_path, state_dim, action_dim


def train_cql_example():
    """
    Example of training CQL on a dataset.
    """
    print("CQL Training Example")
    print("=" * 50)
    
    # Create example dataset
    dataset_path, state_dim, action_dim = create_example_dataset()
    
    # Load dataset
    data = np.load(dataset_path)
    print(f"Loaded dataset with {len(data['observations'])} transitions")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create networks
    print("\nCreating CQL networks...")
    networks = CQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        num_layers=2
    )
    
    # Create optimizers
    q_optimizer = optim.Adam(learning_rate=3e-4)
    policy_optimizer = optim.Adam(learning_rate=3e-4)
    
    # Create CQL agent
    print("Initializing CQL agent...")
    agent = CQL(
        q_network=networks.q_network,
        q_target_network=networks.q_target_network,
        policy_network=networks.policy_network,
        q_optimizer=q_optimizer,
        policy_optimizer=policy_optimizer,
        cql_alpha=1.0,  # Conservative penalty weight
        cql_temp=1.0,   # Temperature for CQL loss
        cql_min_q_weight=1.0,
        cql_lagrange=False,  # Don't use lagrange multiplier
        cql_target_action_gap=1.0,
    )
    
    # Training parameters
    batch_size = 256
    num_epochs = 100
    gamma = 0.99
    tau = 0.005
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Gamma: {gamma}, Tau: {tau}")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_q_loss = 0
        epoch_policy_loss = 0
        num_batches = 0
        
        # Sample batches from dataset
        for _ in range(10):  # 10 batches per epoch
            # Sample random batch
            batch_indices = np.random.randint(0, len(data['observations']), size=batch_size)
            
            batch = {
                'observations': mx.array(data['observations'][batch_indices]),
                'actions': mx.array(data['actions'][batch_indices]),
                'rewards': mx.array(data['rewards'][batch_indices]),
                'next_observations': mx.array(data['next_observations'][batch_indices]),
                'dones': mx.array(data['dones'][batch_indices]),
            }
            
            # Update Q-network
            q_loss = agent.update_q_network(
                observations=batch['observations'],
                actions=batch['actions'],
                rewards=batch['rewards'],
                next_observations=batch['next_observations'],
                dones=batch['dones'],
                gamma=gamma
            )
            
            # Update policy (every other step)
            if num_batches % 2 == 0:
                policy_loss = agent.update_policy(batch['observations'])
                epoch_policy_loss += policy_loss
            
            # Update target network
            if num_batches % 10 == 0:
                agent.soft_update_target_network(tau)
            
            epoch_q_loss += q_loss
            num_batches += 1
        
        # Log progress
        if epoch % 10 == 0:
            avg_q_loss = epoch_q_loss / num_batches
            avg_policy_loss = epoch_policy_loss / (num_batches // 2) if num_batches > 0 else 0
            print(f"Epoch {epoch:3d}: Q Loss = {avg_q_loss:.4f}, Policy Loss = {avg_policy_loss:.4f}")
    
    print("\nTraining completed!")
    
    # Test the trained policy
    print("\nTesting trained policy...")
    test_states = mx.array(data['observations'][:5])  # Test on first 5 states
    
    # Get actions from trained policy
    actions, log_probs = agent.get_action(test_states, deterministic=True)
    print(f"Test states shape: {test_states.shape}")
    print(f"Policy actions shape: {actions.shape}")
    print(f"Sample actions: {actions[0]}")
    
    # Get Q-values for these state-action pairs
    q_values = networks.q_network(test_states, actions)
    print(f"Q-values: {q_values}")
    
    print("\n✓ CQL training example completed successfully!")


def demonstrate_cql_conservatism():
    """
    Demonstrate the conservative nature of CQL by comparing Q-values
    for dataset actions vs random actions.
    """
    print("\nCQL Conservatism Demonstration")
    print("=" * 50)
    
    # Create a simple example
    state_dim = 4
    action_dim = 2
    batch_size = 32
    
    # Create networks
    networks = CQLNetworks(state_dim, action_dim, hidden_dim=64, num_layers=2)
    
    # Create optimizers and agent
    q_optimizer = optim.Adam(learning_rate=3e-4)
    policy_optimizer = optim.Adam(learning_rate=3e-4)
    
    agent = CQL(
        q_network=networks.q_network,
        q_target_network=networks.q_target_network,
        policy_network=networks.policy_network,
        q_optimizer=q_optimizer,
        policy_optimizer=policy_optimizer,
        cql_alpha=1.0,
    )
    
    # Create some test data
    states = mx.random.normal((batch_size, state_dim))
    dataset_actions = mx.random.uniform(-1, 1, (batch_size, action_dim))
    random_actions = mx.random.uniform(-1, 1, (batch_size, action_dim))
    
    # Train for a few steps to see the effect
    print("Training CQL for a few steps...")
    for step in range(50):
        # Create dummy batch
        batch = {
            'observations': states,
            'actions': dataset_actions,
            'rewards': mx.random.normal((batch_size,)),
            'next_observations': states + mx.random.normal((batch_size, state_dim)) * 0.1,
            'dones': mx.random.uniform(0, 1, (batch_size,)) < 0.1,
        }
        
        # Update Q-network
        agent.update_q_network(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones'], gamma=0.99
        )
        
        # Update policy
        if step % 2 == 0:
            agent.update_policy(batch['observations'])
        
        # Update target network
        if step % 10 == 0:
            agent.soft_update_target_network(0.005)
    
    # Compare Q-values
    dataset_q_values = networks.q_network(states, dataset_actions)
    random_q_values = networks.q_network(states, random_actions)
    
    print(f"\nQ-values for dataset actions: {mx.mean(dataset_q_values):.4f} ± {mx.std(dataset_q_values):.4f}")
    print(f"Q-values for random actions: {mx.mean(random_q_values):.4f} ± {mx.std(random_q_values):.4f}")
    print(f"Difference (dataset - random): {mx.mean(dataset_q_values - random_q_values):.4f}")
    
    if mx.mean(dataset_q_values) > mx.mean(random_q_values):
        print("✓ CQL is working correctly - dataset actions have higher Q-values!")
    else:
        print("⚠ CQL may need more training or tuning")


if __name__ == "__main__":
    # Set random seed for reproducibility
    mx.random.seed(42)
    np.random.seed(42)
    
    # Run examples
    train_cql_example()
    demonstrate_cql_conservatism()
    
    print("\n" + "=" * 60)
    print("🎉 All CQL examples completed successfully!")
    print("You can now use this CQL implementation for your offline RL projects.")
    print("=" * 60)
