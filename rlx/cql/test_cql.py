#!/usr/bin/env python3
"""
Test script for CQL implementation.
This script tests the basic functionality of the CQL algorithm.
"""

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from cql import CQL
from networks import CQLNetworks


def test_cql_networks():
    """Test that networks can be created and forward pass works."""
    print("Testing CQL networks...")
    
    state_dim = 4
    action_dim = 2
    batch_size = 32
    
    # Create networks
    networks = CQLNetworks(state_dim, action_dim, hidden_dim=64, num_layers=2)
    
    # Create dummy data
    states = mx.random.normal((batch_size, state_dim))
    actions = mx.random.uniform(-1, 1, (batch_size, action_dim))
    
    # Test Q-network
    q_values = networks.q_network(states, actions)
    assert q_values.shape == (batch_size,), f"Expected Q-values shape ({batch_size},), got {q_values.shape}"
    print(f"✓ Q-network forward pass: {q_values.shape}")
    
    # Test policy network
    mean, log_std = networks.policy_network(states)
    assert mean.shape == (batch_size, action_dim), f"Expected mean shape ({batch_size}, {action_dim}), got {mean.shape}"
    assert log_std.shape == (batch_size, action_dim), f"Expected log_std shape ({batch_size}, {action_dim}), got {log_std.shape}"
    print(f"✓ Policy network forward pass: mean {mean.shape}, log_std {log_std.shape}")
    
    print("✓ All network tests passed!")


def test_cql_agent():
    """Test that CQL agent can be created and basic operations work."""
    print("Testing CQL agent...")
    
    state_dim = 4
    action_dim = 2
    batch_size = 32
    
    # Create networks
    networks = CQLNetworks(state_dim, action_dim, hidden_dim=64, num_layers=2)
    
    # Create optimizers
    q_optimizer = optim.Adam(learning_rate=1e-3)
    policy_optimizer = optim.Adam(learning_rate=1e-3)
    
    # Create CQL agent
    agent = CQL(
        q_network=networks.q_network,
        q_target_network=networks.q_target_network,
        policy_network=networks.policy_network,
        q_optimizer=q_optimizer,
        policy_optimizer=policy_optimizer,
        cql_alpha=1.0,
        cql_temp=1.0,
        cql_min_q_weight=1.0,
        cql_lagrange=False,
        cql_target_action_gap=1.0,
    )
    
    # Create dummy data
    states = mx.random.normal((batch_size, state_dim))
    actions = mx.random.uniform(-1, 1, (batch_size, action_dim))
    rewards = mx.random.normal((batch_size,))
    next_states = mx.random.normal((batch_size, state_dim))
    dones = mx.random.uniform(0, 1, (batch_size,)) < 0.1  # 10% chance of done
    
    # Test action sampling
    sampled_actions, log_probs = agent.get_action(states)
    assert sampled_actions.shape == (batch_size, action_dim), f"Expected actions shape ({batch_size}, {action_dim}), got {sampled_actions.shape}"
    assert log_probs.shape == (batch_size, 1), f"Expected log_probs shape ({batch_size}, 1), got {log_probs.shape}"
    print(f"✓ Action sampling: actions {sampled_actions.shape}, log_probs {log_probs.shape}")
    
    # Test deterministic action
    det_actions, det_log_probs = agent.get_action(states, deterministic=True)
    assert det_actions.shape == (batch_size, action_dim), f"Expected det_actions shape ({batch_size}, {action_dim}), got {det_actions.shape}"
    print(f"✓ Deterministic action: {det_actions.shape}")
    
    # Test Q-network update
    q_loss = agent.update_q_network(states, actions, rewards, next_states, dones, gamma=0.99)
    assert isinstance(q_loss, mx.array), f"Expected q_loss to be mx.array, got {type(q_loss)}"
    print(f"✓ Q-network update: loss = {q_loss:.4f}")
    
    # Test policy update
    policy_loss = agent.update_policy(states)
    assert isinstance(policy_loss, mx.array), f"Expected policy_loss to be mx.array, got {type(policy_loss)}"
    print(f"✓ Policy update: loss = {policy_loss:.4f}")
    
    # Test target network update
    agent.soft_update_target_network(tau=0.005)
    print("✓ Target network update")
    
    print("✓ All CQL agent tests passed!")


def test_cql_loss():
    """Test CQL loss computation specifically."""
    print("Testing CQL loss computation...")
    
    state_dim = 4
    action_dim = 2
    batch_size = 32
    
    # Create networks
    networks = CQLNetworks(state_dim, action_dim, hidden_dim=64, num_layers=2)
    
    # Create optimizers
    q_optimizer = optim.Adam(learning_rate=1e-3)
    policy_optimizer = optim.Adam(learning_rate=1e-3)
    
    # Create CQL agent
    agent = CQL(
        q_network=networks.q_network,
        q_target_network=networks.q_target_network,
        policy_network=networks.policy_network,
        q_optimizer=q_optimizer,
        policy_optimizer=policy_optimizer,
        cql_alpha=1.0,
        cql_temp=1.0,
        cql_min_q_weight=1.0,
        cql_lagrange=False,
        cql_target_action_gap=1.0,
    )
    
    # Create dummy data
    states = mx.random.normal((batch_size, state_dim))
    actions = mx.random.uniform(-1, 1, (batch_size, action_dim))
    
    # Test CQL loss computation
    cql_loss = agent._compute_cql_loss(states, actions)
    assert isinstance(cql_loss, mx.array), f"Expected cql_loss to be mx.array, got {type(cql_loss)}"
    assert cql_loss.shape == (), f"Expected cql_loss to be scalar, got {cql_loss.shape}"
    print(f"✓ CQL loss computation: {cql_loss:.4f}")
    
    print("✓ All CQL loss tests passed!")


def main():
    """Run all tests."""
    print("Running CQL implementation tests...\n")
    
    try:
        test_cql_networks()
        print()
        test_cql_agent()
        print()
        test_cql_loss()
        print()
        print("🎉 All tests passed! CQL implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
