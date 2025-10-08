import argparse
import random
import time
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from typing import Dict, Any, Tuple

import rlx.cql.hyperparameters as h
from rlx.cql.cql import CQL
from rlx.cql.networks import CQLNetworks


def parse_args():
    """
    Parse command line arguments for CQL training.
    """
    parser = argparse.ArgumentParser(description="Conservative Q-Learning (CQL) with MLX")
    
    # General Parameters
    parser.add_argument("--exp_name", type=str, default=h.exp_name)
    parser.add_argument("--seed", type=int, default=h.seed)
    parser.add_argument("--env_id", type=str, default=h.env_id)
    
    # Algorithm specific arguments
    parser.add_argument("--total_timesteps", type=int, default=h.total_timesteps)
    parser.add_argument("--learning_rate", type=float, default=h.learning_rate)
    parser.add_argument("--batch_size", type=int, default=h.batch_size)
    parser.add_argument("--buffer_size", type=int, default=h.buffer_size)
    parser.add_argument("--gamma", type=float, default=h.gamma)
    parser.add_argument("--tau", type=float, default=h.tau)
    
    # CQL specific parameters
    parser.add_argument("--cql_alpha", type=float, default=h.cql_alpha)
    parser.add_argument("--cql_lagrange", type=bool, default=h.cql_lagrange)
    parser.add_argument("--cql_target_action_gap", type=float, default=h.cql_target_action_gap)
    parser.add_argument("--cql_temp", type=float, default=h.cql_temp)
    parser.add_argument("--cql_min_q_weight", type=float, default=h.cql_min_q_weight)
    
    # Network architecture
    parser.add_argument("--num_layers", type=int, default=h.num_layers)
    parser.add_argument("--hidden_dim", type=int, default=h.hidden_dim)
    
    # Training parameters
    parser.add_argument("--learning_starts", type=int, default=h.learning_starts)
    parser.add_argument("--target_network_frequency", type=int, default=h.target_network_frequency)
    parser.add_argument("--policy_frequency", type=int, default=h.policy_frequency)
    
    # Evaluation
    parser.add_argument("--eval_frequency", type=int, default=h.eval_frequency)
    parser.add_argument("--num_eval_episodes", type=int, default=h.num_eval_episodes)
    
    # Logging
    parser.add_argument("--log_frequency", type=int, default=h.log_frequency)
    parser.add_argument("--save_frequency", type=int, default=h.save_frequency)
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to offline dataset (numpy .npz file)")
    
    return parser.parse_args()


class OfflineDataset:
    """
    Simple offline dataset loader for CQL training.
    
    Expected dataset format: .npz file with keys:
    - observations: (N, state_dim)
    - actions: (N, action_dim) 
    - rewards: (N,)
    - next_observations: (N, state_dim)
    - dones: (N,)
    """
    
    def __init__(self, dataset_path: str):
        self.data = np.load(dataset_path)
        
        self.observations = self.data['observations']
        self.actions = self.data['actions']
        self.rewards = self.data['rewards']
        self.next_observations = self.data['next_observations']
        self.dones = self.data['dones']
        
        self.size = len(self.observations)
        print(f"Loaded dataset with {self.size} transitions")
    
    def sample(self, batch_size: int) -> Dict[str, mx.array]:
        """Sample a batch of transitions from the dataset."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': mx.array(self.observations[indices]),
            'actions': mx.array(self.actions[indices]),
            'rewards': mx.array(self.rewards[indices]),
            'next_observations': mx.array(self.next_observations[indices]),
            'dones': mx.array(self.dones[indices]),
        }


def create_synthetic_dataset(state_dim: int, action_dim: int, size: int = 100000) -> str:
    """
    Create a synthetic dataset for testing CQL.
    
    This is a placeholder - in practice, you would load your real offline dataset.
    """
    print(f"Creating synthetic dataset with {size} transitions...")
    
    # Generate random transitions
    observations = np.random.randn(size, state_dim)
    actions = np.random.uniform(-1, 1, size=(size, action_dim))
    rewards = np.random.randn(size)
    next_observations = observations + np.random.randn(size, state_dim) * 0.1
    dones = np.random.random(size) < 0.05  # 5% chance of done
    
    # Save dataset
    dataset_path = "synthetic_dataset.npz"
    np.savez(
        dataset_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones
    )
    
    print(f"Synthetic dataset saved to {dataset_path}")
    return dataset_path


def evaluate_policy(agent: CQL, dataset: OfflineDataset, num_episodes: int = 10) -> float:
    """
    Evaluate the policy on the dataset.
    
    This is a simplified evaluation - in practice, you might want to
    evaluate on a separate test set or in a real environment.
    """
    total_rewards = []
    
    for _ in range(num_episodes):
        # Sample random starting state
        idx = np.random.randint(0, dataset.size)
        obs = mx.array(dataset.observations[idx:idx+1])
        
        episode_reward = 0
        done = False
        step = 0
        max_steps = 1000
        
        while not done and step < max_steps:
            action, _ = agent.get_action(obs, deterministic=True)
            
            # Find next state in dataset (simplified)
            # In practice, you'd use the environment or a learned model
            next_idx = (idx + 1) % dataset.size
            next_obs = mx.array(dataset.next_observations[next_idx:next_idx+1])
            reward = dataset.rewards[next_idx]
            done = bool(dataset.dones[next_idx])
            
            episode_reward += reward
            obs = next_obs
            idx = next_idx
            step += 1
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    print(f"Starting CQL training with MLX")
    print(f"Experiment: {args.exp_name}")
    print(f"Dataset: {args.dataset_path}")
    
    # Load dataset
    try:
        dataset = OfflineDataset(args.dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {args.dataset_path}")
        print("Creating synthetic dataset for testing...")
        # Create synthetic dataset for testing
        state_dim = 17  # Typical for HalfCheetah
        action_dim = 6  # Typical for HalfCheetah
        dataset_path = create_synthetic_dataset(state_dim, action_dim)
        dataset = OfflineDataset(dataset_path)
    
    # Get dimensions from dataset
    state_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create networks
    networks = CQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # Create optimizers
    q_optimizer = optim.Adam(learning_rate=args.learning_rate)
    policy_optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Create CQL agent
    agent = CQL(
        q_network=networks.q_network,
        q_target_network=networks.q_target_network,
        policy_network=networks.policy_network,
        q_optimizer=q_optimizer,
        policy_optimizer=policy_optimizer,
        cql_alpha=args.cql_alpha,
        cql_temp=args.cql_temp,
        cql_min_q_weight=args.cql_min_q_weight,
        cql_lagrange=args.cql_lagrange,
        cql_target_action_gap=args.cql_target_action_gap,
    )
    
    # Training loop
    start_time = time.time()
    global_step = 0
    
    print("Starting training...")
    
    while global_step < args.total_timesteps:
        # Sample batch from dataset
        batch = dataset.sample(args.batch_size)
        
        # Update Q-network
        q_loss = agent.update_q_network(
            observations=batch['observations'],
            actions=batch['actions'],
            rewards=batch['rewards'],
            next_observations=batch['next_observations'],
            dones=batch['dones'],
            gamma=args.gamma
        )
        
        # Update policy (delayed updates)
        if global_step % args.policy_frequency == 0:
            policy_loss = agent.update_policy(batch['observations'])
        
        # Update target network
        if global_step % args.target_network_frequency == 0:
            agent.soft_update_target_network(args.tau)
        
        # Update lagrange multiplier (if using lagrange version)
        if args.cql_lagrange and global_step % args.policy_frequency == 0:
            alpha_prime_loss = agent.update_alpha_prime(
                batch['observations'], batch['actions']
            )
        
        # Logging
        if global_step % args.log_frequency == 0:
            elapsed_time = time.time() - start_time
            print(f"Step: {global_step}, "
                  f"Q Loss: {q_loss:.4f}, "
                  f"Elapsed: {elapsed_time:.2f}s")
        
        # Evaluation
        if global_step % args.eval_frequency == 0 and global_step > 0:
            eval_reward = evaluate_policy(agent, dataset, args.num_eval_episodes)
            print(f"Evaluation at step {global_step}: Average reward = {eval_reward:.4f}")
        
        # Save model
        if global_step % args.save_frequency == 0 and global_step > 0:
            # Save model parameters
            model_path = f"cql_model_step_{global_step}.npz"
            mx.savez(
                model_path,
                q_network=networks.q_network.parameters(),
                policy_network=networks.policy_network.parameters(),
            )
            print(f"Model saved to {model_path}")
        
        global_step += 1
    
    print("Training completed!")
    
    # Final evaluation
    final_eval_reward = evaluate_policy(agent, dataset, args.num_eval_episodes)
    print(f"Final evaluation: Average reward = {final_eval_reward:.4f}")
    
    # Save final model
    final_model_path = "cql_final_model.npz"
    mx.savez(
        final_model_path,
        q_network=networks.q_network.parameters(),
        policy_network=networks.policy_network.parameters(),
    )
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
