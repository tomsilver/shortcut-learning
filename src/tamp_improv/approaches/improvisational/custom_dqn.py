"""Custom DQN implementation for goal-conditioned distance learning.

This module provides a transparent, from-scratch implementation of DQN with
Hindsight Experience Replay (HER) for learning goal-conditioned distance functions.
Unlike stable_baselines3, every step is explicit and debuggable.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


@dataclass
class DQNConfig:
    """Configuration for custom DQN."""

    # Network architecture
    hidden_sizes: list[int] = None  # [256, 256] default

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    batch_size: int = 256
    buffer_size: int = 100000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000

    # Target network
    target_update_freq: int = 1000  # Update target network every N steps

    # HER parameters
    her_k: int = 4  # Number of hindsight goals per transition
    her_strategy: str = "final"  # "final" or "future"

    # Training
    learning_starts: int = 1000  # Start learning after N steps
    train_freq: int = 1  # Train every N steps

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]


class QNetwork(nn.Module):
    """Q-network for goal-conditioned DQN.

    Input: Concatenation of [observation, achieved_goal, desired_goal]
    Output: Q-values for each action
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        num_actions: int,
        hidden_sizes: list[int] = None,
    ):
        """Initialize Q-network.

        Args:
            obs_dim: Dimension of observation
            goal_dim: Dimension of goal vector
            num_actions: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.num_actions = num_actions

        # Input is concatenation of obs + achieved_goal + desired_goal
        input_dim = obs_dim + 2 * goal_dim

        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observation tensor [batch, obs_dim]
            achieved_goal: Achieved goal tensor [batch, goal_dim]
            desired_goal: Desired goal tensor [batch, goal_dim]

        Returns:
            Q-values for each action [batch, num_actions]
        """
        # Concatenate inputs
        x = torch.cat([obs, achieved_goal, desired_goal], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Replay buffer with Hindsight Experience Replay (HER) support.

    Stores full episodes and generates virtual transitions using HER.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        goal_dim: int,
        her_k: int = 4,
        her_strategy: str = "future",
        compute_reward_fn: Any = None,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Dimension of observation
            goal_dim: Dimension of goal vector
            her_k: Number of hindsight goals per real transition
            her_strategy: "final" (use final state) or "future" (use future states)
            compute_reward_fn: Function to compute rewards for HER transitions
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.her_k = her_k
        self.her_strategy = her_strategy
        self.compute_reward_fn = compute_reward_fn

        # Storage for complete episodes
        self.episodes: deque = deque(maxlen=capacity // 100)  # Store ~100 step episodes

        # Flattened transition storage for sampling
        self.transitions: deque = deque(maxlen=capacity)

        self.size = 0

    def store_episode(self, episode: list[dict[str, Any]]) -> None:
        """Store a complete episode and generate HER transitions.

        Args:
            episode: List of transitions, each with keys:
                - obs, achieved_goal, desired_goal
                - action, reward, next_obs, next_achieved_goal
                - done
        """
        if len(episode) == 0:
            return

        # Store original episode
        self.episodes.append(episode)

        # Add real transitions
        for transition in episode:
            self.transitions.append(transition)
            self.size = min(self.size + 1, self.capacity)

        # Generate HER transitions
        episode_length = len(episode)
        for t in range(episode_length):
            # Sample k hindsight goals
            hindsight_goals = self._sample_hindsight_goals(episode, t)

            for hindsight_goal in hindsight_goals:
                # Create new transition with hindsight goal
                original = episode[t]

                # Compute new reward with hindsight goal
                if self.compute_reward_fn is not None:
                    new_reward = self.compute_reward_fn(
                        original["next_achieved_goal"],
                        hindsight_goal,
                        {}
                    )[0]
                else:
                    # Default: 0 if goal achieved, -1 otherwise
                    new_reward = 0.0 if np.allclose(original["next_achieved_goal"], hindsight_goal) else -1.0

                # Check if episode terminates with this hindsight goal
                new_done = np.allclose(original["next_achieved_goal"], hindsight_goal)

                # Create hindsight transition
                hindsight_transition = {
                    "obs": original["obs"],
                    "achieved_goal": original["achieved_goal"],
                    "desired_goal": hindsight_goal,
                    "action": original["action"],
                    "reward": new_reward,
                    "next_obs": original["next_obs"],
                    "next_achieved_goal": original["next_achieved_goal"],
                    "done": new_done,
                }

                self.transitions.append(hindsight_transition)
                self.size = min(self.size + 1, self.capacity)

    def _sample_hindsight_goals(self, episode: list[dict], t: int) -> list[NDArray]:
        """Sample hindsight goals for transition at timestep t.

        Args:
            episode: Full episode
            t: Timestep index

        Returns:
            List of goal vectors to use as hindsight goals
        """
        goals = []
        episode_length = len(episode)

        if self.her_strategy == "final":
            # Use final achieved state as goal
            final_achieved = episode[-1]["next_achieved_goal"]
            goals.extend([final_achieved] * self.her_k)

        elif self.her_strategy == "future":
            # Sample from future timesteps in the episode
            for _ in range(self.her_k):
                # Sample a future timestep (t < future_t <= episode_length)
                if t < episode_length - 1:
                    future_t = random.randint(t + 1, episode_length - 1)
                    future_achieved = episode[future_t]["next_achieved_goal"]
                    # I'd like to do something here where we randomly draw from the set
                    # of future nodes rather than the set of future states
                    goals.append(future_achieved)
                else:
                    # If at end of episode, use final state
                    goals.append(episode[-1]["next_achieved_goal"])

        return goals

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched tensors
        """
        # Sample random transitions
        batch = random.sample(self.transitions, min(batch_size, len(self.transitions)))

        # Stack into tensors
        obs = torch.FloatTensor(np.stack([t["obs"] for t in batch]))
        achieved_goal = torch.FloatTensor(np.stack([t["achieved_goal"] for t in batch]))
        desired_goal = torch.FloatTensor(np.stack([t["desired_goal"] for t in batch]))
        actions = torch.LongTensor([t["action"] for t in batch])
        rewards = torch.FloatTensor([t["reward"] for t in batch])
        next_obs = torch.FloatTensor(np.stack([t["next_obs"] for t in batch]))
        next_achieved_goal = torch.FloatTensor(np.stack([t["next_achieved_goal"] for t in batch]))
        dones = torch.FloatTensor([float(t["done"]) for t in batch])

        return {
            "obs": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "next_achieved_goal": next_achieved_goal,
            "dones": dones,
        }

    def __len__(self) -> int:
        return len(self.transitions)


class CustomDQN:
    """Custom DQN implementation with full transparency.

    This implementation provides complete control and visibility over the
    training process, unlike stable_baselines3's opaque implementation.
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        num_actions: int,
        config: DQNConfig | None = None,
        compute_reward_fn: Any = None,
    ):
        """Initialize Custom DQN.

        Args:
            obs_dim: Dimension of observation
            goal_dim: Dimension of goal vector
            num_actions: Number of discrete actions
            config: DQN configuration
            compute_reward_fn: Function to compute rewards for HER
        """
        self.config = config or DQNConfig()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.num_actions = num_actions

        # Set device
        if self.config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        print(f"CustomDQN using device: {self.device}")

        # Create Q-network and target network
        self.q_network = QNetwork(
            obs_dim, goal_dim, num_actions, self.config.hidden_sizes
        ).to(self.device)

        self.target_network = QNetwork(
            obs_dim, goal_dim, num_actions, self.config.hidden_sizes
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            her_k=self.config.her_k,
            her_strategy=self.config.her_strategy,
            compute_reward_fn=compute_reward_fn,
        )

        # Training state
        self.total_steps = 0
        self.epsilon = self.config.epsilon_start

        # Statistics
        self.q_losses = []
        self.avg_q_values = []
        self.avg_td_errors = []

    def select_action(
        self,
        obs: NDArray,
        achieved_goal: NDArray,
        desired_goal: NDArray,
        epsilon: float | None = None,
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation
            achieved_goal: Current achieved goal
            desired_goal: Desired goal
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        # Greedy action from Q-network
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            achieved_t = torch.FloatTensor(achieved_goal).unsqueeze(0).to(self.device)
            desired_t = torch.FloatTensor(desired_goal).unsqueeze(0).to(self.device)

            q_values = self.q_network(obs_t, achieved_t, desired_t)
            action = q_values.argmax(dim=1).item()

        return action

    def train_step(self) -> dict[str, float]:
        """Perform one training step.

        Returns:
            Dictionary of training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Move to device
        obs = batch["obs"].to(self.device)
        achieved_goal = batch["achieved_goal"].to(self.device)
        desired_goal = batch["desired_goal"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        next_achieved_goal = batch["next_achieved_goal"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(obs, achieved_goal, desired_goal)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_obs, next_achieved_goal, desired_goal)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards + (1.0 - dones) * self.config.gamma * max_next_q

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Statistics
        avg_q = current_q.mean().item()
        avg_td_error = (target_q - current_q).abs().mean().item()

        self.q_losses.append(loss.item())
        self.avg_q_values.append(avg_q)
        self.avg_td_errors.append(avg_td_error)

        return {
            "loss": loss.item(),
            "avg_q": avg_q,
            "avg_td_error": avg_td_error,
        }

    def update_target_network(self) -> None:
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self) -> None:
        """Update exploration rate."""
        # Linear decay
        decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon - decay_rate
        )

    def get_q_value(
        self,
        obs: NDArray,
        achieved_goal: NDArray,
        desired_goal: NDArray,
    ) -> float:
        """Get max Q-value for a state-goal pair.

        Args:
            obs: Observation
            achieved_goal: Achieved goal
            desired_goal: Desired goal

        Returns:
            Maximum Q-value over all actions
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            achieved_t = torch.FloatTensor(achieved_goal).unsqueeze(0).to(self.device)
            desired_t = torch.FloatTensor(desired_goal).unsqueeze(0).to(self.device)

            q_values = self.q_network(obs_t, achieved_t, desired_t)
            max_q = q_values.max().item()

        return max_q

    def train(
        self,
        env: Any,
        state_pairs: list[tuple[Any, Any]],
        perceiver: Any,
        max_training_steps: int,
        max_episode_steps: int = 100,
        log_freq: int = 1000,
        eval_freq: int = 5000,
        eval_episodes: int = 10,
    ) -> None:
        """Train the DQN on a set of state pairs using HER.

        This method provides full transparency into the training process with
        detailed logging at every step. It wraps the environment to provide
        goal-conditioned observations and trains using episodes sampled from
        the provided state pairs.

        Args:
            env: Base environment (must support reset_from_state)
            state_pairs: List of (source_state, target_state) tuples
            perceiver: Perceiver to convert states to atoms/predicates
            max_training_steps: Total number of environment steps to train
            max_episode_steps: Maximum steps per episode
            log_freq: Print training statistics every N steps
            eval_freq: Evaluate policy every N steps
            eval_episodes: Number of episodes for evaluation
        """
        from tamp_improv.approaches.improvisational.distance_heuristic import (
            DistanceHeuristicWrapper,
        )

        print("\n" + "=" * 80)
        print("CUSTOM DQN TRAINING")
        print("=" * 80)
        print(f"Training on {len(state_pairs)} state pairs")
        print(f"Max training steps: {max_training_steps}")
        print(f"Max episode steps: {max_episode_steps}")
        print(f"Device: {self.device}")
        print(f"Replay buffer capacity: {self.config.buffer_size}")
        print(f"HER strategy: '{self.config.her_strategy}' with k={self.config.her_k}")
        print("=" * 80 + "\n")

        # Wrap environment for goal-conditioned training
        wrapped_env = DistanceHeuristicWrapper(
            env,
            state_pairs,
            perceiver,
            max_episode_steps=max_episode_steps,
        )

        # Store wrapper's compute_reward function for HER
        self.replay_buffer.compute_reward_fn = wrapped_env.compute_reward

        # Training statistics
        episode_count = 0
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        current_episode = []

        # Sample first episode
        obs_dict, info = wrapped_env.reset()
        current_episode = []

        print("Starting training loop...\n")

        for step in range(max_training_steps):
            self.total_steps = step

            # Select action
            action = self.select_action(
                obs_dict["observation"],
                obs_dict["achieved_goal"],
                obs_dict["desired_goal"],
            )

            # Take step in environment
            next_obs_dict, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated

            # Store transition in current episode
            transition = {
                "obs": obs_dict["observation"],
                "achieved_goal": obs_dict["achieved_goal"],
                "desired_goal": obs_dict["desired_goal"],
                "action": action,
                "reward": reward,
                "next_obs": next_obs_dict["observation"],
                "next_achieved_goal": next_obs_dict["achieved_goal"],
                "done": done,
            }
            current_episode.append(transition)

            # If episode ended, store it in replay buffer (with HER)
            if done:
                self.replay_buffer.store_episode(current_episode)

                # Track episode statistics
                episode_reward = sum(t["reward"] for t in current_episode)
                episode_length = len(current_episode)
                episode_success = info.get("is_success", False)

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_successes.append(float(episode_success))
                episode_count += 1

                # Reset for next episode
                obs_dict, info = wrapped_env.reset()
                current_episode = []
            else:
                obs_dict = next_obs_dict

            # Update epsilon
            self.update_epsilon()

            # Training step
            if step >= self.config.learning_starts and len(self.replay_buffer) >= self.config.batch_size:
                if step % self.config.train_freq == 0:
                    metrics = self.train_step()

                    # Update target network
                    if step % self.config.target_update_freq == 0:
                        self.update_target_network()

            # Logging
            if step > 0 and step % log_freq == 0:
                print(f"\n{'='*80}")
                print(f"Step {step}/{max_training_steps}")
                print(f"{'='*80}")
                print(f"Episodes completed: {episode_count}")
                print(f"Replay buffer size: {len(self.replay_buffer)}/{self.config.buffer_size}")
                print(f"Epsilon: {self.epsilon:.3f}")

                if episode_rewards:
                    recent_n = min(100, len(episode_rewards))
                    recent_rewards = episode_rewards[-recent_n:]
                    recent_lengths = episode_lengths[-recent_n:]
                    recent_successes = episode_successes[-recent_n:]

                    print(f"\nRecent {recent_n} episodes:")
                    print(f"  Avg reward: {np.mean(recent_rewards):.2f}")
                    print(f"  Avg length: {np.mean(recent_lengths):.1f}")
                    print(f"  Success rate: {np.mean(recent_successes):.2%}")

                if self.q_losses:
                    recent_losses = self.q_losses[-100:]
                    recent_q_vals = self.avg_q_values[-100:]
                    recent_td_errs = self.avg_td_errors[-100:]

                    print(f"\nTraining metrics (last 100 updates):")
                    print(f"  Avg loss: {np.mean(recent_losses):.4f}")
                    print(f"  Avg Q-value: {np.mean(recent_q_vals):.2f}")
                    print(f"  Avg TD error: {np.mean(recent_td_errs):.2f}")

                print(f"{'='*80}\n")

            # Evaluation
            if step > 0 and step % eval_freq == 0:
                self._evaluate(
                    wrapped_env,
                    state_pairs,
                    eval_episodes,
                    max_episode_steps,
                    step,
                )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total steps: {max_training_steps}")
        print(f"Total episodes: {episode_count}")
        print(f"Final epsilon: {self.epsilon:.3f}")
        print(f"Final replay buffer size: {len(self.replay_buffer)}")

        if episode_rewards:
            print(f"\nFinal performance (last 100 episodes):")
            recent_n = min(100, len(episode_rewards))
            print(f"  Avg reward: {np.mean(episode_rewards[-recent_n:]):.2f}")
            print(f"  Avg length: {np.mean(episode_lengths[-recent_n:]):.1f}")
            print(f"  Success rate: {np.mean(episode_successes[-recent_n:]):.2%}")

        print("=" * 80 + "\n")

    def _evaluate(
        self,
        wrapped_env: Any,
        state_pairs: list[tuple[Any, Any]],
        num_episodes: int,
        max_steps: int,
        current_step: int,
    ) -> None:
        """Evaluate current policy on random state pairs.

        Args:
            wrapped_env: Wrapped environment
            state_pairs: List of state pairs to sample from
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            current_step: Current training step (for logging)
        """
        print(f"\n{'─'*80}")
        print(f"EVALUATION at step {current_step}")
        print(f"{'─'*80}")

        eval_rewards = []
        eval_lengths = []
        eval_successes = []

        # Save current epsilon and set to greedy
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # Greedy evaluation

        for ep in range(num_episodes):
            obs_dict, _ = wrapped_env.reset()
            episode_reward = 0.0
            episode_length = 0

            for _ in range(max_steps):
                action = self.select_action(
                    obs_dict["observation"],
                    obs_dict["achieved_goal"],
                    obs_dict["desired_goal"],
                )

                obs_dict, reward, terminated, truncated, info = wrapped_env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(float(info.get("is_success", False)))

        # Restore epsilon
        self.epsilon = old_epsilon

        print(f"Evaluation results ({num_episodes} episodes):")
        print(f"  Avg reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  Avg length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")
        print(f"  Success rate: {np.mean(eval_successes):.2%}")
        print(f"{'─'*80}\n")

    def save(self, path: str) -> None:
        """Save model weights.

        Args:
            path: Path to save file
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
        }, path)
        print(f"CustomDQN saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights.

        Args:
            path: Path to save file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]
        print(f"CustomDQN loaded from {path}")
