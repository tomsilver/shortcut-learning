"""Goal-conditioned distance heuristic V2 for pruning shortcuts.

This module implements a learned distance function f(s, s') that estimates
the number of steps required to reach target state s' from source state s.

V2 operates on state-state pairs rather than state-node pairs to avoid
discontinuity issues when goal nodes change. The heuristic is trained using
goal-conditioned RL with a reward of -1 per step, which causes the value
function to approximate V(s, s') = -distance(s, s').
"""

import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class DistanceHeuristicConfig:
    """Configuration for distance heuristic training."""

    # Algorithm (SAC for continuous, DQN for discrete)
    # Will be automatically selected based on action space
    algorithm: str = "auto"  # "auto", "SAC", "DQN", or "custom_dqn"

    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000

    # Training parameters
    max_episode_steps: int = 100
    learning_starts: int = 1000

    # Device settings
    device: str = "cuda"

    # Network architecture
    hidden_sizes: list[int] = None  # [256, 256] default

    # Learning parameters
    gamma: float = 0.99  # Discount factor

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000

    # Target network
    train_freq: int = 500
    target_update_freq: int = 1000  # Update target network every N steps

    # HER parameters
    her_k: int = 4  # Number of hindsight goals per transition
    her_strategy: str = "future"  # "final" or "future"

    # Training
    learning_starts: int = 1000  # Start learning after N steps

    # Logging and evaluation
    log_freq: int = 1000  # Log progress every N steps
    eval_freq: int = 5000  # Evaluate every N steps

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


def compute_reward(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        eps=1e-3
    ) -> np.ndarray:
        """Compute reward for HER.

        Returns 0.0 if goal is achieved (for success), -1.0 otherwise.
        This sparse reward structure is what HER is designed to handle.

        Goals are represented as multi-hot vectors where each dimension
        corresponds to an atom. A goal is satisfied when achieved_goal
        contains all atoms present in desired_goal (superset check).
        This follows the standard goal-conditioned RL semantics.

        The learned Q-values will still approximate negative distance because:
        - Reward is -1 per step until goal
        - Total return = -(number of steps to goal)
        - Therefore Q(s,a,g) ≈ -distance(s,g)
        """
        # Batch of goals - check each row
        if achieved_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)
            achieved_goal = achieved_goal.reshape(1, -1)
        batch_size = achieved_goal.shape[0]
        rewards = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            goal_achieved = (np.linalg.norm(achieved_goal[i] - desired_goal[i]) < eps)
            rewards[i] = 0.0 if goal_achieved else -1.0
        return rewards




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
        gamma: float = 0.99,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Dimension of observation
            goal_dim: Dimension of goal vector
            her_k: Number of hindsight goals per real transition
            her_strategy: "final" (use final state) or "future" (use future states)
            gamma: Discount factor (used for geometric sampling of future goals)
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.her_k = her_k
        self.her_strategy = her_strategy
        self.gamma = gamma

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
        for k in range(self.her_k):
            # Sample k hindsight goals
            t = np.random.randint(0, episode_length)
            hindsight_goals = self._sample_hindsight_goals(episode, t)

            for hindsight_goal in hindsight_goals:
                # Create new transition with hindsight goal
                original = episode[t]

                new_reward = compute_reward(original["next_achieved_goal"], hindsight_goal)[0]

                # Check if episode terminates with this hindsight goal
                new_done = (new_reward == 0)

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

        Uses a geometric distribution to bias toward distant future states,
        which helps avoid sampling goals that are too similar to the current state.

        Args:
            episode: Full episode
            t: Timestep index

        Returns:
            List of goal vectors to use as hindsight goals
        """
        goals = []
        episode_length = len(episode)

        num_goals = 1

        if self.her_strategy == "final":
            # Use final achieved state as goal
            final_achieved = episode[-1]["next_achieved_goal"]
            goals.extend([final_achieved] * num_goals)

        elif self.her_strategy == "future":
            # Sample from future timesteps using geometric distribution
            # This biases toward distant states, avoiding trivial Q=0 examples
            # We use (1 - gamma) as the probability parameter, where gamma is the discount factor
            # Higher gamma (e.g., 0.99) → lower p (0.01) → bias toward distant states
            # This aligns with the RL intuition: high gamma = care about distant future
            p = 1.0 - self.gamma

            for _ in range(num_goals):
                if t < episode_length - 1:
                    # Number of future timesteps available
                    num_future = episode_length - t - 1

                    # Sample offset from geometric distribution
                    # offset ranges from 0 to num_future-1
                    offset = min(
                        np.random.geometric(p) - 1,  # -1 because geometric starts at 1
                        num_future - 1
                    )

                    future_t = t + 1 + offset
                    future_achieved = episode[future_t]["next_achieved_goal"]
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

        
class DistanceHeuristicWrapperV2(gym.Wrapper):
    """Wrapper that provides -1 reward per step for distance learning.

    This wrapper modifies the reward structure to be -1 per step, regardless
    of whether the goal is reached. This causes the learned value function
    V(s, z') to approximate -distance(s, z'), where s is the low-level state
    and z' is the abstract goal state (atoms).
    """

    def __init__(
        self,
        env: gym.Env,
        state_pairs: list[tuple[ObsType, ObsType]],
        perceiver: Any,
        max_episode_steps: int = 100,
    ):
        """Initialize wrapper with state pairs to sample from.

        Args:
            env: Base environment (should support reset_from_state)
            state_pairs: List of (source_state, goal_state) tuples
            perceiver: Perceiver (not used in V2, kept for compatibility)
            max_episode_steps: Maximum steps per episode
        """
        super().__init__(env)
        self.state_pairs = state_pairs
        self.perceiver = perceiver
        self.max_episode_steps = max_episode_steps
        self.steps = 0

        # DEBUG: Track rewards
        self._reward_history: list[float] = []
        self._episode_count = 0

        # Determine observation space structure
        if len(state_pairs) > 0:
            sample_state = state_pairs[0][0]

            # Determine state observation space
            if hasattr(sample_state, "nodes"):
                # Graph observation
                flattened_size = sample_state.nodes.flatten().shape[0]
                obs_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32
                )
            else:
                # Array observation
                flattened = np.array(sample_state).flatten()
                obs_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=flattened.shape, dtype=np.float32
                )


            # Goal-conditioned observation space
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": obs_space,
                    "achieved_goal": obs_space,
                    "desired_goal": obs_space,
                }
            )

    def flatten_obs(self, obs: ObsType) -> np.ndarray:
        """Flatten observation to array."""
        if hasattr(obs, "nodes"):
            return obs.nodes.flatten().astype(np.float32)
        return np.array(obs).flatten().astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment by sampling a random state pair."""
        self.steps = 0

        # Sample a random state pair
        pair_idx = np.random.randint(len(self.state_pairs))
        source_state, goal_state = self.state_pairs[pair_idx]
        self.goal_state = goal_state


        # Reset from source state
        # Need to access unwrapped env to get reset_from_state method
        unwrapped_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(unwrapped_env, "reset_from_state"):
            obs, info = unwrapped_env.reset_from_state(source_state, seed=seed)
        else:
            raise AttributeError(
                f"Environment must have reset_from_state method. "
                f"Environment type: {type(unwrapped_env)}"
            )

        # Create goal-conditioned observation
        flat_obs = self.flatten_obs(obs)
        goal_vector = self.flatten_obs(self.goal_state)


        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": flat_obs,
            "desired_goal": goal_vector,
        }

        return dict_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step environment with -1 reward per step."""
        next_obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        flat_obs = self.flatten_obs(next_obs)
        goal_vector = self.flatten_obs(self.goal_state)

        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": flat_obs,
            "desired_goal": goal_vector,
        }

        # Use compute_reward for consistency with HER
        # Returns 0.0 on success, -1.0 otherwise
        reward = float(compute_reward(flat_obs, goal_vector)[0])

        # DEBUG: Track rewards for first few steps
        self._reward_history.append(reward)
        if len(self._reward_history) <= 10:
            print(f"[DEBUG REWARD STEP {len(self._reward_history)}] reward={reward:.1f}")

        # Check if goal is reached (atoms match)
        goal_reached = (reward == 0)
        info["is_success"] = bool(goal_reached)

        # Terminate if goal reached or max steps
        terminated = terminated or goal_reached
        truncated = truncated or (self.steps >= self.max_episode_steps)

        # DEBUG: Log episode completion
        if terminated or truncated:
            self._episode_count += 1
            if self._episode_count <= 3:
                episode_reward = sum(self._reward_history[-self.steps:])
                print(f"[DEBUG EPISODE {self._episode_count}] "
                      f"length={self.steps}, total_reward={episode_reward:.1f}, "
                      f"success={goal_reached}")

        return dict_obs, reward, terminated, truncated, info

    


class GoalConditionedDistanceHeuristicV2:
    """Learns distance function f(s, s') via goal-conditioned RL.

    Trains a policy with reward=-1 per step, causing the value function to
    learn V(s, g) ≈ -distance(s, g). The heuristic can then estimate distances
    by querying the critic/value network.
    """

    def __init__(self, config: DistanceHeuristicConfig | None = None, seed: int = 42):
        """Initialize distance heuristic.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config or DistanceHeuristicConfig()
        self.seed = seed

        if self.config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Networks (V2 only uses custom implementation, no stable-baselines3)
        self.q_network: QNetwork | None = None
        self.target_network: QNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ReplayBuffer | None = None

        self.perceiver: Any | None = None
        self.wrapper: DistanceHeuristicWrapperV2 | None = None
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

        # Training state
        self.total_steps = 0
        self.epsilon = self.config.epsilon_start

    def train(
        self,
        env: gym.Env,
        state_pairs: list[tuple[ObsType, ObsType]],
        perceiver: Any,
        max_training_steps: int,
    ) -> None:
        """Train distance heuristic on sampled state pairs.

        Args:
            env: Base environment
            state_pairs: List of (source, target) state pairs
            perceiver: Perceiver to convert states to atoms
            max_training_steps: Total training timesteps
        """
        print(f"\nTraining distance heuristic V2 on {len(state_pairs)} state pairs...")
        print(f"Device: {self.device}")

        # Store perceiver for use in estimate_distance
        self.perceiver = perceiver

        # Wrap environment for distance learning
        wrapped_env = DistanceHeuristicWrapperV2(
            env,
            state_pairs,
            perceiver,
            max_episode_steps=self.config.max_episode_steps,
        )

        # Store wrapper
        self.wrapper = wrapped_env

        # Compute observation statistics for normalization
        self._compute_obs_statistics(state_pairs)

        print("\n" + "=" * 80)
        print("Using V2 self-contained DQN implementation")
        print("=" * 80)

        # Determine observation and goal dimensions
        sample_obs = wrapped_env.reset()[0]
        obs_dim = sample_obs["observation"].shape[0]
        goal_dim = sample_obs["desired_goal"].shape[0]
        num_actions = wrapped_env.action_space.n

        print(f"Observation dim: {obs_dim}")
        print(f"Goal dim: {goal_dim}")
        print(f"Number of actions: {num_actions}")

        # Create networks
        self.q_network = QNetwork(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes or [256, 256],
        ).to(self.device)

        self.target_network = QNetwork(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes or [256, 256],
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            her_k=self.config.her_k,
            her_strategy=self.config.her_strategy,
            gamma=self.config.gamma,
        )

        # Training loop
        self.total_steps = 0
        self.epsilon = self.config.epsilon_start

        print(f"\nStarting training for {max_training_steps} steps...")

        while self.total_steps < max_training_steps:
            # Collect episode
            episode = self._collect_episode(wrapped_env)

            # Store episode in replay buffer (with HER)
            self.replay_buffer.store_episode(episode)

            # Train if we have enough data
            if len(self.replay_buffer) >= self.config.learning_starts:
                if self.total_steps % self.config.train_freq == 0:
                    self._train_step()

            # Update target network
            if self.total_steps % self.config.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Logging
            if self.total_steps % self.config.log_freq == 0 and self.total_steps > 0:
                self._log_progress(wrapped_env)

            # Evaluation
            if self.total_steps % self.config.eval_freq == 0 and self.total_steps > 0:
                self._evaluate(wrapped_env)

        print(f"\nTraining complete!")

    def _collect_episode(self, env: gym.Env) -> list[dict[str, Any]]:
        """Collect a single episode using epsilon-greedy policy."""
        episode = []
        obs, _ = env.reset()

        done = False
        truncated = False

        while not done and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action = env.action_space.sample()
            else:
                action = self._select_action(obs)

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Store transition
            transition = {
                "obs": obs["observation"],
                "achieved_goal": obs["achieved_goal"],
                "desired_goal": obs["desired_goal"],
                "action": action,
                "reward": reward,
                "next_obs": next_obs["observation"],
                "next_achieved_goal": next_obs["achieved_goal"],
                "done": done or truncated,
            }
            episode.append(transition)

            obs = next_obs
            self.total_steps += 1

            # Update epsilon
            if self.total_steps < self.config.epsilon_decay_steps:
                self.epsilon = self.config.epsilon_start + (
                    self.config.epsilon_end - self.config.epsilon_start
                ) * (self.total_steps / self.config.epsilon_decay_steps)
            else:
                self.epsilon = self.config.epsilon_end

        return episode

    def _select_action(self, obs: dict[str, np.ndarray]) -> int:
        """Select action using Q-network (greedy)."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(self.device)
            achieved_goal_t = torch.FloatTensor(obs["achieved_goal"]).unsqueeze(0).to(self.device)
            desired_goal_t = torch.FloatTensor(obs["desired_goal"]).unsqueeze(0).to(self.device)

            q_values = self.q_network(obs_t, achieved_goal_t, desired_goal_t)
            action = torch.argmax(q_values, dim=1).item()

        return action

    def _train_step(self) -> None:
        """Perform one training step (batch update)."""
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

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_obs, next_achieved_goal, desired_goal)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * max_next_q

        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        print("Loss:", loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _log_progress(self, env: gym.Env) -> None:
        """Log training progress."""
        print(f"\n[Step {self.total_steps}]")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Replay buffer size: {len(self.replay_buffer)}")

        # Sample Q-values for diagnostics
        if len(self.replay_buffer) > 0:
            batch = self.replay_buffer.sample(min(256, len(self.replay_buffer)))
            with torch.no_grad():
                obs = batch["obs"].to(self.device)
                achieved_goal = batch["achieved_goal"].to(self.device)
                desired_goal = batch["desired_goal"].to(self.device)
                q_values = self.q_network(obs, achieved_goal, desired_goal)
                avg_q = torch.mean(torch.max(q_values, dim=1)[0]).item()
                print(f"  Avg Q-value: {avg_q:.3f}")

    def _evaluate(self, env: gym.Env) -> None:
        """Evaluate policy with greedy actions."""
        print(f"\n[Evaluation at step {self.total_steps}]")

        num_episodes = 10
        episode_lengths = []
        episode_rewards = []
        success_count = 0

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_length = 0
            episode_reward = 0.0

            while not done and not truncated:
                # Greedy action selection
                action = self._select_action(obs)
                obs, reward, done, truncated, info = env.step(action)

                episode_length += 1
                episode_reward += reward

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            if info.get("is_success", False):
                success_count += 1

        print(f"  Avg episode length: {np.mean(episode_lengths):.2f}")
        print(f"  Avg episode reward: {np.mean(episode_rewards):.2f}")
        print(f"  Success rate: {success_count}/{num_episodes}")

    def estimate_distance(self, source_state: ObsType, target_state: ObsType) -> float:
        """Estimate distance from source to target state.

        Args:
            source_state: Starting state
            target_state: Goal state (raw state, not atoms)

        Returns:
            Estimated distance (number of steps)
        """
        assert self.q_network is not None, "Model must be trained before estimation"

        # Flatten source and target states (no normalization, no atom conversion)
        if hasattr(source_state, "nodes"):
            source_flat = source_state.nodes.flatten().astype(np.float32)
        else:
            source_flat = np.array(source_state).flatten().astype(np.float32)

        if hasattr(target_state, "nodes"):
            target_flat = target_state.nodes.flatten().astype(np.float32)
        else:
            target_flat = np.array(target_state).flatten().astype(np.float32)

        # In V2, achieved_goal and desired_goal are just flattened state vectors
        # (not atom vectors like in V1)
        achieved_goal_vector = source_flat
        desired_goal_vector = target_flat

        # Get value estimate from Q-network (Q(s, a, g) ≈ -distance)
        with torch.no_grad():
            obs_t = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            achieved_goal_t = torch.FloatTensor(achieved_goal_vector).unsqueeze(0).to(self.device)
            desired_goal_t = torch.FloatTensor(desired_goal_vector).unsqueeze(0).to(self.device)

            q_values = self.q_network(obs_t, achieved_goal_t, desired_goal_t)
            q_value = torch.max(q_values).item()

        # Since reward = -1 per step, Q(s,a,g) ≈ -distance
        # So distance ≈ -Q(s,a,g)
        estimated_distance = -q_value

        # Clamp to reasonable range
        estimated_distance = max(0.0, estimated_distance)

        return float(estimated_distance)

    def batch_estimate_distances(
        self,
        state_pairs: list[tuple[ObsType, ObsType]],
    ) -> np.ndarray:
        """Estimate distances for multiple state pairs efficiently.

        Args:
            state_pairs: List of (source, target) state pairs

        Returns:
            Array of estimated distances
        """
        distances = np.zeros(len(state_pairs))

        for i, (source, target) in enumerate(state_pairs):
            distances[i] = self.estimate_distance(source, target)

        return distances

    def save(self, path: str) -> None:
        """Save the distance heuristic.

        Args:
            path: Directory path to save model

        Note:
            Perceiver is not saved and must be provided when loading.
        """
        assert self.q_network is not None, "Model must be trained before saving"

        os.makedirs(path, exist_ok=True)

        # Save Q-network weights
        torch.save(self.q_network.state_dict(), f"{path}/q_network.pt")

        # Save network architecture info
        with open(f"{path}/network_config.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_dim": self.q_network.obs_dim,
                    "goal_dim": self.q_network.goal_dim,
                    "num_actions": self.q_network.num_actions,
                    "hidden_sizes": self.config.hidden_sizes,
                },
                f,
            )

        # Save observation statistics
        with open(f"{path}/obs_stats.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_mean": self._obs_mean,
                    "obs_std": self._obs_std,
                },
                f,
            )

        print(f"Distance heuristic V2 saved to {path}")

    def load(self, path: str, perceiver: Any) -> None:
        """Load a pre-trained distance heuristic.

        Args:
            path: Directory path containing saved model
            perceiver: Perceiver to convert states to atoms (not used in V2)
        """
        # Store perceiver
        self.perceiver = perceiver

        # Load observation statistics
        with open(f"{path}/obs_stats.pkl", "rb") as f:
            stats = pickle.load(f)
            self._obs_mean = stats["obs_mean"]
            self._obs_std = stats["obs_std"]

        # Load network config
        with open(f"{path}/network_config.pkl", "rb") as f:
            network_config = pickle.load(f)

        # Create Q-network with saved architecture
        self.q_network = QNetwork(
            obs_dim=network_config["obs_dim"],
            goal_dim=network_config["goal_dim"],
            num_actions=network_config["num_actions"],
            hidden_sizes=network_config["hidden_sizes"],
        ).to(self.device)

        # Load weights
        self.q_network.load_state_dict(torch.load(f"{path}/q_network.pt", map_location=self.device))
        self.q_network.eval()

        print(f"Distance heuristic V2 loaded from {path}")

    def _flatten_and_normalize(self, obs: ObsType) -> np.ndarray:
        """Flatten and normalize observation."""
        if hasattr(obs, "nodes"):
            flat = obs.nodes.flatten().astype(np.float32)
        else:
            flat = np.array(obs).flatten().astype(np.float32)

        # Normalize if statistics are available
        if self._obs_mean is not None and self._obs_std is not None:
            flat = (flat - self._obs_mean) / (self._obs_std + 1e-8)

        return flat

    def _compute_obs_statistics(self, state_pairs: list[tuple[ObsType, ObsType]]) -> None:
        """Compute mean and std of observations for normalization."""
        all_obs = []

        for source, target in state_pairs:
            if hasattr(source, "nodes"):
                all_obs.append(source.nodes.flatten())
                all_obs.append(target.nodes.flatten())
            else:
                all_obs.append(np.array(source).flatten())
                all_obs.append(np.array(target).flatten())

        all_obs_array = np.array(all_obs, dtype=np.float32)
        self._obs_mean = np.mean(all_obs_array, axis=0)
        self._obs_std = np.std(all_obs_array, axis=0)

        print(f"Computed observation statistics: mean={self._obs_mean.mean():.3f}, std={self._obs_std.mean():.3f}")
