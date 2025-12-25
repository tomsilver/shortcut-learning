"""Goal-conditioned distance heuristic for pruning shortcuts.

This module implements a learned distance function f(s, z') that estimates
the number of steps required to reach abstract goal state z' from state s.
The heuristic is trained using goal-conditioned RL with a reward of -1 per step,
which causes the value function to approximate V(s, z') = -distance(s, z').

Note: z' is an abstract state (set of atoms/predicates), not a low-level state.
This matches what policies actually see during execution.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her import HerReplayBuffer

from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper

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

    # Custom DQN specific parameters
    use_custom_dqn: bool = False  # If True, use custom DQN implementation instead of stable_baselines3
    custom_dqn_hidden_sizes: list[int] | None = None  # Hidden layer sizes for custom DQN
    custom_dqn_target_update_freq: int = 1000  # Target network update frequency
    custom_dqn_epsilon_decay_steps: int = 10000  # Steps to decay epsilon
    custom_dqn_her_k: int = 4  # Number of HER goals per transition
    custom_dqn_log_freq: int = 1000  # Logging frequency
    custom_dqn_eval_freq: int = 5000  # Evaluation frequency
    custom_dqn_train_freq: int = 500 # Train frequency


class DistanceHeuristicCallback(BaseCallback):
    """Callback to track distance heuristic training progress."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1, debug: bool = False):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.debug = debug
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.current_length = 0
        self.current_reward = 0.0
        self.debug_sample_count = 0

    def _on_step(self) -> bool:
        """Called at each step of training."""
        self.current_length += 1
        self.current_reward += self.locals["rewards"][0]
        dones = self.locals["dones"]

        # DEBUG: Log first few transitions in detail
        if self.debug and self.debug_sample_count < 5:
            obs = self.locals.get("new_obs", [{}])[0]
            reward = self.locals["rewards"][0]
            print(f"\n[DEBUG TRANSITION #{self.debug_sample_count}]")
            print(f"  Reward: {reward:.3f}")
            if "desired_goal" in obs:
                desired_goal = obs["desired_goal"]
                achieved_goal = obs["achieved_goal"]
                goal_atoms = np.where(desired_goal > 0.5)[0]
                achieved_atoms = np.where(achieved_goal > 0.5)[0]
                print(f"  Desired goal atoms (indices): {goal_atoms}")
                print(f"  Achieved goal atoms (indices): {achieved_atoms}")
            self.debug_sample_count += 1

        if dones[0]:
            self.episode_lengths.append(self.current_length)
            self.episode_rewards.append(self.current_reward)
            self.current_length = 0
            self.current_reward = 0.0

        if self.num_timesteps % self.check_freq == 0:
            print("\nDistance Heuristic Training Progress:")
            print(f"Timesteps: {self.num_timesteps}")

            if self.episode_lengths:
                n_recent = min(100, len(self.episode_lengths))
                recent_lengths = self.episode_lengths[-n_recent:]
                recent_rewards = self.episode_rewards[-n_recent:]

                print(f"Episodes completed: {len(self.episode_lengths)}")
                print(
                    f"Recent Avg Episode Length: {np.mean(recent_lengths):.2f}"
                )
                print(
                    f"Recent Avg Reward: {np.mean(recent_rewards):.2f}"
                )

            buffer = self.locals.get("replay_buffer")
            if buffer is not None:
                print(f"Replay buffer size: {buffer.size()}/{buffer.buffer_size}")

                # DEBUG: Sample from replay buffer and show Q-values
                if self.debug and buffer.size() > 100:
                    self._debug_replay_buffer(buffer)

        return True

    def _debug_replay_buffer(self, buffer):
        """Debug helper to inspect replay buffer contents."""
        try:
            # Sample a small batch from buffer
            if hasattr(buffer, 'sample'):
                batch_size = min(5, buffer.size())
                replay_data = buffer.sample(batch_size)

                print(f"\n[DEBUG REPLAY BUFFER SAMPLE - {batch_size} transitions]")
                print(f"  Rewards: {replay_data.rewards.cpu().numpy().flatten()}")
                print(f"  Dones: {replay_data.dones.cpu().numpy().flatten()}")

                # Try to get Q-values if model is available
                if hasattr(self.model, 'q_net'):
                    with torch.no_grad():
                        q_values = self.model.q_net(replay_data.observations)
                        max_q = torch.max(q_values, dim=1)[0]
                        print(f"  Current Q-values (max): {max_q.cpu().numpy()}")
        except Exception as e:
            print(f"[DEBUG] Could not sample buffer: {e}")


class DistanceHeuristicWrapper(gym.Wrapper):
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
        max_atom_size: int = 50,
    ):
        """Initialize wrapper with state pairs to sample from.

        Args:
            env: Base environment (should support reset_from_state)
            state_pairs: List of (source_state, goal_state) tuples
            perceiver: Perceiver to convert observations to atoms
            max_episode_steps: Maximum steps per episode
            max_atom_size: Maximum number of unique atoms (for fixed-size vector)
        """
        super().__init__(env)
        self.state_pairs = state_pairs
        self.perceiver = perceiver
        self.max_episode_steps = max_episode_steps
        self.max_atom_size = max_atom_size
        self.steps = 0
        self.current_goal_atoms: set | None = None

        # Dynamic atom indexing (like GoalConditionedWrapper)
        self.atom_to_index: dict[str, int] = {}
        self._next_index = 0

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

            # Goal space is fixed-size atom vector (multi-hot encoding)
            goal_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(max_atom_size,), dtype=np.float32
            )

            # Goal-conditioned observation space
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": obs_space,
                    "achieved_goal": goal_space,
                    "desired_goal": goal_space,
                }
            )

    def flatten_obs(self, obs: ObsType) -> np.ndarray:
        """Flatten observation to array."""
        if hasattr(obs, "nodes"):
            return obs.nodes.flatten().astype(np.float32)
        return np.array(obs).flatten().astype(np.float32)

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom (dynamic indexing)."""
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert (
            self._next_index < self.max_atom_size
        ), f"No more space for new atom at index {self._next_index}. Increase max_atom_size (currently {self.max_atom_size})."
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def create_atom_vector(self, atoms: set) -> np.ndarray:
        """Create a multi-hot vector representation of the set of atoms."""
        vector = np.zeros(self.max_atom_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment by sampling a random state pair."""
        self.steps = 0

        # Sample a random state pair
        pair_idx = np.random.randint(len(self.state_pairs))
        source_state, goal_state = self.state_pairs[pair_idx]

        # Convert goal state to atoms (abstract representation)
        self.current_goal_atoms = self.perceiver.step(goal_state)

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
        goal_vector = self.create_atom_vector(self.current_goal_atoms)

        # Current achieved atoms
        current_atoms = self.perceiver.step(obs)
        achieved_goal_vector = self.create_atom_vector(current_atoms)

        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": achieved_goal_vector,
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
        goal_vector = self.create_atom_vector(self.current_goal_atoms)

        # Get current achieved atoms
        current_atoms = self.perceiver.step(next_obs)
        achieved_goal_vector = self.create_atom_vector(current_atoms)

        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": achieved_goal_vector,
            "desired_goal": goal_vector,
        }

        # Use compute_reward for consistency with HER
        # Returns 0.0 on success, -1.0 otherwise
        reward = float(self.compute_reward(achieved_goal_vector, goal_vector, info)[0])

        # DEBUG: Track rewards for first few steps
        self._reward_history.append(reward)
        if len(self._reward_history) <= 10:
            print(f"[DEBUG REWARD STEP {len(self._reward_history)}] reward={reward:.1f}, "
                  f"goal_reached={np.allclose(achieved_goal_vector, goal_vector)}")

        # Check if goal is reached (atoms match)
        goal_reached = current_atoms == self.current_goal_atoms
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

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: list[dict[str, Any]] | dict[str, Any],
        _indices: list[int] | None = None,
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
        if achieved_goal.ndim == 1:
            # Single goal - check if achieved contains all desired atoms
            goal_indices = np.where(desired_goal > 0.5)[0]
            if len(goal_indices) == 0:
                # No atoms required - trivially satisfied
                goal_achieved = True
            else:
                goal_achieved = np.all(achieved_goal[goal_indices] > 0.5)
            return np.array([0.0 if goal_achieved else -1.0], dtype=np.float32)
        else:
            # Batch of goals - check each row
            batch_size = achieved_goal.shape[0]
            rewards = np.zeros(batch_size, dtype=np.float32)
            for i in range(batch_size):
                goal_indices = np.where(desired_goal[i] > 0.5)[0]
                if len(goal_indices) == 0:
                    goal_satisfied = True
                else:
                    goal_satisfied = np.all(achieved_goal[i][goal_indices] > 0.5)
                rewards[i] = 0.0 if goal_satisfied else -1.0
            return rewards


class GoalConditionedDistanceHeuristic:
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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.model: SAC | DQN | None = None
        self.algorithm_used: str | None = None  # Track which algorithm was used
        self.perceiver: Any | None = None
        self.wrapper: DistanceHeuristicWrapper | None = None
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

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
        print(f"\nTraining distance heuristic on {len(state_pairs)} state pairs...")
        print(f"Device: {self.device}")

        # DEBUG: Show statistics about training pairs
        print(f"\n[DEBUG TRAINING PAIRS]")
        print(f"  Total pairs: {len(state_pairs)}")
        if len(state_pairs) > 0:
            # Sample a few pairs and show their atoms
            print(f"  Sampling first 3 pairs to show atom representations:")
            for i, (source, target) in enumerate(state_pairs[:3]):
                source_atoms = perceiver.step(source)
                target_atoms = perceiver.step(target)
                print(f"    Pair {i}: {len(source_atoms)} source atoms -> {len(target_atoms)} target atoms")
                print(f"      Source: {sorted([str(a) for a in source_atoms])[:3]}...")
                print(f"      Target: {sorted([str(a) for a in target_atoms])[:3]}...")

        # Store perceiver for use in estimate_distance
        self.perceiver = perceiver

        # Wrap environment for distance learning (converts goal states to atoms internally)
        wrapped_env = DistanceHeuristicWrapper(
            env,
            state_pairs,
            perceiver,
            max_episode_steps=self.config.max_episode_steps,
        )

        # Store wrapper so we can use its atom_to_index mapping
        self.wrapper = wrapped_env

        # Compute observation statistics for normalization
        self._compute_obs_statistics(state_pairs)

        # Check if using custom DQN implementation
        if self.config.use_custom_dqn:
            from tamp_improv.approaches.improvisational.custom_dqn import (
                CustomDQN,
                DQNConfig,
            )

            print("\n" + "=" * 80)
            print("Using CUSTOM DQN implementation (transparent, debuggable)")
            print("=" * 80)

            # Determine observation and goal dimensions
            sample_obs = wrapped_env.reset()[0]
            obs_dim = sample_obs["observation"].shape[0]
            goal_dim = sample_obs["desired_goal"].shape[0]
            num_actions = wrapped_env.action_space.n

            print(f"Observation dim: {obs_dim}")
            print(f"Goal dim: {goal_dim}")
            print(f"Number of actions: {num_actions}")

            # Create custom DQN config
            custom_config = DQNConfig(
                hidden_sizes=self.config.custom_dqn_hidden_sizes or [256, 256],
                learning_rate=self.config.learning_rate,
                gamma=0.99,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_steps=self.config.custom_dqn_epsilon_decay_steps,
                target_update_freq=self.config.custom_dqn_target_update_freq,
                her_k=self.config.custom_dqn_her_k,
                her_strategy="future",
                learning_starts=self.config.learning_starts,
                train_freq=self.config.custom_dqn_train_freq,
                device=self.device,
            )

            # Create custom DQN
            self.model = CustomDQN(
                obs_dim=obs_dim,
                goal_dim=goal_dim,
                num_actions=num_actions,
                config=custom_config,
                compute_reward_fn=wrapped_env.compute_reward,
            )

            self.algorithm_used = "custom_dqn"

            # Train using custom DQN's train method
            self.model.train(
                env=env,
                state_pairs=state_pairs,
                perceiver=perceiver,
                max_training_steps=max_training_steps,
                max_episode_steps=self.config.max_episode_steps,
                log_freq=self.config.custom_dqn_log_freq,
                eval_freq=self.config.custom_dqn_eval_freq,
            )

            print(f"\nCustom DQN training complete!")

        else:
            # Use stable_baselines3 implementation
            # Determine which algorithm to use based on action space
            action_space = wrapped_env.action_space
            if self.config.algorithm == "auto":
                if isinstance(action_space, gym.spaces.Box):
                    algorithm = "SAC"
                elif isinstance(action_space, gym.spaces.Discrete):
                    algorithm = "DQN"
                else:
                    raise ValueError(
                        f"Unsupported action space type: {type(action_space)}. "
                        "Only Box (continuous) and Discrete are supported."
                    )
            else:
                algorithm = self.config.algorithm

            self.algorithm_used = algorithm
            print(f"Using algorithm: {algorithm} for action space: {type(action_space).__name__}")

            # Create model based on algorithm
            if algorithm == "SAC":
                self.model = SAC(
                    "MultiInputPolicy",
                    wrapped_env,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    buffer_size=self.config.buffer_size,
                    learning_starts=self.config.learning_starts,
                    device=self.device,
                    seed=self.seed,
                    verbose=1,
                )
            elif algorithm == "DQN":
                # Use HER (Hindsight Experience Replay) for goal-conditioned learning
                # This is CRITICAL for learning with sparse rewards
                self.model = DQN(
                    "MultiInputPolicy",
                    wrapped_env,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    buffer_size=self.config.buffer_size,
                    learning_starts=self.config.learning_starts,
                    device=self.device,
                    seed=self.seed,
                    verbose=1,
                    gamma=0.99,
                    exploration_fraction=0.1,  # DQN-specific: fraction of training for exploration
                    exploration_final_eps=0.05,  # DQN-specific: final exploration rate
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                        n_sampled_goal=4,  # Number of virtual transitions per real transition
                        goal_selection_strategy="final",  # Use future states as goals
                    ),
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Train with callback (enable debug for detailed logging)
            callback = DistanceHeuristicCallback(check_freq=1000, debug=True)
            self.model.learn(total_timesteps=max_training_steps, callback=callback)

        print(f"\nDistance heuristic training complete using {algorithm}!")

    def estimate_distance(self, source_state: ObsType, target_state: ObsType) -> float:
        """Estimate distance from source to target state.

        Args:
            source_state: Starting state
            target_state: Goal state (will be converted to atoms)

        Returns:
            Estimated distance (number of steps)
        """
        assert self.model is not None, "Model must be trained before estimation"
        assert self.perceiver is not None, "Perceiver must be set before estimation"
        assert self.wrapper is not None, "Wrapper must be set before estimation"

        # Flatten source observation
        # NOTE: Custom DQN trains on unnormalized observations, so don't normalize here!
        if self.algorithm_used == "custom_dqn":
            # Flatten without normalization for custom DQN
            if hasattr(source_state, "nodes"):
                source_flat = source_state.nodes.flatten().astype(np.float32)
            else:
                source_flat = np.array(source_state).flatten().astype(np.float32)
        else:
            # Stable-baselines3 may expect normalized observations
            source_flat = self._flatten_and_normalize(source_state)

        # Convert target state to atoms, then to goal vector
        target_atoms = self.perceiver.step(target_state)
        target_goal_vector = self.wrapper.create_atom_vector(target_atoms)

        # Convert source state to atoms for achieved goal
        source_atoms = self.perceiver.step(source_state)
        achieved_goal_vector = self.wrapper.create_atom_vector(source_atoms)

        # Create goal-conditioned observation
        obs = {
            "observation": source_flat,
            "achieved_goal": achieved_goal_vector,
            "desired_goal": target_goal_vector,
        }

        # Get value estimate from Q-network (Q(s, a, g) ≈ -distance)
        if self.algorithm_used == "custom_dqn":
            # Custom DQN: Use get_q_value method
            q_value = self.model.get_q_value(
                source_flat,
                achieved_goal_vector,
                target_goal_vector,
            )
        else:
            # Stable-baselines3 implementation
            with torch.no_grad():
                obs_tensor = {
                    k: torch.as_tensor(v).unsqueeze(0).to(self.device)
                    for k, v in obs.items()
                }

                if self.algorithm_used == "SAC":
                    # SAC: Get action from policy, then Q-value from critics
                    action, _ = self.model.predict(obs, deterministic=True)
                    action_tensor = torch.as_tensor(action).unsqueeze(0).to(self.device)

                    # Get Q-values from both critics
                    q1_value = self.model.critic(obs_tensor, action_tensor)[0]
                    q2_value = self.model.critic(obs_tensor, action_tensor)[1]

                    # Use minimum Q-value (standard in SAC)
                    q_value = torch.min(q1_value, q2_value).item()

                elif self.algorithm_used == "DQN":
                    # DQN: Get Q-values for all actions, use max (greedy action)
                    q_values = self.model.q_net(obs_tensor)

                    # Take max Q-value (value of best action)
                    q_value = torch.max(q_values).item()

                    # DEBUG: Print first few Q-values to diagnose
                    if not hasattr(self, '_debug_printed'):
                        print(f"\n[DEBUG] Sample Q-values: {q_values.cpu().numpy()}")
                        print(f"[DEBUG] Max Q-value: {q_value}")
                        print(f"[DEBUG] Estimated distance: {-q_value}")
                        self._debug_printed = True

                else:
                    raise ValueError(f"Unknown algorithm: {self.algorithm_used}")

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
        assert self.model is not None, "Model must be trained before saving"
        assert self.wrapper is not None, "Wrapper must be available before saving"

        os.makedirs(path, exist_ok=True)

        # Save model
        self.model.save(f"{path}/distance_model")

        # Save algorithm type
        with open(f"{path}/algorithm.pkl", "wb") as f:
            pickle.dump({"algorithm": self.algorithm_used}, f)

        # Save observation statistics
        with open(f"{path}/obs_stats.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_mean": self._obs_mean,
                    "obs_std": self._obs_std,
                },
                f,
            )

        # Save atom indexing from wrapper
        with open(f"{path}/atom_to_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "atom_to_index": self.wrapper.atom_to_index,
                    "_next_index": self.wrapper._next_index,
                    "max_atom_size": self.wrapper.max_atom_size,
                },
                f,
            )

        # Save observation/action spaces (only for stable-baselines3)
        if self.algorithm_used != "custom_dqn":
            with open(f"{path}/observation_space.pkl", "wb") as f:
                pickle.dump(self.model.observation_space, f)
            with open(f"{path}/action_space.pkl", "wb") as f:
                pickle.dump(self.model.action_space, f)

        print(f"Distance heuristic ({self.algorithm_used}) saved to {path}")

    def load(self, path: str, perceiver: Any) -> None:
        """Load a pre-trained distance heuristic.

        Args:
            path: Directory path containing saved model
            perceiver: Perceiver to convert states to atoms
        """
        # Store perceiver
        self.perceiver = perceiver

        # Load algorithm type
        with open(f"{path}/algorithm.pkl", "rb") as f:
            algo_data = pickle.load(f)
            self.algorithm_used = algo_data["algorithm"]

        # Load observation statistics
        with open(f"{path}/obs_stats.pkl", "rb") as f:
            stats = pickle.load(f)
            self._obs_mean = stats["obs_mean"]
            self._obs_std = stats["obs_std"]

        # Load atom indexing
        with open(f"{path}/atom_to_index.pkl", "rb") as f:
            atom_data = pickle.load(f)

        # Handle loading based on algorithm type
        if self.algorithm_used == "custom_dqn":
            from tamp_improv.approaches.improvisational.custom_dqn import (
                CustomDQN,
                DQNConfig,
            )

            # Create minimal wrapper to hold atom indexing
            class MinimalEnv(gym.Env):
                def __init__(self):
                    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
                    self.action_space = gym.spaces.Discrete(5)

            minimal_env = MinimalEnv()

            # Create wrapper with empty state pairs (we just need it for atom indexing)
            self.wrapper = DistanceHeuristicWrapper(
                minimal_env,
                [],  # Empty state pairs
                perceiver,
                max_atom_size=atom_data["max_atom_size"],
            )

            # Restore atom indexing
            self.wrapper.atom_to_index = atom_data["atom_to_index"]
            self.wrapper._next_index = atom_data["_next_index"]

            # Load custom DQN model
            # We need to infer dimensions from atom_to_index
            obs_dim = 10  # Placeholder - will be overwritten by loaded model
            goal_dim = atom_data["max_atom_size"]
            num_actions = 5  # Placeholder - will be overwritten by loaded model

            # Create config (values don't matter since we're loading)
            custom_config = DQNConfig(device=self.device)

            # Create custom DQN instance
            self.model = CustomDQN(
                obs_dim=obs_dim,
                goal_dim=goal_dim,
                num_actions=num_actions,
                config=custom_config,
            )

            # Load the saved weights
            self.model.load(f"{path}/distance_model")

        else:
            # Stable-baselines3 loading
            # Create a dummy wrapper to hold the atom indexing
            class MinimalEnv(gym.Env):
                def __init__(self, obs_space, act_space):
                    self.observation_space = obs_space
                    self.action_space = act_space

            with open(f"{path}/observation_space.pkl", "rb") as f:
                observation_space = pickle.load(f)
            with open(f"{path}/action_space.pkl", "rb") as f:
                action_space = pickle.load(f)

            minimal_env = MinimalEnv(observation_space["observation"], action_space)

            # Create wrapper with empty state pairs (we just need it for atom indexing)
            self.wrapper = DistanceHeuristicWrapper(
                minimal_env,
                [],  # Empty state pairs
                perceiver,
                max_atom_size=atom_data["max_atom_size"],
            )

            # Restore atom indexing
            self.wrapper.atom_to_index = atom_data["atom_to_index"]
            self.wrapper._next_index = atom_data["_next_index"]

            # Load spaces again for model loading
            with open(f"{path}/observation_space.pkl", "rb") as f:
                observation_space = pickle.load(f)
            with open(f"{path}/action_space.pkl", "rb") as f:
                action_space = pickle.load(f)

            # Create dummy env for loading
            class DummyEnv(gym.Env):  # pylint: disable=abstract-method
                """Dummy environment for loading model."""

                def __init__(self, obs_space, act_space):
                    self.observation_space = obs_space
                    self.action_space = act_space

                def compute_reward(self, achieved_goal, _desired_goal, _info, _indices=None):
                    if isinstance(achieved_goal, np.ndarray):
                        return np.full(achieved_goal.shape[0], -1.0, dtype=np.float32)
                    return -1.0

                def reset(self, **_kwargs):
                    obs = {}
                    for key, space in self.observation_space.spaces.items():
                        obs[key] = np.zeros(space.shape, dtype=space.dtype)
                    return obs, {}

                def step(self, action):
                    obs, _ = self.reset()
                    return obs, -1.0, False, False, {}

            dummy_env = DummyEnv(observation_space, action_space)  # type: ignore

            # Load model based on algorithm type
            if self.algorithm_used == "SAC":
                self.model = SAC.load(f"{path}/distance_model", env=dummy_env, device=self.device)
            elif self.algorithm_used == "DQN":
                self.model = DQN.load(f"{path}/distance_model", env=dummy_env, device=self.device)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm_used}")

        print(f"Distance heuristic ({self.algorithm_used}) loaded from {path}")

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
