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

from relational_structs import GroundAtom


from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper
import time
import os
import pickle
import random
import wandb
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar, Callable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
import gymnasium as gym


from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class DQNHeuristicConfig:
    """Configuration for distance heuristic training."""
    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000

    # Training parameters
    max_episode_steps: int = 100
    num_steps_per_round: int = 10000
    num_rounds: int = 1

    threshold: float = 0.05
    beta: float = 1.0
    gamma: float = 0.99

    exploration_factor: float = 0.05
    keep_fraction: float = 0.2

    # Device settings
    device: str = "cuda"

class DQNHeuristicCallback(BaseCallback):
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
            # print("New obs:", self.locals.get("new_obs", [{}])['desired_goal'])
            obs = self.locals.get("new_obs", [{}])
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


class DQNHeuristicWrapper(gym.Wrapper):
    """Wrapper that provides -1 reward per step for distance learning.

    This wrapper modifies the reward structure to be -1 per step, regardless
    of whether the goal is reached. This causes the learned value function
    V(s, z') to approximate -distance(s, z'), where s is the low-level state
    and z' is the abstract goal state (atoms).
    """

    def __init__(
        self,
        env: gym.Env,
        state_node_pairs: list[tuple[ObsType, int]],
        node_atoms:  dict[int, set[GroundAtom]],
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
        self.state_node_pairs = state_node_pairs
        self.node_atoms = node_atoms
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
        if len(state_node_pairs) > 0:
            sample_state = state_node_pairs[0][0]

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
        pair_idx = np.random.randint(len(self.state_node_pairs))
        source_state, goal_node = self.state_node_pairs[pair_idx]

        # Convert goal state to atoms (abstract representation)
        self.current_goal_atoms = self.node_atoms[goal_node]

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
    
    def reconfigure(self, state_node_pairs: list[tuple[ObsType, int]]):
        self.state_node_pairs = state_node_pairs



class DQNHeuristic(BaseHeuristic):
    """Learns distance function f(s, s') via goal-conditioned RL.

    Trains a policy with reward=-1 per step, causing the value function to
    learn V(s, g) ≈ -distance(s, g). The heuristic can then estimate distances
    by querying the critic/value network.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        config: DQNHeuristicConfig,
        seed: int | None = None,
    ):
        """Initialize distance heuristic.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        self.system = system
        self.training_data = training_data
        self.graph_distances = graph_distances

        if self.config.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.model: SAC | DQN | None = None
        self.algorithm_used: str | None = None  # Track which algorithm was used
        self.perceiver: Any | None = None
        self.wrapper: DQNHeuristicWrapper | None = None
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

        # Extract state-node pairs (state from node A, target node B)
        self.atoms_to_node = {}
        planning_graph = training_data.graph
        for node in planning_graph.nodes:
            self.atoms_to_node[node.atoms] = node.id

        state_node_pairs = []
        for node_id, states in self.training_data.node_states.items():
            # Find node by atoms

            for state in states:

                # Add pairs to all other nodes
                for target_node in planning_graph.nodes:
                    if target_node.id != node_id:
                        state_node_pairs.append((state, target_node.id))

        self.state_node_pairs = state_node_pairs

        # Create get_node function
        def get_node(state: ObsType) -> int:
            """Convert state to node ID."""
            atoms = self.system.perceiver.step(state)
            # print(atoms)
            # print(self.atoms_to_node)
            if frozenset(atoms) not in self.atoms_to_node:
                return -1
            return self.atoms_to_node[frozenset(atoms)]
        
        self.get_node = get_node


        # Wrap environment for distance learning (converts goal states to atoms internally)
        wrapped_env = DQNHeuristicWrapper(
            self.system.env,
            state_node_pairs,
            self.training_data.node_atoms,
            self.system.perceiver,
            max_episode_steps=self.config.max_episode_steps,
        )

        # Store wrapper so we can use its atom_to_index mapping
        self.wrapper = wrapped_env

        # Compute observation statistics for normalization
        # self._compute_obs_statistics(state_node_pairs)

        
        action_space = wrapped_env.action_space
        if isinstance(action_space, gym.spaces.Box):
            algorithm = "SAC"
        elif isinstance(action_space, gym.spaces.Discrete):
            algorithm = "DQN"
        else:
            raise ValueError(
                f"Unsupported action space type: {type(action_space)}. "
                "Only Box (continuous) and Discrete are supported."
            )
        
        print(f"Using algorithm: {algorithm} for action space: {type(action_space).__name__}")

        # Create model based on algorithm
        if algorithm == "SAC":
            self.model = SAC(
                "MultiInputPolicy",
                wrapped_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                device=self.device,
                seed=self.seed,
                verbose=1,
                gamma=self.config.gamma,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,  # Number of virtual transitions per real transition
                    goal_selection_strategy="final",  # Use future states as goals
                ),
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
                device=self.device,
                seed=self.seed,
                verbose=1,
                gamma=self.config.gamma,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,  # Number of virtual transitions per real transition
                    goal_selection_strategy="final",  # Use future states as goals
                ),
            )

            self.algorithm_used = algorithm
            # Train with callback (enable debug for detailed logging)
        self.callback = DQNHeuristicCallback(check_freq=1000, debug=True)

    def train(
        self,
        state_node_pairs: list[tuple[ObsType, int]],
        num_steps: int
    ) -> None:
        """Train distance heuristic on sampled state pairs.

        Args:
            env: Base environment
            state_pairs: List of (source, target) state pairs
            perceiver: Perceiver to convert states to atoms
            max_training_steps: Total training timesteps
        """
        print(f"\nTraining distance heuristic on {len(state_node_pairs)} state pairs...")
        print(f"Device: {self.device}")

        # DEBUG: Show statistics about training pairs
        print(f"\n[DEBUG TRAINING PAIRS]")
        print(f"  Total pairs: {len(state_node_pairs)}")
        # Store perceiver for use in estimate_distance

        self.wrapper.reconfigure(state_node_pairs)
        self.model.set_env(self.wrapper)

        self.model.learn(total_timesteps=num_steps, callback=self.callback)

        
    
    def multi_train(
        self,
    ) -> dict:
        """Multi-round training that iteratively prunes to promising shortcuts.

        For each round:
        1. Train on current subset of state-node pairs
        2. Compute estimated vs graph distances at NODE-NODE level
        3. Keep top keep_fraction of NODE-NODE pairs with best differences
        4. Keep ALL state-node pairs for the selected node-node pairs
        5. Continue training on this pruned subset

        This focuses learning on node-node pairs that show promise for shortcuts.

        Args:
            state_node_pairs: List of (source_state, target_node) pairs
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            num_rounds: Number of pruning rounds
            num_epochs_per_round: Training epochs per round
            trajectories_per_epoch: Trajectories collected per epoch
            max_episode_steps: Max steps per trajectory
            keep_fraction: Fraction of NODE-NODE pairs to keep each round (0.5 = keep top 50%)

        Returns:
            Dictionary with combined training history across all rounds
        """
        state_node_pairs = self.state_node_pairs
        graph_distances = self.graph_distances

        num_rounds = self.config.num_rounds
        num_steps_per_round = self.config.num_steps_per_round

        keep_fraction = self.config.keep_fraction

        # wandb.init(project="slap_crl_heuristic", config=self.config.__dict__)

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING: {num_rounds} rounds, {num_steps_per_round} steps/round")
        print(f"Starting with {len(state_node_pairs)} state-node pairs")
        print(f"Keep fraction: {keep_fraction} (pruning {1-keep_fraction:.1%} each round)")
        print(f"{'='*80}\n")

        # Build mapping: (source_node, target_node) -> list of (source_state, target_node) pairs
        print("Building node-node to state-node mapping...")
        node_pair_to_state_pairs: dict[tuple[int, int], list[tuple[ObsType, int]]] = {}
        for state, target_node in state_node_pairs:
            source_node = self.get_node(state)
            key = (source_node, target_node)
            if key not in node_pair_to_state_pairs:
                node_pair_to_state_pairs[key] = []
            node_pair_to_state_pairs[key].append((state, target_node))

        print(f"Found {len(node_pair_to_state_pairs)} unique node-node pairs")
        print(f"Average {len(state_node_pairs) / len(node_pair_to_state_pairs):.1f} state-node pairs per node-node pair")

        # Combined history across all rounds
        combined_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'num_state_pairs_per_round': [],  # Track state-node dataset size per round
            'num_node_pairs_per_round': [],  # Track node-node dataset size per round
            'distance_matrices': [],  # Distance matrix after each round (for debugging)
            'graph_distances': graph_distances,  # Store graph distances for reference
        }

        # Start with all node-node pairs
        current_node_pairs = list(node_pair_to_state_pairs.keys())


        for round_idx in range(num_rounds):
            # Get all state-node pairs for current node-node pairs
            current_state_pairs = []
            for node_pair in current_node_pairs:
                current_state_pairs.extend(node_pair_to_state_pairs[node_pair])

            print(f"\n{'='*80}")
            print(f"ROUND {round_idx + 1}/{num_rounds}")
            print(f"Training on {len(current_node_pairs)} node-node pairs")
            print(f"  -> {len(current_state_pairs)} state-node pairs")
            print(f"{'='*80}\n")

            # Mark start of this round in history
            combined_history['num_state_pairs_per_round'].append(len(current_state_pairs))
            combined_history['num_node_pairs_per_round'].append(len(current_node_pairs))

            # Train on current subset
            self.train(
                state_node_pairs=current_state_pairs,
                num_steps=num_steps_per_round
            )

            # Compute distance matrix for ALL node-node pairs (not just current ones)
            # This allows us to "bring back" previously pruned pairs if they become promising
            print(f"\n[DEBUG] Computing distance matrix for ALL {len(node_pair_to_state_pairs)} node pairs after round {round_idx + 1}...")
            all_node_pairs = list(node_pair_to_state_pairs.keys())
            distance_matrix = self._compute_distance_matrix(all_node_pairs)
            combined_history['distance_matrices'].append({
                'round': round_idx + 1,
                'distances': distance_matrix,
                'active_node_pairs': current_node_pairs.copy(),  # Which pairs we trained on
            })
            print(f"[DEBUG] Distance matrix computed with {len(distance_matrix)} entries")

            # If not the last round, prune to most promising NODE-NODE pairs
            # IMPORTANT: Evaluate ALL node pairs, not just current ones!
            if round_idx < num_rounds - 1:
                print(f"\n{'='*80}")
                print(f"PRUNING after round {round_idx + 1}")
                print(f"{'='*80}\n")

                # Call prune() to get selected node pairs
                new_data = self.prune_explore()
                selected_node_pairs_set = new_data.unique_shortcuts
                current_node_pairs = list(selected_node_pairs_set)

                # Count resulting state-node pairs
                resulting_state_pairs = sum(
                    len(node_pair_to_state_pairs.get((src, tgt), []))
                    for src, tgt in current_node_pairs
                )

                print(f"Pruned node-node pairs: {len(all_node_pairs)} -> {len(current_node_pairs)}")
                print(f"Resulting state-node pairs: {resulting_state_pairs}")

        # Final state-node pairs
        final_state_pairs = []
        for node_pair in current_node_pairs:
            final_state_pairs.extend(node_pair_to_state_pairs[node_pair])

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING COMPLETE")
        print(f"Final node-node pairs: {len(current_node_pairs)}")
        print(f"Final state-node pairs: {len(final_state_pairs)}")
        print(f"{'='*80}\n")

        # wandb.finish()

        combined_history['episode_lengths'] = self.callback.episode_lengths
        combined_history['episode_rewards'] = self.callback.episode_rewards

        return combined_history


    def estimate_distance(self, source_state: ObsType, target_node: int) -> float:
        """Estimate distance from source to target state.

        Args:
            source_state: Starting state
            target_state: Goal state (will be converted to atoms)

        Returns:
            Estimated distance (number of steps)
        """

        source_flat = self._flatten_state(source_state)
        
        target_atoms = self.training_data.node_atoms[target_node]

        # Convert target state to atoms, then to goal vector
        target_goal_vector = self.wrapper.create_atom_vector(target_atoms)

        # Convert source state to atoms for achieved goal
        source_atoms = self.system.perceiver.step(source_state)
        achieved_goal_vector = self.wrapper.create_atom_vector(source_atoms)

        # Create goal-conditioned observation
        obs = {
            "observation": source_flat,
            "achieved_goal": achieved_goal_vector,
            "desired_goal": target_goal_vector,
        }

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

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        source_states = self.training_data.node_states[source_node]
        avg = 0
        n = 0
        for source_state in random.sample(source_states, min(100, len(source_states))):
            avg += self.estimate_distance(source_state, target_node)
            n += 1
        return avg / n
    
    def estimate_probability(self, source_node: int, target_node: int) -> float:
        est_dist = self.estimate_node_distance(source_node, target_node)

        if est_dist <= 0:
            p_rr = 1.0
        else:
            p_rr = np.clip(self.config.beta * np.exp(-(est_dist)**2 / (2 * self.config.max_episode_steps)), 0, 1)
        

        print("Rollout probability of reaching node", target_node, "from node", source_node, ":", p_rr, "for distance", est_dist)

        k = np.log(0.5) / np.log(1 - self.config.threshold)
        return 1 - (1 - p_rr)**k

    # def estimate_gain(self, source_node: int, target_node: int) -> float:
    #     """Estimate gain of training on a shortcut, relative to distance in the
    #     initial graph. Higher gain means more useful shortcut."""

    #     graph_distance = self.graph_distances.get((source_node, target_node), float('inf'))
    #     estimated_distance = self.estimate_node_distance(source_node, target_node)
        
    #     gain = np.clip(graph_distance - estimated_distance, 0, self.config.max_episode_steps)
    #     return gain

    def estimate_gain(self, source_node: int, target_node: int) -> float:
        """Estimate gain of training on a shortcut, relative to distance in the
        initial graph. Higher gain means more useful shortcut."""

        # graph_distance = self.graph_distances.get((source_node, target_node), float('inf'))
        length = self.estimate_node_distance(source_node, target_node)
        
        # Use this if we're learning too many backwards shortcuts. Might not be a great idea in general.
        # gain = max(graph_distance - estimated_distance, 0)
        # if math.isinf(gain):
        #     gain = 0

        x = source_node
        y = target_node

        nodes = self.training_data.node_states.keys()


        def d(u, v):
            return self.graph_distances[(u, v)]

        # --- Phase 1: affected node sets ---
        A_x = [u for u in nodes if d(u, y) + length < d(u, x)]
        A_y = [v for v in nodes if d(v, x) + length < d(v, y)]

        # Iterate over smaller set (optimization)
        if len(A_y) < len(A_x):
            A_x, A_y = A_y, A_x
            x, y = y, x

        RG = 0.0

        # --- Phase 2: affected pairs ---
        for u in A_x:
            du_y = d(u, y)
            for v in A_y:
                d_old = d(u, v)
                d_new = du_y + length + d(x, v)

                if d_new < d_old:
                    RG += (d_old - d_new)

        gain = np.clip(RG, 0, self.config.max_episode_steps)
        return gain

    def prune_explore(self) -> GoalConditionedTrainingData:
        print(f"\n[DEBUG] Pruning with keep_fraction={self.config.keep_fraction}, max_episode_steps={self.config.max_episode_steps}")

        # Score all node pairs: score = estimated_distance - graph_distance
        # Negative scores = shortcuts (estimated < graph)
        # Lower scores = more promising

        score_tuples = []
        for source_id, target_id in self.training_data.unique_shortcuts:
            p = self.estimate_probability(source_id, target_id)
            g = self.estimate_gain(source_id, target_id)
            score = p * g
            score_tuples.append((source_id, target_id, score, p, g))

        # Sort score tuples first by score, and then by probability
        score_tuples.sort(key=lambda x: (x[2], x[3]), reverse=True)
        print("  Shortcut scores (source -> target: score (dist, prob, gain)):")
        for source_id, target_id, score, p, g in score_tuples:
            d = self.estimate_node_distance(source_id, target_id)
            print(f"    ({source_id} -> {target_id}): {score:.4f} ({d:.2f}, {p:.2f}, {g:.2f})")
        
        

        # # Strategy 1: Keep all negative scores (shortcuts)
        # shortcuts = set()
        # for score, source_node, target_node, est_dist in node_pair_scores:
        #     if score < 0:
        #         shortcuts.add((source_node, target_node))

        # print(f"[DEBUG] Found {len(shortcuts)} shortcuts (negative scores)")

        # Strategy 2: Keep top keep_fraction by score
        num_to_keep = max(1, int(len(score_tuples) * self.config.keep_fraction))
        top_pairs = set()
        for source_node, target_node, score, p, g in score_tuples[:num_to_keep]:
            top_pairs.add((source_node, target_node))

        print(f"[DEBUG] Keeping top {num_to_keep} pairs ({self.config.keep_fraction:.1%})")

        # Strategy 3: Add exploration pairs randomly
        num_exploration = int(len(self.training_data.unique_shortcuts) * self.config.exploration_factor)  # 5% exploration
        all_pairs = set(self.training_data.unique_shortcuts)
        unexplored_pairs = all_pairs - top_pairs
        exploration_pairs = random.sample(list(unexplored_pairs), min(num_exploration, len(unexplored_pairs)))
        for source_node, target_node in exploration_pairs:
            top_pairs.add((source_node, target_node))
        print(f"[DEBUG] Added {len(exploration_pairs)} exploration pairs")

        # Return union
        result = top_pairs
        print(f"[DEBUG] Total pairs to keep: {len(result)} (union of shortcuts and top)")


        # Combine result with self.training_data to get back a new TrainingData
        selected_set = set(result)
        selected_indices = []
        for i, (source_id, target_id) in enumerate(self.training_data.valid_shortcuts):
            if (source_id, target_id) in selected_set:
                selected_indices.append(i)
        
        print(f"  ({len(selected_indices)} state-node pairs)")
        # Filter shortcut_info to match the pruned data
        original_shortcut_info = self.training_data.config.get("shortcut_info", [])
        pruned_shortcut_info = (
            [original_shortcut_info[i] for i in selected_indices]
            if original_shortcut_info
            else []
        )


        pruned_data = GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[self.training_data.current_atoms[i] for i in selected_indices],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[self.training_data.valid_shortcuts[i] for i in selected_indices],
            unique_shortcuts=list(selected_set),  # Unique node-node pairs
            node_states=self.training_data.node_states,  # Keep all node states
            node_atoms=self.training_data.node_atoms,  # Keep all node atoms
            graph=self.training_data.graph,  # Keep planning graph
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "dqn",
            },
        )

        return pruned_data

    def prune(self, max_shortcuts: int | None, ) -> GoalConditionedTrainingData:
        """Like prune_explore, but only prunes the top max_shortcuts shortcuts from the list
        instead of keep_fraction/explore_fraction."""

        if max_shortcuts is None:
            return self.training_data

        print(f"\n[DEBUG] Pruning to max_shortcuts={max_shortcuts}")

        # Compute success rates and select shortcuts (use unique_shortcuts for node-node pairs)
        score_tuples = []
        for source_id, target_id in self.training_data.unique_shortcuts:
            p = self.estimate_probability(source_id, target_id)
            g = self.estimate_gain(source_id, target_id)
            score = p * g
            score_tuples.append((source_id, target_id, score, p, g))
        
        # Sort score tuples first by score, and then by probability
        score_tuples.sort(key=lambda x: (x[2], x[3]), reverse=True)
        print("  Shortcut scores (source -> target: score (dist, prob, gain)):")
        for source_id, target_id, score, p, g in score_tuples:
            d = self.estimate_node_distance(source_id, target_id)
            dg = self.graph_distances.get((source_id, target_id), np.inf)
            print(f"    ({source_id} -> {target_id}): {score:.4f} ({d:.2f}, {dg:.2f}, {p:.2f}, {g:.2f})")
        
        # Select top max_shortcuts shortcuts
        selected_shortcuts = score_tuples[:max_shortcuts]
        selected_unique_shortcuts = [
            (source_id, target_id) for source_id, target_id, _, _, _ in selected_shortcuts
        ]

        # Filter training data to match selected unique shortcuts
        # Keep all state-node pairs that correspond to selected node-node pairs
        selected_set = set(selected_unique_shortcuts)
        selected_indices = []

        for i, (source_id, target_id) in enumerate(self.training_data.valid_shortcuts):
            if (source_id, target_id) in selected_set:
                selected_indices.append(i)

        print(f"  ({len(selected_indices)} state-node pairs)")

        # Filter shortcut_info to match the pruned data
        original_shortcut_info = self.training_data.config.get("shortcut_info", [])
        pruned_shortcut_info = (
            [original_shortcut_info[i] for i in selected_indices]
            if original_shortcut_info
            else []
        )

        pruned_data = GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[self.training_data.current_atoms[i] for i in selected_indices],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[self.training_data.valid_shortcuts[i] for i in selected_indices],
            unique_shortcuts=selected_unique_shortcuts,  # Unique node-node pairs
            node_states=self.training_data.node_states,  # Keep all node states
            node_atoms=self.training_data.node_atoms,  # Keep all node atoms
            graph=self.training_data.graph,  # Keep planning graph
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "dqn",
                "threshold": self.config.threshold,
            },
        )

        return pruned_data
    
    def _compute_distance_matrix(self, node_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Compute estimated distances for all node pairs.

        Args:
            node_pairs: List of (source_node, target_node) pairs to compute distances for

        Returns:
            Dictionary mapping (source_node, target_node) -> estimated distance
        """
        distance_matrix = {}
        for source_node, target_node in node_pairs:
            est_dist = self.estimate_node_distance(source_node, target_node)
            distance_matrix[(source_node, target_node)] = est_dist
        return distance_matrix

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
            self.wrapper = DQNHeuristicWrapper(
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
            self.wrapper = DQNHeuristicWrapper(
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
    
    def _flatten_state(self, state: ObsType) -> np.ndarray:
        """Flatten state to array."""
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32) #[1:3]
        return np.array(state).flatten().astype(np.float32)
