"""Goal-conditioned distance heuristic V2 using SAC-based contrastive learning.

This module implements a learned distance function f(s, a, g) that estimates
the number of steps required to reach goal node g from source state s when
taking action a. Unlike V1/V3/V4 which use greedy action selection, V2 uses
proper Soft Actor-Critic (SAC) to learn both a critic and an actor jointly.

Key components:
- sa_encoder: Neural network that maps (state, action) pairs to k-dimensional embeddings
- g_encoder: Learnable matrix (num_nodes x k) where row i is the embedding of node i
- actor: Policy network π(a|s,g) that outputs actions conditioned on state and goal
- Contrastive critic loss: Aligns (state, action, future_node) tuples
- Actor loss: Maximizes alignment between sa_encoder(s, π(s,g)) and g_encoder(g)
- Normalized embeddings: All embeddings are L2-normalized to unit length

Based on Eysenbach et al. "Contrastive Learning as Goal-Conditioned RL" (2021)
"""

import math
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from numpy.typing import NDArray

from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
from tamp_improv.approaches.improvisational.heuristics.networks import (  # noqa: E402
    ContinuousActor,
    DiscreteActor,
    GoalEncoder,
    ResidualContinuousActor,
    StateActionEncoder,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)

from task_then_motion_planning.planning import TaskThenMotionPlanningFailure

from relational_structs import GroundAtom, GroundOperator
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class CRLV2HeuristicConfig:
    """Configuration for SAC-based contrastive state-action-node distance heuristic."""

    wandb_enabled: bool = False  # Whether to enable Weights & Biases logging

    # Pruning
    threshold: float = 0.05
    beta: float = 1  # Complex distance scaling parameter

    # Network architecture
    latent_dim: int = 32  # Dimension of embedding space (k)
    hidden_dims: list[int] | None = None  # Hidden layer sizes [64, 64]
    normalize_embeddings: bool = False  # Whether to L2-normalize embeddings to unit sphere

    # L2 regularization via dual Lagrangian (replaces normalization)
    c: float = 1.0  # Target for E[||psi||²]; constrains embedding scale

    # Actor architecture (separate from critic)
    actor_hidden_dims: list[int] | None = None  # Actor hidden layers [64, 64]
    goal_encoder_hidden_dims: list[int] | None = None  # Goal encoder hidden layers [32]

    # Training parameters
    critic_lr: float = 0.001  # Critic learning rate
    actor_lr: float = 0.0003  # Actor learning rate (typically lower than critic)
    batch_size: int = 256
    buffer_size: int = 10000
    grad_clip: float = 1.0  # Gradient clipping threshold
    sampling_method: str = "uniform"  # "uniform" or "ucb"
    ucb_beta: float = 1.0  # UCB exploration parameter
    gain_method: str = "exact"
    dist_scale: float = 1.0  # Scaling factor for distance when computing gains
    dist_quantile: float = 1.0  # Quantile for distance scaling when computing gains
    auto_dist_scale: bool = False  # Whether to automatically set dist_scale based on quantile
    residualize: bool = False  # Whether to residualize actor actions with base skill actions

    # Contrastive learning
    gamma: float = 0.99  # For geometric sampling of future states
    repetition_factor: int = 4  # CRTR: repeat each trajectory this many times in batch

    # Training
    iters_per_epoch: int = 1  # Gradient steps per training call
    learn_frequency: int = 10  # Learn every N epochs
    gain_update_frequency: int = 10  # Update UCB gains every N epochs
    node_distance_samples: int = 5  # States to average over in estimate_node_distance
    num_epochs_per_round: int = 200
    trajectories_per_epoch: int = 10
    max_episode_steps: int = 100

    # Atom encoding
    max_atom_size: int = 50  # Maximum number of unique atoms for multi-hot encoding

    # Success tracking
    num_reliability_trials: int = 10  # Window size for success/failure deque per node pair

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


class ContrastiveReplayBuffer:
    """Replay buffer for storing complete trajectories.

    Each trajectory is a sequence of (obs, action, node_id) tuples where
    obs is the raw environment observation (needed for skill.get_action in residual
    mode) and node_id is the planning node reached at that timestep.
    state_flat is derived from obs on demand via _flatten_state.
    """

    def __init__(self, max_size: int, gamma: float = 0.99):
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of trajectories to store
            gamma: Discount factor for geometric sampling of future states
        """
        self.max_size = max_size
        self.gamma = gamma
        self.trajectories: deque[list[tuple[Any, NDArray, int]]] = deque(
            maxlen=max_size
        )

    def add_trajectory(
        self, trajectory: list[tuple[Any, NDArray, int]]
    ) -> None:
        """Add a complete trajectory to the buffer.

        Args:
            trajectory: List of (obs, action, node_id) tuples
        """
        if len(trajectory) > 0:
            self.trajectories.append(trajectory)

    def sample_batch_crtr(
        self, batch_size: int, repetition_factor: int = 4
    ) -> tuple[list[Any], NDArray, NDArray, NDArray]:
        """Sample batch using CRTR (Contrastive Random Trajectory Repetition).

        For each trajectory sampled:
        1. Sample t uniformly from [0, len(traj)-1]
        2. Sample t' geometrically after t using gamma
        3. Return (obs[t], action[t], node[t'], node[t]) as a positive pair
        4. Repeat each trajectory `repetition_factor` times

        Args:
            batch_size: Number of unique trajectories to sample
            repetition_factor: Number of times to repeat each trajectory

        Returns:
            obs_list: raw observations at t (list, length batch_size * repetition_factor)
            actions: (batch_size * repetition_factor, action_dim)
            future_nodes: (batch_size * repetition_factor,) - node IDs at t'
            current_nodes: (batch_size * repetition_factor,) - node IDs at t
        """
        if len(self.trajectories) == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample trajectories with replacement
        sampled_trajs = random.choices(self.trajectories, k=batch_size)

        obs_list: list[Any] = []
        actions_list = []
        future_nodes_list = []
        current_nodes_list = []

        for traj in sampled_trajs:
            # Repeat this trajectory `repetition_factor` times
            for _ in range(repetition_factor):
                if len(traj) == 1:
                    # Special case: trajectory has only one step
                    obs, action, node = traj[0]
                    obs_list.append(obs)
                    actions_list.append(action)
                    future_nodes_list.append(node)
                    current_nodes_list.append(node)
                else:
                    # Sample current timestep t uniformly
                    t = random.randint(0, len(traj) - 1)

                    # Sample future timestep t' geometrically using gamma
                    # Probability of sampling k steps ahead: (1-gamma) * gamma^k
                    max_k = len(traj) - t - 1  # Maximum steps we can go ahead
                    if max_k == 0:
                        # Already at last step, future = current
                        t_future = t
                    else:
                        # Geometric distribution
                        probs = [
                            (1 - self.gamma) * (self.gamma**k) for k in range(max_k)
                        ]
                        # Renormalize if needed
                        probs_sum = sum(probs)
                        if probs_sum > 0:
                            probs = [p / probs_sum for p in probs]
                            k = random.choices(range(max_k), weights=probs)[0]
                            t_future = t + k + 1
                        else:
                            t_future = t + 1

                    obs_t, action_t, node_t = traj[t]
                    _, _, node_future = traj[t_future]

                    obs_list.append(obs_t)
                    actions_list.append(action_t)
                    future_nodes_list.append(node_future)
                    current_nodes_list.append(node_t)

        return (
            obs_list,
            np.array(actions_list),
            np.array(future_nodes_list),
            np.array(current_nodes_list),
        )

    def __len__(self) -> int:
        return len(self.trajectories)


class CRLv2Heuristic(BaseHeuristic):
    """SAC-based contrastive learning heuristic for goal-conditioned RL.

    This heuristic learns:
    1. A critic that estimates distances via contrastive learning on (state, action, goal) tuples
    2. An actor that outputs actions to minimize distance to goal

    The critic and actor are trained jointly, similar to Soft Actor-Critic.
    """

    def __init__(
        self,
        training_data: GoalConditionedTrainingData,
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
        config: CRLV2HeuristicConfig | None = None,
        first_edge_dict: dict[tuple[frozenset[GroundAtom], frozenset[GroundAtom]], PlanningGraphEdge] | None = None,
        **kwargs: Any,
    ):
        """Initialize CRL V2 heuristic.

        Args:
            training_data: Training data with states, nodes, and graph
            graph_distances: Precomputed shortest path distances between nodes
            system: TAMP system for environment interaction
            rng: Random number generator
            config: Configuration object
            first_edge_dict: Mapping from (source_atoms, target_atoms) to first edge on path
            **kwargs: Additional config parameters (override config)
        """
        super().__init__(training_data, graph_distances)

        self.system = system
        self.rng = rng
        self.first_edge_dict = first_edge_dict
        self.internal_graph = training_data.graph  # Keep an internal copy of the graph for edge lookups (updated in multi-round training)

        # Initialize config
        if config is None:
            config = CRLV2HeuristicConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Determine state dimension from actual state (handles graph observations)
        sample_states = next(iter(training_data.node_states.values()))
        if len(sample_states) > 0:
            sample_state = sample_states[0]
            state_flat = self._flatten_state(sample_state)
            self.state_dim = state_flat.shape[0]
        else:
            # Fallback to observation space if no sample states available
            env = system.env
            if isinstance(env.observation_space, gym.spaces.Box):
                self.state_dim = int(np.prod(env.observation_space.shape))
            else:
                raise ValueError(
                    f"Unsupported observation space: {type(env.observation_space)}"
                )

        # Determine action space type
        env = system.env
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space_type = "discrete"
            self.action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_space_type = "continuous"
            self.action_dim = int(np.prod(env.action_space.shape))
        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")

        # Number of nodes
        self.num_nodes = len(training_data.node_states)

        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_gains = np.zeros((self.num_nodes, self.num_nodes))
        self.original_node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in graph_distances.items():
            self.original_node_pair_graph_dists[i, j] = dist
            self.node_pair_graph_dists[i, j] = dist

        self.valid_pairs = list(training_data.unique_shortcuts)
        self.valid_pairs_mask = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        for (i, j) in self.valid_pairs:
            self.valid_pairs_mask[i, j] = True

        self.node_pair_successes = {}
        for (i, j) in self.valid_pairs:
            # Buffer of size k for each node pair to track recent success/failure outcomes
            k = self.config.num_reliability_trials
            if k > 0:
                self.node_pair_successes[(i, j)] = deque(maxlen=k)
            else:
                self.node_pair_successes[(i, j)] = []


        print(f"State dimension: {self.state_dim}")
        print(f"Action space: {self.action_space_type}")
        print(f"Action dimension: {self.action_dim}")

        # Initialize networks
        self._init_networks()

        # Initialize replay buffer
        self.replay_buffer = ContrastiveReplayBuffer(
            max_size=self.config.buffer_size, gamma=self.config.gamma
        )

        # Lagrange multiplier for L2 constraint E[||psi||²] ≈ c
        self.log_lambda = torch.zeros(1, device=torch.device(self.config.device), requires_grad=True)

        # Initialize optimizers
        self.critic_optimizer = torch.optim.Adam(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()) + [self.log_lambda],
            lr=self.config.critic_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

        # Training statistics
        self.training_step = 0

        # Pre-compute node atoms for fast lookup
        self._node_atoms_dict = dict(training_data.node_atoms)
        # Reverse map: frozenset(atoms) -> node_id for O(1) lookup
        self._atoms_to_node_id: dict[frozenset, int] = {
            frozenset(atoms): node_id
            for node_id, atoms in self._node_atoms_dict.items()
        }

        # Dynamic atom indexing (like DQNHeuristicWrapper)
        self.atom_to_index: dict[str, int] = {}
        self._next_atom_index = 0

        # Pre-build atom index for all known atoms
        for node_id, atoms in self._node_atoms_dict.items():
            for atom in atoms:
                self._get_atom_index(str(atom))

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom (dynamic indexing).

        Args:
            atom_str: String representation of atom

        Returns:
            Unique index for this atom
        """
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert (
            self._next_atom_index < self.config.max_atom_size
        ), f"No more space for new atom at index {self._next_atom_index}. Increase max_atom_size (currently {self.config.max_atom_size})."
        idx = self._next_atom_index
        self.atom_to_index[atom_str] = idx
        self._next_atom_index += 1
        return idx

    def create_atom_vector(self, atoms: set) -> np.ndarray:
        """Create a multi-hot vector representation of the set of atoms.

        Args:
            atoms: Set of atoms

        Returns:
            Multi-hot vector (max_atom_size,)
        """
        vector = np.zeros(self.config.max_atom_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector

    def _flatten_state(self, state: "ObsType") -> np.ndarray:
        """Flatten state to array (handles graph observations).

        Args:
            state: State observation

        Returns:
            Flattened state as float32 array
        """
        if hasattr(state, "nodes"):
            # Graph observation - flatten the nodes attribute
            return state.nodes.flatten().astype(np.float32)
        return np.array(state).flatten().astype(np.float32)

    def update_system(
        self,
        new_system: "ImprovisationalTAMPSystem",
        new_graph: "PlanningGraph",  # noqa: F841 (reserved for future use)
        new_graph_distances: dict[tuple[int, int], float],
        new_first_edge_dict: dict[tuple[int, int], PlanningGraphEdge],
        reset_actor: bool = False,
    ) -> None:
        """Update the TAMP system (for multi-round training)."""
        print("Updating system for new round of training...")

        print("Change in every graph distance:")
        for (i, j), dist in new_graph_distances.items():
            old_dist = self.graph_distances.get((i, j), np.inf)
            print(f"  Node {i} -> Node {j}: {old_dist} -> {dist}")

        self.system = new_system
        self.internal_graph = new_graph
        self.graph_distances = new_graph_distances
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in new_graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist
        self.first_edge_dict = new_first_edge_dict

        # Reset actor network for new round
        print("Resetting actor network for new round...")
        device = torch.device(self.config.device)
        if self.action_space_type == "discrete":
            self.actor = DiscreteActor(
                state_dim=self.state_dim,
                num_actions=self.action_dim,
                atom_dim=self.config.max_atom_size,
                hidden_dims=self.config.actor_hidden_dims,
            ).to(device)
        else:
            self.actor = ContinuousActor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                atom_dim=self.config.max_atom_size,
                hidden_dims=self.config.actor_hidden_dims,
            ).to(device)

    def _init_networks(self) -> None:
        """Initialize neural networks."""
        device = torch.device(self.config.device)

        # State-action encoder (critic)
        self.sa_encoder = StateActionEncoder(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(device)

        # Goal encoder (neural network that encodes atom vectors)
        self.g_encoder = GoalEncoder(
            atom_dim=self.config.max_atom_size,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.goal_encoder_hidden_dims,
        ).to(device)

        # Actor network
        if self.action_space_type == "discrete":
            self.actor = DiscreteActor(
                state_dim=self.state_dim,
                num_actions=self.action_dim,
                atom_dim=self.config.max_atom_size,
                hidden_dims=self.config.actor_hidden_dims,
            ).to(device)
        elif self.config.residualize:
            self.actor = ResidualContinuousActor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                atom_dim=self.config.max_atom_size,
                hidden_dims=self.config.actor_hidden_dims,
            ).to(device)
        else:
            self.actor = ContinuousActor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                atom_dim=self.config.max_atom_size,
                hidden_dims=self.config.actor_hidden_dims,
            ).to(device)

    # def multi_train(self, **kwargs: Any) -> dict[str, Any]:
    #     """Train critic and actor jointly using contrastive learning.

    #     Returns:
    #         Dictionary with training statistics
    #     """
    #     print(f"\n{'='*80}")
    #     print(f"Training CRL V2 Heuristic (SAC-based)")
    #     print(f"{'='*80}")
    #     print(f"State dim: {self.state_dim}")
    #     print(f"Action space: {self.action_space_type}")
    #     print(f"Action dim: {self.action_dim}")
    #     print(f"Num nodes: {self.num_nodes}")
    #     print(f"Latent dim: {self.config.latent_dim}")
    #     print(f"Critic LR: {self.config.critic_lr}")
    #     print(f"Actor LR: {self.config.actor_lr}")
    #     print(f"Batch size: {self.config.batch_size}")
    #     print(f"Device: {self.config.device}")
    #     print(f"{'='*80}\n")

    #     training_history = {
    #         "critic_losses": [],
    #         "actor_losses": [],
    #         "rounds": [],
    #     }

    #     round_history = self._train_one_round()
    #     training_history["rounds"].append(round_history)
    #     training_history["critic_losses"].extend(round_history["critic_losses"])
    #     training_history["actor_losses"].extend(round_history["actor_losses"])

    #     return training_history

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Train the heuristic on shortcut data.

        For non-learning methods (e.g., rollouts), this may do nothing.

        Args:
            **kwargs: Additional training parameters (epochs, rounds, etc.)

        Returns:
            Dictionary with training history/metadata
        """
        return {
            "critic_losses": 0,
            "actor_losses": 0,
            "buffer_size": 0,
        }

    def train_one_round(self) -> dict[str, Any]:
        """Train for one round (multiple epochs with rollouts).

        Returns:
            Dictionary with round statistics
        """
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_successes = {}
        for (i, j) in self.valid_pairs:
            # Buffer of size k for each node pair to track recent success/failure outcomes
            k = self.config.num_reliability_trials
            if k > 0:
                self.node_pair_successes[(i, j)] = deque(maxlen=k)
            else:
                self.node_pair_successes[(i, j)] = []


        critic_losses = []
        actor_losses = []

        self._update_gains()

        for epoch in range(self.config.num_epochs_per_round):
            # Collect trajectories
            for _ in range(self.config.trajectories_per_epoch):
                trajectory = self._collect_trajectory()
                self.replay_buffer.add_trajectory(trajectory)

            # Update networks
            if (epoch + 1) % self.config.learn_frequency == 0:
                if len(self.replay_buffer) >= self.config.batch_size:
                    for _ in range(self.config.iters_per_epoch):
                        critic_loss, actor_loss = self._update_networks()
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

            if self.config.sampling_method != "uniform" and (epoch + 1) % self.config.gain_update_frequency == 0:
                self._update_gains()

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                critic_loss_str = f"{critic_losses[-1]:.4f}" if critic_losses else "N/A"
                actor_loss_str = f"{actor_losses[-1]:.4f}" if actor_losses else "N/A"
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs_per_round} | "
                    f"Buffer size: {len(self.replay_buffer)} | "
                    f"Critic loss: {critic_loss_str} | "
                    f"Actor loss: {actor_loss_str}"
                )

        self._update_gains()

        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "buffer_size": len(self.replay_buffer),
        }

    def _collect_trajectory(self) -> list[tuple[Any, NDArray, int]]:
        """Collect one trajectory using current actor.

        Returns:
            List of (obs, action, node_id) tuples
        """
        # Sample random source and target nodes
        node_ids = list(self.training_data.node_states.keys())

        if self.config.sampling_method == "uniform":
            idx = self.rng.integers(len(self.valid_pairs))
            source_id, target_id = self.valid_pairs[idx]

        elif self.config.sampling_method == "ucb":
            gains = self.node_pair_gains
            ucbs = np.sqrt(2 * np.log(max(self.total_samples, 1)) / (self.node_pair_samples + 1e-8))
            weights = np.where(self.valid_pairs_mask, gains + self.config.ucb_beta * ucbs, -np.inf)
            flat_index = int(np.argmax(weights))
            source_id = flat_index // self.num_nodes
            target_id = flat_index % self.num_nodes

        elif self.config.sampling_method == "stochastic_ucb":
            gains = self.node_pair_gains
            ucbs = np.sqrt(2 * np.log(max(self.total_samples, 1)) / (self.node_pair_samples + 1e-8))
            weights = gains + self.config.ucb_beta * ucbs
            weights = np.where(self.valid_pairs_mask, weights, -np.inf)
            exp_weights = np.exp(weights - np.max(weights))
            exp_weights = np.where(self.valid_pairs_mask, exp_weights, 0.0)
            probabilities = exp_weights / np.sum(exp_weights)
            flat_index = self.rng.choice(self.num_nodes * self.num_nodes, p=probabilities.flatten())
            source_id = flat_index // self.num_nodes
            target_id = flat_index % self.num_nodes
        
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
        
        self.node_pair_samples[source_id, target_id] += 1
        self.total_samples += 1


        # source_id = self.rng.choice(node_ids)
        # target_id = self.rng.choice(node_ids)

        # Sample random source state
        source_states = self.training_data.node_states[source_id]
        if len(source_states) == 0:
            return []

        source_state = source_states[self.rng.integers(len(source_states))]

        # Reset environment to source state
        env = self.system.env
        current_state, _ = env.reset_from_state(source_state)

        trajectory = []

        successful = False

        for step in range(self.config.max_episode_steps):
            # Flatten current state
            state_flat = self._flatten_state(current_state)

            # Get current node ID
            current_atoms = self.system.perceiver.step(current_state)
            current_node_id = self._find_node_for_atoms(current_atoms)
            if current_node_id is None:
                current_node_id = source_id  # Fallback to source

            # Select action using actor; pass current_atoms to avoid second perceiver call
            action = self.get_action(current_state, target_id, current_atoms=current_atoms)

            # Store (obs, action, node_id); state_flat computed on demand from obs
            trajectory.append((current_state, action, current_node_id))

            # Check if reached goal
            if current_node_id == target_id:
                successful = True
                break

            # Execute action
            current_state, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        if successful:
            self.node_pair_successes[(source_id, target_id)].append(1)
        else:
            self.node_pair_successes[(source_id, target_id)].append(0)

        return trajectory

    def _get_base_action(self, obs: "ObsType", target_node: int) -> NDArray:
        """Get the base skill action for (obs, target_node), or zeros if unavailable."""
        if not self.config.residualize or self.action_space_type != "continuous":
            return np.zeros(self.action_dim, dtype=np.float32)

        start_atoms = self.system.perceiver.step(obs)
        goal_atoms = self._node_atoms_dict.get(target_node, set())
        edge = self.first_edge_dict.get(
            (frozenset(start_atoms), frozenset(goal_atoms)), None
        )
        if edge is not None:
            operator = edge.operator
            skills = [s for s in self.system.skills if s.can_execute(operator)]
            if skills:
                skill = skills[0]
                skill.reset(operator)
                try:
                    action = skill.get_action(obs)
                except Exception:
                    action = None
                if action is not None:
                    return np.array(action, dtype=np.float32)

        return np.zeros(self.action_dim, dtype=np.float32)

    def _get_base_actions_batch(
        self,
        obs_list: list[Any],
        current_node_ids: NDArray,
        goal_node_ids: NDArray,
    ) -> NDArray:
        """Compute base skill actions for a batch of (obs, current_node, goal_node) pairs.

        Args:
            obs_list: raw observations at the sampled timestep
            current_node_ids: (batch_size,) node IDs at the sampled timestep
            goal_node_ids: (batch_size,) goal node IDs for each sample

        Returns:
            base_actions: (batch_size, action_dim) float32 array
        """
        batch_size = len(obs_list)
        base_actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)

        if not self.config.residualize or self.action_space_type != "continuous":
            return base_actions

        for i in range(batch_size):
            start_atoms = self._node_atoms_dict.get(int(current_node_ids[i]), set())
            goal_atoms = self._node_atoms_dict.get(int(goal_node_ids[i]), set())
            edge = self.first_edge_dict.get(
                (frozenset(start_atoms), frozenset(goal_atoms)), None
            )
            if edge is None:
                continue
            operator = edge.operator
            skills = [s for s in self.system.skills if s.can_execute(operator)]
            if not skills:
                continue
            skill = skills[0]
            skill.reset(operator)
            try:
                action = skill.get_action(obs_list[i])
            except Exception:
                action = None
            if action is not None:
                base_actions[i] = np.array(action, dtype=np.float32)

        return base_actions

    def get_action(
        self, obs: "ObsType", target_node: int, current_atoms: set | None = None
    ) -> NDArray | int:
        state_flat = self._flatten_state(obs)
        base_action = self._get_base_action(obs, target_node)
        return self._select_action_with_actor(state_flat, target_node, base_action)

    def _select_action_with_actor(
        self, state_flat: NDArray, target_node: int, base_action: NDArray | None = None
    ) -> NDArray | int:
        """Select action using learned actor.

        Args:
            state_flat: Flattened current state
            target_node: Target node ID
            base_action: Base skill action for residual actor (ignored if not residualize)

        Returns:
            Action (int for discrete, ndarray for continuous)
        """
        device = torch.device(self.config.device)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)

            # Convert target node to atom vector
            target_atoms = self._node_atoms_dict.get(target_node, set())
            goal_atom_vector = self.create_atom_vector(target_atoms)
            goal_tensor = torch.FloatTensor(goal_atom_vector).unsqueeze(0).to(device)

            if self.action_space_type == "discrete":
                action_tensor = self.actor.sample(state_tensor, goal_tensor)
                return int(action_tensor.item())
            else:
                if isinstance(self.actor, ResidualContinuousActor) and base_action is not None:
                    base_tensor = torch.FloatTensor(base_action).unsqueeze(0).to(device)
                    action_tensor, _ = self.actor.sample(
                        state_tensor, goal_tensor, base_tensor, deterministic=True
                    )
                else:
                    action_tensor, _ = self.actor.sample(
                        state_tensor, goal_tensor, deterministic=True
                    )
                return action_tensor.squeeze(0).cpu().numpy()

    def _find_node_for_atoms(self, atoms: set) -> int | None:
        """Find node ID that matches given atoms."""
        return self._atoms_to_node_id.get(frozenset(atoms), None)

    def _update_networks(self) -> tuple[float, float]:
        """Update critic and actor networks.

        Returns:
            (critic_loss, actor_loss)
        """
        device = torch.device(self.config.device)

        # Sample batch from replay buffer (obs_list contains raw observations)
        obs_list, actions, future_nodes, current_nodes = self.replay_buffer.sample_batch_crtr(
            batch_size=self.config.batch_size,
            repetition_factor=self.config.repetition_factor,
        )

        # Flatten observations → states array
        states = np.array(
            [self._flatten_state(obs) for obs in obs_list], dtype=np.float32
        )

        # Convert node IDs to atom vectors
        future_atom_vectors = np.array(
            [
                self.create_atom_vector(self._node_atoms_dict.get(node_id, set()))
                for node_id in future_nodes
            ]
        )

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        future_atoms_tensor = torch.FloatTensor(future_atom_vectors).to(device)

        # === Update Critic ===
        self.critic_optimizer.zero_grad()

        # Encode (state, action) pairs and goal atoms
        sa_repr = self.sa_encoder(states_tensor, actions_tensor)  # phi
        g_repr = self.g_encoder(future_atoms_tensor)               # psi

        B = sa_repr.shape[0]

        # L2 regularization: constrain E[||psi||²] ≈ c via dual Lagrangian
        l2 = (g_repr ** 2).sum(dim=1).mean()

        # Alignment: mean squared L2 between matched (phi_i, psi_i) pairs
        l_align = ((sa_repr - g_repr) ** 2).mean(dim=1)  # (B,)

        # Uniformity: log-sum-exp of negative pairwise distances (off-diagonal)
        pdist = ((sa_repr.unsqueeze(1) - g_repr.unsqueeze(0)) ** 2).mean(dim=-1)  # (B, B)
        I = torch.eye(B, device=device)
        l_unif = (
            torch.logsumexp(-(pdist * (1 - I)), dim=1) +
            torch.logsumexp(-(pdist.T * (1 - I)), dim=1)
        ) / 2.0  # (B,)

        # Dual Lagrangian: equality constraint E[||psi||²] = c
        dual_loss = self.log_lambda * (self.config.c - l2.detach())
        critic_loss = (l_align + l_unif).mean() + self.log_lambda.detach() * l2 + dual_loss

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()),
            self.config.grad_clip,
        )
        self.critic_optimizer.step()

        # === Update Actor ===
        self.actor_optimizer.zero_grad()

        # Compute live base actions for this batch (using current skill policy)
        if isinstance(self.actor, ResidualContinuousActor):
            base_actions_np = self._get_base_actions_batch(obs_list, current_nodes, future_nodes)
            base_actions_tensor = torch.FloatTensor(base_actions_np).to(device)
        else:
            base_actions_tensor = None

        # Sample new actions from actor for current states and goals
        # Goal: maximize alignment between sa_encoder(s, π(s,g)) and g_encoder(g)

        # For discrete actions: sample from policy
        # For continuous actions: use reparameterization trick

        if self.action_space_type == "discrete":
            # Sample actions from actor
            sampled_actions = self.actor.sample(states_tensor, future_atoms_tensor)

            # Get log probs for REINFORCE-style gradient
            log_probs = self.actor.get_log_prob(
                states_tensor, future_atoms_tensor, sampled_actions
            )

            # Convert sampled actions to one-hot for encoder
            # (Discrete actions need to be encoded properly)
            # For simplicity, we'll use a workaround: encode action index as float
            sampled_actions_float = F.one_hot(
                sampled_actions, num_classes=self.action_dim
            ).float()

            # Encode (state, sampled_action)
            sa_repr_actor = self.sa_encoder(states_tensor, sampled_actions_float)

        else:
            # Continuous: use reparameterization trick
            if base_actions_tensor is not None:
                sampled_actions, log_probs = self.actor.sample(
                    states_tensor, future_atoms_tensor, base_actions_tensor
                )
            else:
                sampled_actions, log_probs = self.actor.sample(
                    states_tensor, future_atoms_tensor
                )
            sampled_actions.retain_grad()  # DEBUG: Keep gradient for non-leaf
            sa_repr_actor = self.sa_encoder(states_tensor, sampled_actions)

        # Encode goal atoms (detached - only actor should receive gradients)
        g_repr_actor = self.g_encoder(future_atoms_tensor).detach()

        # Actor loss: minimize L2 alignment between sa_repr and g_repr
        actor_loss = ((sa_repr_actor - g_repr_actor) ** 2).mean(dim=1).mean()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        self.training_step += 1

        return critic_loss.item(), actor_loss.item()

    def latent_dist(self, source_state: ObsType, target_node: int) -> float:
        """Compute distance in latent space between state and node.

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            L2 distance in embedding space
        """

        device = torch.device(self.config.device)

        # Flatten state first
        state_flat = self._flatten_state(source_state)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)

            # Convert target node to atom vector
            target_atoms = self._node_atoms_dict.get(target_node, set())
            goal_atom_vector = self.create_atom_vector(target_atoms)
            goal_tensor = torch.FloatTensor(goal_atom_vector).unsqueeze(0).to(device)

            # Sample action from actor
            if self.action_space_type == "discrete":
                action_tensor = self.actor.sample(state_tensor, goal_tensor)
                action_onehot = F.one_hot(
                    action_tensor, num_classes=self.action_dim
                ).float()
                sa_repr = self.sa_encoder(state_tensor, action_onehot)
            else:
                action_tensor, _ = self.actor.sample(state_tensor, goal_tensor)
                sa_repr = self.sa_encoder(state_tensor, action_tensor)

            # Encode goal atoms
            g_repr = self.g_encoder(goal_tensor)

            return torch.norm(sa_repr - g_repr, p=2).item()


    def estimate_distance(self, source_state: ObsType, target_node: int) -> float:
        """Estimate trajectory distance from state to node.

        For V4, we use a simple transformation of the latent distance.
        This can be refined based on empirical results.

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            Estimated number of steps to reach target from source
        """

        d_sg_sq = (self.latent_dist(source_state, target_node)) ** 2
        d_gg_sq = 0
        target_states = self.training_data.node_states[target_node]

        # print(target_states)

        n = 0
        for target_state in random.sample(target_states, min(100, len(target_states))):
            d_gg_sq += (self.latent_dist(target_state, target_node)) ** 2
            n += 1
        d_gg_sq /= n

        # print("D_gg:", d_gg_sq, "D_sg:", d_sg_sq)

        return max(0, (1 / (2 * np.log(self.config.gamma))) * (d_gg_sq - d_sg_sq))


    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Estimate distance between nodes.

        Samples a random state from source_node and estimates distance.

        Args:
            source_node: Source node ID
            target_node: Target node ID

        Returns:
            Estimated distance
        """
        source_states = self.training_data.node_states[source_node]
        target_states = self.training_data.node_states[target_node]
        k = self.config.node_distance_samples

        src_sample = random.sample(source_states, min(k, len(source_states)))
        tgt_sample = random.sample(target_states, min(k, len(target_states)))
        if not src_sample or not tgt_sample:
            return float(self.config.max_episode_steps)

        d_sg_sq = float(np.mean([(self.latent_dist(s, target_node)) ** 2 for s in src_sample]))
        d_gg_sq = float(np.mean([(self.latent_dist(t, target_node)) ** 2 for t in tgt_sample]))

        # print("D_gg:", d_gg_sq, "D_sg:", d_sg_sq)

        # return self.config.dist_scale * d_sg_sq

        return self.config.dist_scale * max(0, -(1 / (2 * np.log(self.config.gamma))) * (d_sg_sq - d_gg_sq))


    
    def prune_by_success(self, success_threshold: float, max_steps: int) -> GoalConditionedTrainingData:
        """Prune shortcuts based on estimated success rate of reaching target from source.

        Args:
            success_threshold: Minimum success rate to keep a shortcut
            max_steps: Maximum steps used to estimate success probability
        Returns:
            Pruned training data with only shortcuts above success threshold
        """
        print("Heuristic-Estimated Success Probs of All Shortcuts:")
        pruned_pairs = []
        for (x, y) in self.training_data.unique_shortcuts:
            k = self.config.num_reliability_trials
            successes = self.node_pair_successes.get((x, y), [])
            if k > 0:
                prob = np.mean(successes) if (len(successes) >= k) else 0.0
            else:
                prob = np.mean(successes) if successes else 0.0

            print(f"  ({x} -> {y}) success prob: {prob:.2f})")

            if prob > success_threshold:
                # print(f"  Keeping shortcut ({x} -> {y}) with success prob {prob:.2f}")
                pruned_pairs.append((x, y))

        selected_set = set(pruned_pairs)
        selected_indices = [
            i for i, (src, tgt) in enumerate(self.training_data.valid_shortcuts)
            if (src, tgt) in selected_set
        ]
        print(f"  ({len(selected_indices)} state-node pairs)")

        original_shortcut_info = self.training_data.config.get("shortcut_info", [])
        pruned_shortcut_info = (
            [original_shortcut_info[i] for i in selected_indices]
            if original_shortcut_info else []
        )

        return GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[self.training_data.current_atoms[i] for i in selected_indices],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[self.training_data.valid_shortcuts[i] for i in selected_indices],
            unique_shortcuts=pruned_pairs,
            node_states=self.training_data.node_states,
            node_atoms=self.training_data.node_atoms,
            graph=self.training_data.graph,
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "crl",
                "threshold": self.config.threshold,
            },
        )

    def prune(self, max_shortcuts: int) -> GoalConditionedTrainingData:

        print(f"\n[DEBUG] Pruning greedily to max_shortcuts={max_shortcuts}")

        print("Lengths of All Shortcuts:")
        for (x, y) in self.training_data.unique_shortcuts:
            d = self.estimate_node_distance(x, y)
            print(f"  ({x} -> {y}): {d:.2f}")

        if max_shortcuts is None or max_shortcuts >= len(self.training_data.unique_shortcuts):
            return self.training_data

        # Reshape self.training_data.unique_shortcuts
        starts = np.array([x for (x, _) in self.training_data.unique_shortcuts])
        ends = np.array([y for (_, y) in self.training_data.unique_shortcuts])

        curr_gains = self.node_pair_gains.copy()
        curr_dists = self.node_pair_graph_dists.copy()

        pruned_pairs = []

        self.config.auto_dist_scale = False

        for i in range(max_shortcuts):
            self._update_gains()

            # Take maximum gain shortcut using node_pairs as indices into self.node_pair_gains
            gains = self.node_pair_gains[starts, ends]
            max_idx = np.argmax(gains)
            source_id = starts[max_idx]
            target_id = ends[max_idx]
            pruned_pairs.append((source_id, target_id))

            # Update graph distances
            self._update_graph_distances(source_id, target_id)

            # Remove this pair from consideration
            starts = np.delete(starts, max_idx)
            ends = np.delete(ends, max_idx)

            print(f"  Selected shortcut {i+1}: {source_id} -> {target_id} with gain {gains[max_idx]:.2f}")

        self.config.auto_dist_scale = True
        self.node_pair_gains = curr_gains
        self.node_pair_graph_dists = curr_dists
        
        # Keep all state-node pairs that correspond to selected node-node pairs
        selected_set = set(pruned_pairs)
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
            current_atoms=[
                self.training_data.current_atoms[i] for i in selected_indices
            ],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[
                self.training_data.valid_shortcuts[i] for i in selected_indices
            ],
            unique_shortcuts=pruned_pairs,  # Unique node-node pairs
            node_states=self.training_data.node_states,  # Keep all node states
            node_atoms=self.training_data.node_atoms,  # Keep all node atoms
            graph=self.training_data.graph,  # Keep planning graph
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "crl",
                "threshold": self.config.threshold,
            },
        )

        return pruned_data

    
    def exact_gain(self, source_node: int, target_node: int) -> float:
        """Estimate gain of training on a shortcut, relative to distance in the
        initial graph.

        Higher gain means more useful shortcut.
        """

        x = source_node
        y = target_node

        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)

        U = np.where(np.isfinite(d[:, x]))[0]   # can reach x
        V = np.where(np.isfinite(d[y, :]))[0]   # reachable from y
        
        d_ux = d[U, x][:, None]

        d_yv = d[y, V][None, :]

        new_paths = d_ux + L + d_yv

        old_paths = np.minimum(d[np.ix_(U, V)], self.config.max_episode_steps)

        improvement = np.maximum(0, old_paths - new_paths)

        return float(np.sum(improvement))
    
    def estimate_gain(self, source_node: int, target_node: int) -> float:
        """Estimate gain of training on a shortcut, relative to distance in the
        initial graph.

        Higher gain means more useful shortcut.
        """

        x = source_node
        y = target_node

        d = np.clip(self.node_pair_graph_dists, 0, self.config.max_episode_steps)
        L = self.estimate_node_distance(source_node, target_node)
        
        delta_in = d[:, y] - (d[:, x] + L)
        delta_in = delta_in[delta_in > 0]
        if delta_in.size == 0:
            return 0.0
        
        delta_out = d[x, :] - (L + d[y, :])
        delta_out = delta_out[delta_out > 0]
        if delta_out.size == 0:
            return 0.0
        
        min_matrix = np.minimum(delta_in[:, None], delta_out[None, :])
        gain = np.sum(min_matrix)

        return float(gain)
        
    def naive_gain(self, source_node: int, target_node: int) -> float:
        
        L = self.estimate_node_distance(source_node, target_node)
        d_xy = self.node_pair_graph_dists[source_node, target_node]
        gain = np.clip(d_xy - L, 0, self.config.max_episode_steps)
        return float(gain)
    
    def get_gain(self, source_node: int, target_node: int) -> float:
        
        if self.config.gain_method == "exact":
            return self.exact_gain(source_node, target_node)
        elif self.config.gain_method == "estimate":
            return self.estimate_gain(source_node, target_node)
        elif self.config.gain_method == "naive":
            return self.naive_gain(source_node, target_node)
        else:
            raise ValueError(f"Unknown gain method: {self.config.gain_method}")


    def _update_gains(self) -> None:
        """Update gain estimates for all node pairs based on current networks."""
        if self.config.auto_dist_scale:
            self.config.dist_scale = 1.0  # Reset to 1.0 before auto-scaling
            shortcut_dists = np.zeros((self.num_nodes, self.num_nodes))
            for source_id, target_id in self.training_data.unique_shortcuts:
                dist = self.estimate_node_distance(source_id, target_id)
                shortcut_dists[source_id, target_id] = dist
            ratios = shortcut_dists / (self.original_node_pair_graph_dists + 1e-8)
            valid = np.isfinite(ratios) & (self.original_node_pair_graph_dists > 0)

            q = self.config.dist_quantile
            if valid.any():
                ratio = 1 / np.quantile(ratios[valid], q)
                if not np.isinf(ratio) and ratio > 0:
                    self.config.dist_scale = ratio if ratio < 1 else 1.0
                else:
                    self.config.dist_scale = 1.0
            else:
                self.config.dist_scale = 1.0
        for source_id, target_id in self.training_data.unique_shortcuts:
            gain = self.get_gain(source_id, target_id)
            self.node_pair_gains[source_id, target_id] = gain

    def _update_graph_distances(self, source_node: int, target_node: int) -> None:
        """Update graph distances after adding a shortcut."""
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)

        # d(u,x) column, shape (n, 1)
        d_ux = d[:, source_node][:, None]

        # d(y,v) row, shape (1, n)
        d_yv = d[target_node, :][None, :]

        # Candidate new distances via x -> y
        via_xy = d_ux + L + d_yv

        # Elementwise minimum
        self.node_pair_graph_dists = np.minimum(d, via_xy)

    def estimate_weight(self, source_node: int, target_node: int) -> float:
        """Estimate weight of one node pair -- the UCB notion of whether it should be sampled
        for learning.

        Higher weight means more likely to be chosen.
        """

        if source_node == target_node:
            return 0.0
        
        gain = self.node_pair_gains[source_node, target_node]

        ucb = np.sqrt(2 * np.log(self.total_samples) / (self.node_pair_samples[source_node, target_node] + 1e-8))
        return gain + self.config.ucb_beta * ucb
    

    def save(self, path: str) -> None:
        """Save heuristic to disk.

        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)

        # Save networks
        torch.save(self.sa_encoder.state_dict(), os.path.join(path, "sa_encoder.pt"))
        torch.save(self.g_encoder.state_dict(), os.path.join(path, "g_encoder.pt"))
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.log_lambda, os.path.join(path, "log_lambda.pt"))

        # Save atom indexing
        with open(os.path.join(path, "atom_index.pkl"), "wb") as f:
            pickle.dump(
                {"atom_to_index": self.atom_to_index, "next_index": self._next_atom_index},
                f,
            )

        # Save config
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)

        print(f"Saved CRL V2 heuristic to {path}")

    def load(self, path: str) -> None:
        """Load heuristic from disk.

        Args:
            path: Directory path to load from
        """
        device = torch.device(self.config.device)

        # Load networks
        self.sa_encoder.load_state_dict(
            torch.load(os.path.join(path, "sa_encoder.pt"), map_location=device)
        )
        self.g_encoder.load_state_dict(
            torch.load(os.path.join(path, "g_encoder.pt"), map_location=device)
        )
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"), map_location=device)
        )
        log_lambda_path = os.path.join(path, "log_lambda.pt")
        if os.path.exists(log_lambda_path):
            self.log_lambda = torch.load(log_lambda_path, map_location=device)
            self.log_lambda.requires_grad_(True)

        # Load atom indexing
        with open(os.path.join(path, "atom_index.pkl"), "rb") as f:
            atom_data = pickle.load(f)
            self.atom_to_index = atom_data["atom_to_index"]
            self._next_atom_index = atom_data["next_index"]

        print(f"Loaded CRL V2 heuristic from {path}")
