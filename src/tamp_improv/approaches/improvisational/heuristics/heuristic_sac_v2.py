"""Goal-conditioned distance heuristic V2 using manual SAC + HER.

Implements Soft Actor-Critic with Hindsight Experience Replay for
goal-conditioned distance estimation. Reward is -1 everywhere except
at the goal node (reward = 0), so Q(s,a,g) ≈ -distance(s,g).

Shares the cmd_v2/crl_v2 interface: exact/estimate/naive gain methods,
greedy shortcut pruning, UCB trajectory sampling, residualization, and
multi-round training.

Key components:
- GoalConditionedQNetwork: Q(state, action, goal_atoms) → scalar
- ContinuousActor: π(state, goal_atoms) → action (Gaussian + tanh)
- SACReplayBuffer: (s, a, r, s', done, goal) transitions + HER
- SAC update: clipped double-Q critics + entropy-regularized actor
- Auto-entropy tuning: log_alpha updated to maintain target entropy
"""

import copy
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from relational_structs import GroundAtom, GroundOperator

from task_then_motion_planning.planning import TaskThenMotionPlanningFailure

from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.graph_training import compute_first_edge
from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
from tamp_improv.approaches.improvisational.heuristics.networks import (
    ContinuousActor,
    ResidualContinuousActor,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class SACV2HeuristicConfig:
    """Configuration for manual SAC-based goal-conditioned distance heuristic."""

    wandb_enabled: bool = False
    threshold: float = 0.05
    beta: float = 1.0

    # Network architecture
    hidden_dims: list[int] | None = None  # Q-network hidden layers; default [256, 256]
    actor_hidden_dims: list[int] | None = None  # Actor hidden layers; falls back to hidden_dims

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # SAC hyperparameters
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005  # Polyak soft-update coefficient
    target_update_interval: int = 1  # Soft-update every N gradient steps
    auto_entropy: bool = True
    init_temperature: float = 0.2  # Initial α

    # Hindsight Experience Replay
    her_n_sampled_goal: int = 4  # HER relabelled transitions per real transition
    her_strategy: str = "future"  # "future" or "final"

    # Training loop
    trajectories_per_epoch: int = 10
    num_epochs_per_round: int = 20
    max_episode_steps: int = 100
    learn_frequency: int = 1  # Start gradient updates after this many epochs
    gain_update_frequency: int = 10  # Update UCB gains every N epochs (decoupled from learn_frequency)
    medoid_update_frequency: int = 10  # Re-elect node medoids every N epochs
    iters_per_epoch: int = 50  # Gradient steps per epoch (SAC needs many; ~steps collected)
    grad_clip: float = 1.0  # Gradient clipping for actor and critic
    log_alpha_min: float = -5.0  # Lower bound on log_alpha to prevent alpha → 0

    # Multi-round
    num_rounds: int = 1
    keep_fraction: float = 0.1
    exploration_factor: float = 0.0

    # Trajectory sampling
    sampling_method: str = "uniform"  # "uniform", "ucb", or "stochastic_ucb"
    ucb_beta: float = 1.0

    # Gain estimation
    gain_method: str = "exact"  # "exact", "estimate", or "naive"
    dist_scale: float = 1.0
    dist_quantile: float = 1.0
    auto_dist_scale: bool = False
    residualize: bool = False
    max_atom_size: int = 50

    # Success tracking
    num_reliability_trials: int = 10  # Window size for success/failure deque per node pair

    device: str = "cuda"


# ── Networks ──────────────────────────────────────────────────────────


class GoalConditionedQNetwork(nn.Module):
    """Q(state, action, goal_atoms) → scalar Q-value."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        input_dim = state_dim + action_dim + atom_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([states, actions, goals], dim=-1)
        return self.net(x)  # (B, 1)


# ── Replay buffer ─────────────────────────────────────────────────────

# Trajectory step: (state_flat, action, reward, next_state_flat, done,
#                   achieved_atoms_vec, next_achieved_atoms_vec, obs_raw)
_TrajStep = tuple[NDArray, NDArray, float, NDArray, bool, NDArray, NDArray, Any]


class SACReplayBuffer:
    """Circular replay buffer storing (s, a, r, s', done, goal, obs_raw, goal_node_id)."""

    def __init__(self, max_size: int):
        self.buffer: deque[
            tuple[NDArray, NDArray, float, NDArray, bool, NDArray, Any, int, int]
        ] = deque(maxlen=max_size)

    def add(
        self,
        state: NDArray,
        action: NDArray,
        reward: float,
        next_state: NDArray,
        done: bool,
        goal: NDArray,
        obs_raw: Any,
        goal_node_id: int,
        source_node_id: int,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, goal, obs_raw, goal_node_id, source_node_id))

    def sample(
        self, batch_size: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, list, NDArray, NDArray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals, obs_raws, goal_node_ids, source_node_ids = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            np.array(goals, dtype=np.float32),
            list(obs_raws),
            np.array(goal_node_ids, dtype=np.int64),
            np.array(source_node_ids, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Main heuristic class ──────────────────────────────────────────────


class SACv2Heuristic(BaseHeuristic):
    """Manual SAC with HER for goal-conditioned distance estimation.

    Training produces Q(s, a, g) ≈ -distance(s, g) by rewarding -1 per step
    and 0 on goal reaching. Distances are queried via estimate_distance().
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
        config: SACV2HeuristicConfig | None = None,
        first_edge_dict: (
            dict[
                tuple[frozenset["GroundAtom"], frozenset["GroundAtom"]],
                PlanningGraphEdge,
            ]
            | None
        ) = None,
        **kwargs: Any,
    ):
        super().__init__(training_data, graph_distances)
        self.system = system
        self.rng = rng
        self.first_edge_dict = first_edge_dict
        self.internal_graph = training_data.graph  # Keep an internal copy of the graph for edge lookups (updated in multi-round training)
        self.virtual_system = copy.deepcopy(system)

        if config is None:
            config = SACV2HeuristicConfig()
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.config = config

        # Determine state dimension
        sample_states = next(iter(training_data.node_states.values()))
        self.state_dim = self._flatten_state(sample_states[0]).shape[0]

        # Continuous action space only
        env = system.env
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("SACv2Heuristic requires a continuous (Box) action space.")
        self.action_dim = int(np.prod(env.action_space.shape))

        # Node bookkeeping
        self.num_nodes = len(training_data.node_states)
        self._node_atoms_dict = dict(training_data.node_atoms)

        # Dynamic atom indexing
        self.atom_to_index: dict[str, int] = {}
        self._next_atom_index = 0
        for atoms in self._node_atoms_dict.values():
            for atom in atoms:
                self._get_atom_index(str(atom))

        # Reverse lookup: frozenset of active atom indices → node_id (O(1) HER lookup)
        self._atom_indices_to_node_id: dict[frozenset, int] = {
            frozenset(self.atom_to_index[str(a)] for a in atoms if str(a) in self.atom_to_index): node_id
            for node_id, atoms in self._node_atoms_dict.items()
        }

        # UCB / gain arrays
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

        self.node_pair_successes: dict[tuple[int, int], Any] = {}
        for (i, j) in self.valid_pairs:
            k = self.config.num_reliability_trials
            if k > 0:
                self.node_pair_successes[(i, j)] = deque(maxlen=k)
            else:
                self.node_pair_successes[(i, j)] = []

        print(f"SAC V2: state_dim={self.state_dim}, action_dim={self.action_dim}, "
              f"num_nodes={self.num_nodes}")

        self._init_networks()
        self.replay_buffer = SACReplayBuffer(config.buffer_size)
        self.training_step = 0

        # Medoid cache: node_id → representative state (initially random, skips empty nodes)
        self._node_medoids: dict[int, Any] = {
            node_id: random.choice(states)
            for node_id, states in training_data.node_states.items()
            if states
        }

    # ── Network initialisation ────────────────────────────────────────

    def _init_networks(self) -> None:
        device = torch.device(self.config.device)
        hidden_dims = self.config.hidden_dims or [256, 256]
        actor_hidden_dims = self.config.actor_hidden_dims or hidden_dims
        atom_dim = self.config.max_atom_size

        if self.config.residualize:
            self.actor = ResidualContinuousActor(
                self.state_dim, self.action_dim, atom_dim, actor_hidden_dims
            ).to(device)
        else:
            self.actor = ContinuousActor(
                self.state_dim, self.action_dim, atom_dim, actor_hidden_dims
            ).to(device)
        self.q1 = GoalConditionedQNetwork(
            self.state_dim, self.action_dim, atom_dim, hidden_dims
        ).to(device)
        self.q2 = GoalConditionedQNetwork(
            self.state_dim, self.action_dim, atom_dim, hidden_dims
        ).to(device)
        self.q1_target = GoalConditionedQNetwork(
            self.state_dim, self.action_dim, atom_dim, hidden_dims
        ).to(device)
        self.q2_target = GoalConditionedQNetwork(
            self.state_dim, self.action_dim, atom_dim, hidden_dims
        ).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.critic_lr,
        )

        # Auto-entropy: target entropy = -|A| (standard SAC default)
        self.target_entropy = float(-self.action_dim)
        self.log_alpha = torch.tensor(
            [np.log(self.config.init_temperature)],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.actor_lr
        )

    # ── System update (multi-round) ───────────────────────────────────

    def update_system(
        self,
        new_system: "ImprovisationalTAMPSystem",
        new_graph: "PlanningGraph",  # noqa: F841
        new_graph_distances: dict[tuple[int, int], float],
        new_first_edge_dict: dict[
            tuple[frozenset["GroundAtom"], frozenset["GroundAtom"]], PlanningGraphEdge
        ],
        reset_actor: bool = False,
    ) -> None:
        """Update the TAMP system and distances for a new training round."""
        self.system = new_system
        self.internal_graph = new_graph
        self.graph_distances = new_graph_distances
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in new_graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist
        self.first_edge_dict = new_first_edge_dict

        # Reset actor for fresh policy in the new round
        device = torch.device(self.config.device)
        actor_hidden_dims = self.config.actor_hidden_dims or (
            self.config.hidden_dims or [256, 256]
        )
        if self.config.residualize:
            self.actor = ResidualContinuousActor(
                self.state_dim, self.action_dim, self.config.max_atom_size, actor_hidden_dims
            ).to(device)
        else:
            self.actor = ContinuousActor(
                self.state_dim, self.action_dim, self.config.max_atom_size, actor_hidden_dims
            ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

    # ── Training ──────────────────────────────────────────────────────

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Stub — pipeline calls train_one_round() directly."""
        return {"critic_losses": [], "actor_losses": [], "buffer_size": 0}

    def train_one_round(self) -> dict[str, Any]:
        """Train for one round (multiple epochs)."""
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        k = self.config.num_reliability_trials
        for key in self.node_pair_successes:
            self.node_pair_successes[key] = deque(maxlen=k) if k > 0 else []

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        self._update_medoids()
        self._update_gains()

        for epoch in range(self.config.num_epochs_per_round):
            # Collect trajectories
            for _ in range(self.config.trajectories_per_epoch):
                trajectory, goal_vec, target_id, source_id = self._collect_trajectory()
                self._add_to_buffer_with_her(trajectory, goal_vec, target_id, source_id)

            # Update networks
            if (epoch + 1) % self.config.learn_frequency == 0:
                if len(self.replay_buffer) >= self.config.batch_size:
                    for _ in range(self.config.iters_per_epoch):
                        c_loss, a_loss = self._update_networks()
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

            if (epoch + 1) % self.config.medoid_update_frequency == 0:
                self._update_medoids()

            if self.config.sampling_method != "uniform" and (epoch + 1) % self.config.gain_update_frequency == 0:
                self._update_gains()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                c_str = f"{critic_losses[-1]:.4f}" if critic_losses else "N/A"
                a_str = f"{actor_losses[-1]:.4f}" if actor_losses else "N/A"
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs_per_round} | "
                    f"Buffer: {len(self.replay_buffer)} | "
                    f"Critic: {c_str} | Actor: {a_str}"
                )

        self._update_gains()

        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "buffer_size": len(self.replay_buffer),
        }

    def _collect_trajectory(self) -> tuple[list[_TrajStep], NDArray, int, int]:
        """Collect one trajectory using current actor.

        Returns:
            trajectory: list of (s, a, r, s', done, achieved_vec, next_achieved_vec, obs_raw)
            goal_vec: multi-hot goal atom vector used during collection
            target_id: goal node ID
            source_id: source node ID
        """
        node_ids = list(self.training_data.node_states.keys())

        # Sample (source, target) node pair
        if self.config.sampling_method == "uniform":
            idx = self.rng.integers(len(self.valid_pairs))
            source_id, target_id = self.valid_pairs[idx]
        elif self.config.sampling_method == "ucb":
            gains = self.node_pair_gains
            ucbs = np.sqrt(
                2 * np.log(max(self.total_samples, 1))
                / (self.node_pair_samples + 1e-8)
            )
            weights = np.where(self.valid_pairs_mask, gains + self.config.ucb_beta * ucbs, -np.inf)
            flat_idx = int(np.argmax(weights))
            source_id, target_id = flat_idx // self.num_nodes, flat_idx % self.num_nodes
        elif self.config.sampling_method == "stochastic_ucb":
            gains = self.node_pair_gains
            ucbs = np.sqrt(
                2 * np.log(max(self.total_samples, 1))
                / (self.node_pair_samples + 1e-8)
            )
            weights = gains + self.config.ucb_beta * ucbs
            weights = np.where(self.valid_pairs_mask, weights, -np.inf)
            exp_w = np.exp(weights - np.max(weights))
            exp_w = np.where(self.valid_pairs_mask, exp_w, 0.0)
            probs = exp_w / exp_w.sum()
            flat_idx = int(
                self.rng.choice(self.num_nodes * self.num_nodes, p=probs.flatten())
            )
            source_id, target_id = flat_idx // self.num_nodes, flat_idx % self.num_nodes
        else:
            raise ValueError(f"Unknown sampling_method: {self.config.sampling_method}")

        self.node_pair_samples[source_id, target_id] += 1
        self.total_samples += 1

        # Sample source state
        source_states = self.training_data.node_states[source_id]
        source_state = source_states[self.rng.integers(len(source_states))]

        # Build goal vector
        target_atoms = self._node_atoms_dict.get(target_id, set())
        goal_vec = self.create_atom_vector(target_atoms)

        # Run episode
        env = self.system.env
        current_state, _ = env.reset_from_state(source_state)
        trajectory: list[_TrajStep] = []
        successful = False

        # Pre-compute perception for the initial state; reuse next_atoms as achieved_atoms
        # each step to halve perceiver calls.
        achieved_atoms = self.system.perceiver.step(current_state)
        achieved_vec = self.create_atom_vector(achieved_atoms)
        state_flat = self._flatten_state(current_state)

        for step in range(self.config.max_episode_steps):
            # Stochastic action during training
            action = self._select_action(state_flat, goal_vec, deterministic=False)

            next_state, _, terminated, truncated, _ = env.step(action)
            next_flat = self._flatten_state(next_state)
            next_atoms = self.system.perceiver.step(next_state)
            next_achieved_vec = self.create_atom_vector(next_atoms)

            goal_reached = target_atoms <= next_atoms  # superset check
            reward = 0.0 if goal_reached else -1.0
            done = terminated or truncated or goal_reached

            if goal_reached:
                successful = True

            trajectory.append(
                (state_flat, action, reward, next_flat, done, achieved_vec, next_achieved_vec,
                 current_state)
            )

            if done:
                break

            # Shift: next step's "achieved" = this step's "next"
            current_state = next_state
            achieved_atoms = next_atoms
            achieved_vec = next_achieved_vec
            state_flat = next_flat

        # if (source_id, target_id) in self.node_pair_successes:
        #     self.node_pair_successes[(source_id, target_id)].append(1 if successful else 0)

        if successful:
            print(f"Successfully reached target node {target_id} from source node {source_id} in {step+1} steps.")
            self.node_pair_successes[(source_id, target_id)].append(1)
        else:
            print(f"Failed to reach target node {target_id} from source node {source_id} after {self.config.max_episode_steps} steps.")
            self.node_pair_successes[(source_id, target_id)].append(0)


        return trajectory, goal_vec, target_id, source_id

    def _add_to_buffer_with_her(
        self, trajectory: list[_TrajStep], goal_vec: NDArray, target_id: int, source_id: int
    ) -> None:
        """Add real transitions and HER-relabelled transitions to the buffer."""
        if not trajectory:
            return

        T = len(trajectory)

        # Real transitions (goal_node_id = target_id, known at collection time)
        for s, a, r, s_, done, _, _, obs_raw in trajectory:
            self.replay_buffer.add(s, a, r, s_, done, goal_vec, obs_raw, target_id, source_id)

        # HER transitions — resolve goal node ID from the achieved atom vector
        for t in range(T):
            s, a, _, s_, done, _, next_achieved_vec, obs_raw = trajectory[t]

            if self.config.her_strategy == "final":
                her_indices = [T - 1]
            else:  # "future"
                n_future = min(self.config.her_n_sampled_goal, T - t - 1)
                her_indices = (
                    random.sample(range(t + 1, T), n_future) if n_future > 0 else []
                )

            for t_her in her_indices:
                # Goal = atoms achieved at step t_her
                her_goal_vec = trajectory[t_her][5]  # achieved_vec at t_her
                her_goal_node_id = self._find_node_for_goal_vec(her_goal_vec)

                # Reward: 0 if next state satisfies the HER goal
                goal_atoms_active = her_goal_vec > 0.5
                her_r = (
                    0.0
                    if np.all(next_achieved_vec[goal_atoms_active] > 0.5)
                    else -1.0
                )
                her_done = her_r == 0.0 or done
                self.replay_buffer.add(
                    s, a, her_r, s_, her_done, her_goal_vec, obs_raw, her_goal_node_id, source_id
                )

    def _select_action(
        self, state_flat: NDArray, goal_vec: NDArray, deterministic: bool = False
    ) -> NDArray:
        """Query actor for an action."""
        device = torch.device(self.config.device)
        with torch.no_grad():
            s_t = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
            g_t = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)
            action, _ = self.actor.sample(s_t, g_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def _update_networks(self) -> tuple[float, float]:
        """One SAC gradient step. Returns (critic_loss, actor_loss)."""
        device = torch.device(self.config.device)

        states, actions, rewards, next_states, dones, goals, _, goal_node_ids, source_node_ids = (
            self.replay_buffer.sample(self.config.batch_size)
        )
        s = torch.FloatTensor(states).to(device)
        a = torch.FloatTensor(actions).to(device)
        r = torch.FloatTensor(rewards).to(device)
        s_ = torch.FloatTensor(next_states).to(device)
        d = torch.FloatTensor(dones).to(device)
        g = torch.FloatTensor(goals).to(device)

        alpha = self.log_alpha.exp().detach()

        # ── Compute base actions once for the whole batch ─────────────
        # Used for both the critic target bootstrap and the actor update.
        # Note: obs_raw_list contains observations for s (current state); we reuse
        # these as an approximation for s_ (next state) in the critic target, which
        # is valid since s and s' share the same goal and consecutive states along a
        # trajectory have nearly identical base actions.
        if isinstance(self.actor, ResidualContinuousActor):
            base_actions_np = self._get_base_actions_batch(source_node_ids, goal_node_ids)
            base_actions_tensor = torch.FloatTensor(base_actions_np).to(device)
        else:
            base_actions_tensor = None

        # ── Critic update ────────────────────────────────────────────
        with torch.no_grad():
            if base_actions_tensor is not None:
                next_a, next_log_p = self.actor.sample(s_, g, base_actions_tensor)
            else:
                next_a, next_log_p = self.actor.sample(s_, g)
            q1_next = self.q1_target(s_, next_a, g)
            q2_next = self.q2_target(s_, next_a, g)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_p.unsqueeze(1)
            target = r + self.config.gamma * (1.0 - d) * q_next

        q1_pred = self.q1(s, a, g)
        q2_pred = self.q2(s, a, g)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.grad_clip,
        )
        self.critic_optimizer.step()

        # ── Actor update ─────────────────────────────────────────────
        if base_actions_tensor is not None:
            new_a, log_p = self.actor.sample(s, g, base_actions_tensor)
        else:
            new_a, log_p = self.actor.sample(s, g)
        q1_new = self.q1(s, new_a, g)
        q2_new = self.q2(s, new_a, g)
        actor_loss = (alpha * log_p.unsqueeze(1) - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        # ── Entropy coefficient update ────────────────────────────────
        if self.config.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_p.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Clamp log_alpha to prevent alpha → 0 (vicious saturation cycle)
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.config.log_alpha_min)

        # ── Soft target update ────────────────────────────────────────
        self.training_step += 1
        if self.training_step % self.config.target_update_interval == 0:
            self._soft_update_targets()

        # ── Diagnostics (every 100 steps) ────────────────────────────
        if self.training_step % 100 == 0:
            with torch.no_grad():
                mean_out, _ = self.actor.forward(s[:8], g[:8])
                action_sat = torch.tanh(mean_out).abs().mean().item()
            print(
                f"  [sac diag step={self.training_step}] "
                f"alpha={self.log_alpha.exp().item():.4f} "
                f"log_p={log_p.mean().item():.2f} "
                f"Q={q1_pred.mean().item():.2f} "
                f"actor_sat={action_sat:.3f} "  # near 1.0 = stuck at corners
                f"r0_frac={(r == 0.0).float().mean().item():.3f}"  # goal-reached rate
            )

        return critic_loss.item(), actor_loss.item()

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

    def _get_base_actions_batch(
        self,
        source_node_ids: NDArray,
        goal_node_ids: NDArray,
    ) -> NDArray:
        """Compute base skill actions for a batch of (source_node_id, goal_node_id) pairs.

        Uses stored node atoms for edge lookup — no perceiver calls needed.
        Returns zeros for transitions where no matching edge/skill is found.

        Args:
            source_node_ids: (batch_size,) source node IDs (-1 when unknown)
            goal_node_ids: (batch_size,) goal node IDs (-1 when no node matched)

        Returns:
            base_actions: (batch_size, action_dim) float32 array
        """
        batch_size = len(source_node_ids)
        base_actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)

        if not self.config.residualize or self.first_edge_dict is None:
            return base_actions

        # Cache skill lookups by operator to avoid re-scanning self.system.skills
        skill_cache: dict[Any, Any] = {}

        for i in range(batch_size):
            if source_node_ids[i] < 0 or goal_node_ids[i] < 0:
                continue
            start_atoms = self._node_atoms_dict.get(int(source_node_ids[i]), set())
            goal_atoms = self._node_atoms_dict.get(int(goal_node_ids[i]), set())
            edge = self.first_edge_dict.get(
                (frozenset(start_atoms), frozenset(goal_atoms)), None
            )
            if edge is None:
                continue
            operator = edge.operator
            if operator not in skill_cache:
                skills = [sk for sk in self.system.skills if sk.can_execute(operator)]
                skill_cache[operator] = skills[0] if skills else None
            skill = skill_cache[operator]
            if skill is None:
                continue
            skill.reset(operator)
            try:
                action = skill.get_action(edge.source.state)
            except Exception:
                action = None
            if action is not None:
                base_actions[i] = np.array(action, dtype=np.float32)

        return base_actions

    # ── Action + distance interface ───────────────────────────────────

    def get_action(self, obs: "ObsType", target_node: int) -> NDArray:
        state_flat = self._flatten_state(obs)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        goal_vec = self.create_atom_vector(target_atoms)

        if self.config.residualize and self.first_edge_dict is not None:
            start_atoms = self.system.perceiver.step(obs)
            edge = self.first_edge_dict.get(
                (frozenset(start_atoms), frozenset(target_atoms)), None
            )
            if edge is not None:
                operator = edge.operator
                skills = [sk for sk in self.system.skills if sk.can_execute(operator)]
                if not skills:
                    raise TaskThenMotionPlanningFailure(
                        f"No skill for operator {operator.name}"
                    )
                skill = skills[0]
                skill.reset(operator)
                try:
                    base_action = skill.get_action(obs)
                except Exception:
                    base_action = None
                if base_action is None:
                    base_action = np.zeros(self.action_dim, dtype=np.float32)
            else:
                base_action = np.zeros(self.action_dim, dtype=np.float32)
            device = torch.device(self.config.device)
            with torch.no_grad():
                s_t = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
                g_t = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)
                base_t = torch.FloatTensor(base_action).unsqueeze(0).to(device)
                action, _ = self.actor.sample(s_t, g_t, base_t, deterministic=True)
            return action.squeeze(0).cpu().numpy()

        return self._select_action(state_flat, goal_vec, deterministic=True)

    def estimate_distance(self, source_state: "ObsType", target_node: int) -> float:
        """Estimate distance via min(Q1, Q2) at actor's deterministic action.

        Returns max(0, -Q) since Q ≈ -distance.
        """
        state_flat = self._flatten_state(source_state)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        goal_vec = self.create_atom_vector(target_atoms)

        device = torch.device(self.config.device)
        with torch.no_grad():
            s_t = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
            g_t = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)
            action, _ = self.actor.sample(s_t, g_t, deterministic=True)
            q = torch.min(self.q1(s_t, action, g_t), self.q2(s_t, action, g_t)).item()
        return max(0.0, -q)

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        medoid = self._node_medoids.get(source_node)
        if medoid is None:
            return float("inf")
        return self.config.dist_scale * self.estimate_distance(medoid, target_node)

    def _batch_estimate_raw_distances(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        """Batch-compute raw (unscaled) estimated distances for all (src, tgt) pairs.

        Replaces 30,000 serial forward passes with one batched GPU call.
        Returns array of shape (len(pairs),) with raw distances (dist_scale=1).
        """
        device = torch.device(self.config.device)
        states_list = []
        goals_list = []
        for src, tgt in pairs:
            source_states = self.training_data.node_states[src]
            state = random.choice(source_states)
            states_list.append(self._flatten_state(state))
            target_atoms = self._node_atoms_dict.get(tgt, set())
            goals_list.append(self.create_atom_vector(target_atoms))

        states_t = torch.FloatTensor(np.array(states_list, dtype=np.float32)).to(device)
        goals_t = torch.FloatTensor(np.array(goals_list, dtype=np.float32)).to(device)

        with torch.no_grad():
            actions, _ = self.actor.sample(states_t, goals_t, deterministic=True)
            q = torch.min(
                self.q1(states_t, actions, goals_t),
                self.q2(states_t, actions, goals_t),
            )
        return np.maximum(0.0, -q.cpu().numpy())  # shape (N,), unscaled

    # ── Gain methods (identical to cmd_v2 / crl_v2) ──────────────────

    def exact_gain(self, source_node: int, target_node: int, L: float | None = None) -> float:
        x, y = source_node, target_node
        d = self.node_pair_graph_dists
        if L is None:
            L = self.estimate_node_distance(x, y)

        U = np.where(np.isfinite(d[:, x]))[0]
        V = np.where(np.isfinite(d[y, :]))[0]

        new_paths = d[U, x][:, None] + L + d[y, V][None, :]
        old_paths = np.minimum(d[np.ix_(U, V)], self.config.max_episode_steps)
        return float(np.sum(np.maximum(0, old_paths - new_paths)))

    def estimate_gain(self, source_node: int, target_node: int, L: float | None = None) -> float:
        x, y = source_node, target_node
        d = np.clip(self.node_pair_graph_dists, 0, self.config.max_episode_steps)
        if L is None:
            L = self.estimate_node_distance(x, y)

        delta_in = d[:, y] - (d[:, x] + L)
        delta_in = delta_in[delta_in > 0]
        if delta_in.size == 0:
            return 0.0
        delta_out = d[x, :] - (L + d[y, :])
        delta_out = delta_out[delta_out > 0]
        if delta_out.size == 0:
            return 0.0
        return float(np.sum(np.minimum(delta_in[:, None], delta_out[None, :])))

    def naive_gain(self, source_node: int, target_node: int, L: float | None = None) -> float:
        if L is None:
            L = self.estimate_node_distance(source_node, target_node)
        d_xy = self.node_pair_graph_dists[source_node, target_node]
        return float(np.clip(d_xy - L, 0, self.config.max_episode_steps))

    def get_gain(self, source_node: int, target_node: int, L: float | None = None) -> float:
        if self.config.gain_method == "exact":
            return self.exact_gain(source_node, target_node, L)
        if self.config.gain_method == "estimate":
            return self.estimate_gain(source_node, target_node, L)
        if self.config.gain_method == "naive":
            return self.naive_gain(source_node, target_node, L)
        raise ValueError(f"Unknown gain_method: {self.config.gain_method}")

    def _update_medoids(self) -> None:
        """Re-elect the medoid state for each node: x* = argmin_{x} estimate_distance(x, S)."""
        print("Updating node medoids...")
        for node_id, states in self.training_data.node_states.items():
            if not states:
                continue
            best_state = states[0]
            best_dist = float("inf")
            for state in states:
                d = self.estimate_distance(state, node_id)
                if d < best_dist:
                    best_dist = d
                    best_state = state
            self._node_medoids[node_id] = best_state
        print("Updated node medoids.")

    def _update_gains(self) -> None:
        print("\nUpdating node pair gains...")

        shortcuts = list(self.training_data.unique_shortcuts)
        if not shortcuts:
            print("No shortcuts — skipping gain update.\n")
            return

        # Single batched forward pass for all shortcuts (raw, dist_scale=1)
        raw_dists = self._batch_estimate_raw_distances(shortcuts)

        if self.config.auto_dist_scale:
            self.config.dist_scale = 1.0
            orig = np.array([
                self.original_node_pair_graph_dists[src, tgt] for src, tgt in shortcuts
            ])
            ratios = raw_dists / (orig + 1e-8)
            valid = np.isfinite(ratios) & (orig > 0)
            if valid.any():
                q = self.config.dist_quantile
                ratio = 1.0 / np.quantile(ratios[valid], q)
                if not np.isinf(ratio) and ratio > 0:
                    self.config.dist_scale = ratio
                else:
                    self.config.dist_scale = 1.0
            print(f"  Auto-set dist_scale={self.config.dist_scale:.4f}")

        # Apply dist_scale and compute gains — no more forward passes
        for (src, tgt), raw_L in zip(shortcuts, raw_dists):
            L = self.config.dist_scale * float(raw_L)
            self.node_pair_gains[src, tgt] = self.get_gain(src, tgt, L)

        print("Updated node pair gains.\n")

    def _update_graph_distances(self, source_node: int, target_node: int) -> None:
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)
        via_xy = d[:, source_node][:, None] + L + d[target_node, :][None, :]
        self.node_pair_graph_dists = np.minimum(d, via_xy)

    def _update_first_edge_dict(self, source_id: int, target_id: int) -> None:
        """Incrementally update first_edge_dict after adding edge (source_id -> target_id).

        Only entries keyed by source_id's atoms can change: no other node's
        first step is affected by a new outgoing edge from source_id.
        """
        if self.first_edge_dict is None:
            return
        source_atoms = frozenset(self.training_data.node_atoms[source_id])
        for node in self.internal_graph.nodes:
            if node.atoms == source_atoms:
                continue
            key = (source_atoms, node.atoms)
            self.first_edge_dict[key] = compute_first_edge(
                self.internal_graph, source_atoms, node.atoms
            )

    def train_continuous(
        self,
        graduate_fn: Any = None,
        success_threshold: float = 0.9,
    ) -> dict[str, Any]:
        """Train with continuous graduation: graduate pairs online as they become reliable.

        Args:
            graduate_fn: Callable(heuristic, source_id, target_id) invoked on each
                graduation. Responsible for adding the operator+skill to virtual_system
                and the corresponding edge to internal_graph. If None, only the
                distance/gain updates are performed (useful for testing).
            success_threshold: Sliding-window success rate required to graduate a pair.
                Should match cfg.heuristic.success_threshold (default 0.9).

        Returns:
            Dictionary with training statistics.
        """
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        # Do NOT reset node_pair_successes — the sliding window persists across epochs.

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        self._update_medoids()
        self._update_gains()

        for epoch in range(self.config.num_epochs_per_round):
            for _ in range(self.config.trajectories_per_epoch):
                trajectory, goal_vec, target_id, source_id = self._collect_trajectory()
                self._add_to_buffer_with_her(trajectory, goal_vec, target_id, source_id)

                # Check if this pair just crossed the graduation threshold
                successes = self.node_pair_successes.get((source_id, target_id), [])
                k = self.config.num_reliability_trials
                if k > 0 and len(successes) >= k and np.mean(successes) > success_threshold:
                    print(f"Graduating pair ({source_id} -> {target_id}) with success rate {np.mean(successes):.2f}")
                    if graduate_fn is not None:
                        graduate_fn(self, source_id, target_id)
                    self._update_graph_distances(source_id, target_id)
                    self._update_first_edge_dict(source_id, target_id)
                    self._update_gains()
                    # Clear success record so the pair must re-earn graduation before re-adding
                    self.node_pair_successes[(source_id, target_id)].clear()

            if (epoch + 1) % self.config.learn_frequency == 0:
                if len(self.replay_buffer) >= self.config.batch_size:
                    for _ in range(self.config.iters_per_epoch):
                        c_loss, a_loss = self._update_networks()
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

            if (epoch + 1) % self.config.medoid_update_frequency == 0:
                self._update_medoids()

            if self.config.sampling_method != "uniform" and (epoch + 1) % self.config.gain_update_frequency == 0:
                self._update_gains()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                c_str = f"{critic_losses[-1]:.4f}" if critic_losses else "N/A"
                a_str = f"{actor_losses[-1]:.4f}" if actor_losses else "N/A"
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs_per_round} | "
                    f"Buffer: {len(self.replay_buffer)} | "
                    f"Critic: {c_str} | Actor: {a_str}"
                )

        self._update_medoids()
        self._update_gains()

        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "buffer_size": len(self.replay_buffer),
        }

    def estimate_weight(self, source_node: int, target_node: int) -> float:
        if source_node == target_node:
            return 0.0
        gain = self.node_pair_gains[source_node, target_node]
        ucb = np.sqrt(
            2 * np.log(max(self.total_samples, 1))
            / (self.node_pair_samples[source_node, target_node] + 1e-8)
        )
        return gain + self.config.ucb_beta * ucb

    # ── Pruning (identical to crl_v2) ────────────────────────────────

    def prune_by_success(
        self, success_threshold: float, max_steps: int
    ) -> "GoalConditionedTrainingData":
        print("Heuristic-Estimated Success Probs of All Shortcuts:")
        pruned_pairs = []
        for (x, y) in self.training_data.unique_shortcuts:
            k = self.config.num_reliability_trials
            successes = self.node_pair_successes.get((x, y), [])
            if k > 0:
                prob = np.mean(successes) if (len(successes) >= k) else 0.0
            else:
                prob = np.mean(successes) if successes else 0.0

            print(f"  ({x} -> {y}) success prob: {prob:.2f}) with {len(successes)} samples")

            if prob > success_threshold:
                # print(f"  Keeping shortcut ({x} -> {y}) with success prob {prob:.2f}")
                pruned_pairs.append((x, y))
                
        return self._build_pruned_data(pruned_pairs)

    def prune(self, max_shortcuts: int) -> "GoalConditionedTrainingData":
        print(f"\nPruning greedily to max_shortcuts={max_shortcuts}")

        print("Estimated distances for all shortcuts:")
        for x, y in self.training_data.unique_shortcuts:
            d = self.estimate_node_distance(x, y)
            print(f"  ({x} -> {y}): {d:.2f}")

        if max_shortcuts is None or max_shortcuts >= len(
            self.training_data.unique_shortcuts
        ):
            return self.training_data

        starts = np.array([x for x, _ in self.training_data.unique_shortcuts])
        ends = np.array([y for _, y in self.training_data.unique_shortcuts])

        curr_gains = self.node_pair_gains.copy()
        curr_dists = self.node_pair_graph_dists.copy()

        pruned_pairs: list[tuple[int, int]] = []
        prev_auto = self.config.auto_dist_scale
        self.config.auto_dist_scale = False

        for i in range(max_shortcuts):
            self._update_gains()
            gains = self.node_pair_gains[starts, ends]
            best = int(np.argmax(gains))
            src, tgt = int(starts[best]), int(ends[best])
            pruned_pairs.append((src, tgt))
            self._update_graph_distances(src, tgt)
            starts = np.delete(starts, best)
            ends = np.delete(ends, best)
            print(f"  Selected shortcut {i+1}: {src} -> {tgt} (gain={gains[best]:.2f})")

        self.config.auto_dist_scale = prev_auto
        self.node_pair_gains = curr_gains
        self.node_pair_graph_dists = curr_dists

        return self._build_pruned_data(pruned_pairs)

    def _build_pruned_data(
        self, pruned_pairs: list[tuple[int, int]]
    ) -> "GoalConditionedTrainingData":
        selected_set = set(pruned_pairs)
        selected_indices = [
            i
            for i, (src, tgt) in enumerate(self.training_data.valid_shortcuts)
            if (src, tgt) in selected_set
        ]
        print(f"  ({len(selected_indices)} state-node pairs)")

        original_info = self.training_data.config.get("shortcut_info", [])
        pruned_info = (
            [original_info[i] for i in selected_indices] if original_info else []
        )

        return GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[
                self.training_data.current_atoms[i] for i in selected_indices
            ],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[
                self.training_data.valid_shortcuts[i] for i in selected_indices
            ],
            unique_shortcuts=pruned_pairs,
            node_states=self.training_data.node_states,
            node_atoms=self.training_data.node_atoms,
            graph=self.training_data.graph,
            config={
                **self.training_data.config,
                "shortcut_info": pruned_info,
                "pruning_method": "sac_v2",
                "threshold": self.config.threshold,
            },
        )

    # ── Atom encoding (same as crl_v2) ────────────────────────────────

    def _get_atom_index(self, atom_str: str) -> int:
        if atom_str not in self.atom_to_index:
            assert self._next_atom_index < self.config.max_atom_size, (
                f"Atom index overflow — increase max_atom_size "
                f"(currently {self.config.max_atom_size})"
            )
            self.atom_to_index[atom_str] = self._next_atom_index
            self._next_atom_index += 1
        return self.atom_to_index[atom_str]

    def create_atom_vector(self, atoms: set) -> NDArray:
        vec = np.zeros(self.config.max_atom_size, dtype=np.float32)
        for atom in atoms:
            vec[self._get_atom_index(str(atom))] = 1.0
        return vec

    def _flatten_state(self, state: "ObsType") -> NDArray:
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32)
        return np.array(state).flatten().astype(np.float32)

    def _find_node_for_goal_vec(self, goal_vec: NDArray) -> int:
        """Return the node ID whose atom set matches goal_vec, or -1 if not found."""
        active_indices = frozenset(np.where(goal_vec > 0.5)[0].tolist())
        return self._atom_indices_to_node_id.get(active_indices, -1)

    def _find_node_for_atoms(self, atoms: set) -> int | None:
        for node_id, node_atoms in self._node_atoms_dict.items():
            if node_atoms == atoms:
                return node_id
        return None

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.q1.state_dict(), os.path.join(path, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(path, "q2.pt"))
        with open(os.path.join(path, "atom_index.pkl"), "wb") as f:
            pickle.dump(
                {
                    "atom_to_index": self.atom_to_index,
                    "next_index": self._next_atom_index,
                },
                f,
            )
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        print(f"Saved SAC V2 heuristic to {path}")

    def load(self, path: str) -> None:
        device = torch.device(self.config.device)
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"), map_location=device)
        )
        self.q1.load_state_dict(
            torch.load(os.path.join(path, "q1.pt"), map_location=device)
        )
        self.q2.load_state_dict(
            torch.load(os.path.join(path, "q2.pt"), map_location=device)
        )
        with open(os.path.join(path, "atom_index.pkl"), "rb") as f:
            data = pickle.load(f)
            self.atom_to_index = data["atom_to_index"]
            self._next_atom_index = data["next_index"]
        print(f"Loaded SAC V2 heuristic from {path}")
