"""Goal-conditioned distance heuristic V2 using Distributional SAC (DSAC) + HER.

Implements Soft Actor-Critic with a distributional critic and Hindsight Experience
Replay for goal-conditioned distance estimation.  Follows SoRB (Eysenbach et al.,
2019): reward is -1 everywhere except at the goal node (reward = 0), γ = 1, and
the critic predicts a probability distribution over ``num_bins`` distance bins
rather than a scalar Q-value.

The distributional Bellman update is a right-shift of the bin distribution:
- Goal reached  → all mass in bin 0
- Otherwise     → mass at bin i shifts to bin i+1; the catch-all bin (last bin)
                   absorbs overflow so that unreachable goals have a well-defined
                   representation.

Distance estimate = E[bin distance] = Σ_i bin_value_i * p_i

Using a pessimistic (max-distance) ensemble of two distributional critics avoids
"wormhole" overconfidence for unseen (state, goal) pairs.

Key components:
- DistributionalQNetwork: Q(state, action, goal_atoms) → (num_bins,) distribution
- ContinuousActor: π(state, goal_atoms) → action (Gaussian + tanh)
- SACReplayBuffer: (s, a, r, s', done, goal) transitions + HER
- Distributional Bellman update: right-shift + KL divergence loss
- SAC actor: minimize expected distance + entropy penalty (γ=1)
- Auto-entropy tuning: log_alpha updated to maintain target entropy
"""


import copy
import os
import pickle
import random
import time
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
    DistributionalQNetwork,
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
class DSACv2HeuristicConfig:
    """Configuration for distributional SAC goal-conditioned distance heuristic."""

    wandb_enabled: bool = False
    threshold: float = 0.05
    beta: float = 1.0

    # Distributional critic bins
    num_bins: int = 51    # Number of distance bins (including catch-all)
    max_bin: float = 500  # Maximum distance represented; bin values = linspace(0, max_bin, num_bins)

    # Network architecture
    hidden_dims: list[int] | None = None  # Q-network hidden layers; default [256, 256]
    actor_hidden_dims: list[int] | None = None  # Actor hidden layers; falls back to hidden_dims

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # SAC hyperparameters (γ=1 is implicit — no gamma field)
    batch_size: int = 256
    buffer_size: int = 100_000
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
    gain_update_frequency: int = 10  # Update UCB gains every N epochs
    medoid_update_frequency: int = 10  # Re-elect node medoids every N epochs
    iters_per_epoch: int = 50  # Gradient steps per epoch
    grad_clip: float = 1.0
    log_alpha_min: float = -5.0

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
    residual_lam: float = 1.0  # Scaling factor on base action in residual actor (1.0=add base, 0.0=ignore base)
    max_atom_size: int = 50

    # Success tracking
    num_reliability_trials: int = 10
    node_distance_samples: int = 10  # States to average over in estimate_node_distance
    device: str = "cuda"


# ── Replay buffer ─────────────────────────────────────────────────────

# (state_flat, action, reward, next_state_flat, done, achieved_vec, next_achieved_vec, obs_raw, base_action)
_TrajStep = tuple[NDArray, NDArray, float, NDArray, bool, NDArray, NDArray, Any, NDArray]


class SACReplayBuffer:
    """Circular replay buffer storing (s, a, r, s', done, goal, obs_raw, goal_node_id, source_node_id, base_action)."""

    def __init__(self, max_size: int):
        self.buffer: deque[tuple] = deque(maxlen=max_size)

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
        base_action: NDArray | None = None,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, goal, obs_raw, goal_node_id, source_node_id, base_action))

    def sample(
        self, batch_size: int
    ) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals, obs_raws, goal_node_ids, source_node_ids, base_actions = zip(*batch)
        ba_array = np.array(base_actions, dtype=np.float32)
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
            ba_array,
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Main heuristic class ──────────────────────────────────────────────


class DSACv2Heuristic(BaseHeuristic):
    """Distributional SAC with HER for goal-conditioned distance estimation.

    The distributional critic predicts P(distance = bin_i | s, a, g) and uses
    a right-shift Bellman update with a catch-all bin for large / unreachable
    distances.  Distance estimates = E[bin distance] from a pessimistic ensemble.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
        config: DSACv2HeuristicConfig | None = None,
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
        self.internal_graph = training_data.graph
        self.virtual_system = copy.deepcopy(system)
        # self.virtual_system = None

        if config is None:
            config = DSACv2HeuristicConfig()
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.config = config

        # Bin values: linspace(0, max_bin, num_bins)
        self.bin_values = np.linspace(0, config.max_bin, config.num_bins, dtype=np.float32)

        # Determine state dimension
        sample_states = next(iter(training_data.node_states.values()))
        self.state_dim = self._flatten_state(sample_states[0]).shape[0]

        env = system.env
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DSACv2Heuristic requires a continuous (Box) action space.")
        self.action_dim = int(np.prod(env.action_space.shape))
        self._action_low = env.action_space.low.astype(np.float32)
        self._action_high = env.action_space.high.astype(np.float32)

        self.num_nodes = len(training_data.node_states)
        self._node_atoms_dict = dict(training_data.node_atoms)

        # Dynamic atom indexing
        self.atom_to_index: dict[str, int] = {}
        self._next_atom_index = 0
        for atoms in self._node_atoms_dict.values():
            for atom in atoms:
                self._get_atom_index(str(atom))

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
        self._graduation_cooldown: dict[tuple[int, int], int] = {}
        for (i, j) in self.valid_pairs:
            k = self.config.num_reliability_trials
            if k > 0:
                self.node_pair_successes[(i, j)] = deque(maxlen=k)
            else:
                self.node_pair_successes[(i, j)] = []

        print(f"DSAC V2: state_dim={self.state_dim}, action_dim={self.action_dim}, "
              f"num_nodes={self.num_nodes}, num_bins={config.num_bins}, max_bin={config.max_bin}")

        self._init_networks()
        self.replay_buffer = SACReplayBuffer(config.buffer_size)
        self.training_step = 0

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
        num_bins = self.config.num_bins

        if self.config.residualize:
            self.actor = ResidualContinuousActor(
                self.state_dim, self.action_dim, atom_dim, actor_hidden_dims,
                action_low=self._action_low, action_high=self._action_high,
                lam=self.config.residual_lam,
            ).to(device)
        else:
            self.actor = ContinuousActor(
                self.state_dim, self.action_dim, atom_dim, actor_hidden_dims,
                action_low=self._action_low, action_high=self._action_high,
            ).to(device)

        self.q1 = DistributionalQNetwork(
            self.state_dim, self.action_dim, atom_dim, num_bins, hidden_dims
        ).to(device)
        self.q2 = DistributionalQNetwork(
            self.state_dim, self.action_dim, atom_dim, num_bins, hidden_dims
        ).to(device)
        self.q1_target = DistributionalQNetwork(
            self.state_dim, self.action_dim, atom_dim, num_bins, hidden_dims
        ).to(device)
        self.q2_target = DistributionalQNetwork(
            self.state_dim, self.action_dim, atom_dim, num_bins, hidden_dims
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

        self.target_entropy = float(-self.action_dim)
        self.log_alpha = torch.tensor(
            [np.log(self.config.init_temperature)],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.actor_lr)

    # ── System update (multi-round) ───────────────────────────────────

    def update_system(
        self,
        new_system: "ImprovisationalTAMPSystem",
        new_graph: "PlanningGraph",
        new_graph_distances: dict[tuple[int, int], float],
        new_first_edge_dict: dict[
            tuple[frozenset["GroundAtom"], frozenset["GroundAtom"]], PlanningGraphEdge
        ],
        reset_actor: bool = False,
    ) -> None:
        self.system = new_system
        self.internal_graph = new_graph
        self.graph_distances = new_graph_distances
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in new_graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist
        self.first_edge_dict = new_first_edge_dict

        device = torch.device(self.config.device)
        actor_hidden_dims = self.config.actor_hidden_dims or (self.config.hidden_dims or [256, 256])
        if self.config.residualize:
            self.actor = ResidualContinuousActor(
                self.state_dim, self.action_dim, self.config.max_atom_size, actor_hidden_dims,
                action_low=self._action_low, action_high=self._action_high,
                lam=self.config.residual_lam,
            ).to(device)
        else:
            self.actor = ContinuousActor(
                self.state_dim, self.action_dim, self.config.max_atom_size, actor_hidden_dims,
                action_low=self._action_low, action_high=self._action_high,
            ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

    # ── Training ──────────────────────────────────────────────────────

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        return {"critic_losses": [], "actor_losses": [], "buffer_size": 0}

    def train_one_round(self, checkpoint_callback: Any = None) -> dict[str, Any]:
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        k = self.config.num_reliability_trials
        for key in self.node_pair_successes:
            self.node_pair_successes[key] = deque(maxlen=k) if k > 0 else []

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        self._update_gains(normalize=True)

        for epoch in range(self.config.num_epochs_per_round):
            for _ in range(self.config.trajectories_per_epoch):
                trajectory, goal_vec, target_id, source_id = self._collect_trajectory()
                self._add_to_buffer_with_her(trajectory, goal_vec, target_id, source_id)

            if (epoch + 1) % self.config.learn_frequency == 0:
                if len(self.replay_buffer) >= self.config.batch_size:
                    for _ in range(self.config.iters_per_epoch):
                        c_loss, a_loss = self._update_networks()
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

            if self.config.sampling_method != "uniform" and (epoch + 1) % self.config.gain_update_frequency == 0:
                self._update_gains(normalize=True)

            if (epoch + 1) % 1 == 0 or epoch == 0:
                c_str = f"{critic_losses[-1]:.4f}" if critic_losses else "N/A"
                a_str = f"{actor_losses[-1]:.4f}" if actor_losses else "N/A"
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs_per_round} | "
                    f"Buffer: {len(self.replay_buffer)} | "
                    f"Critic: {c_str} | Actor: {a_str}"
                )

            if checkpoint_callback is not None and (epoch + 1) % 100 == 0:
                checkpoint_callback()

        self._update_gains()

        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "buffer_size": len(self.replay_buffer),
        }

    def _collect_trajectory(self) -> tuple[list[_TrajStep], NDArray, int, int]:
        node_ids = list(self.training_data.node_states.keys())

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
            flat_idx = int(self.rng.choice(self.num_nodes * self.num_nodes, p=probs.flatten()))
            source_id, target_id = flat_idx // self.num_nodes, flat_idx % self.num_nodes
        else:
            raise ValueError(f"Unknown sampling_method: {self.config.sampling_method}")

        self.node_pair_samples[source_id, target_id] += 1
        self.total_samples += 1

        source_states = self.training_data.node_states[source_id]
        source_state = source_states[self.rng.integers(len(source_states))]

        target_atoms = self._node_atoms_dict.get(target_id, set())
        goal_vec = self.create_atom_vector(target_atoms)

        env = self.system.env
        current_state, _ = env.reset_from_state(source_state)
        trajectory: list[_TrajStep] = []
        successful = False

        achieved_atoms = self.system.perceiver.step(current_state)
        achieved_vec = self.create_atom_vector(achieved_atoms)
        state_flat = self._flatten_state(current_state)

        for step in range(self.config.max_episode_steps):
            # Get base action from first_edge_dict for current state
            base_action = self._get_base_action(current_state, target_id)

            action = self._select_action(state_flat, goal_vec, deterministic=False, base_action=base_action)

            next_state, _, terminated, truncated, _ = env.step(action)
            next_flat = self._flatten_state(next_state)
            next_atoms = self.system.perceiver.step(next_state)
            next_achieved_vec = self.create_atom_vector(next_atoms)

            goal_reached = target_atoms <= next_atoms
            reward = 0.0 if goal_reached else -1.0
            done = terminated or truncated or goal_reached

            if goal_reached:
                successful = True

            trajectory.append(
                (state_flat, action, reward, next_flat, done, achieved_vec, next_achieved_vec, current_state, base_action)
            )

            if done:
                break

            current_state = next_state
            achieved_atoms = next_atoms
            achieved_vec = next_achieved_vec
            state_flat = next_flat

        if successful:
            self.node_pair_successes[(source_id, target_id)].append(1)
        else:
            self.node_pair_successes[(source_id, target_id)].append(0)

        return trajectory, goal_vec, target_id, source_id

    def _add_to_buffer_with_her(
        self, trajectory: list[_TrajStep], goal_vec: NDArray, target_id: int, source_id: int
    ) -> None:
        if not trajectory:
            return

        T = len(trajectory)

        # Real transitions — use per-step source node (not trajectory-level source_id)
        for step_data in trajectory:
            s, a, r, s_, done, achieved_vec, _, obs_raw = step_data[:8]
            ba = step_data[8] if len(step_data) > 8 else np.zeros(self.action_dim, dtype=np.float32)
            step_source_id = self._find_node_for_goal_vec(achieved_vec)
            if step_source_id < 0:
                step_source_id = source_id
            self.replay_buffer.add(s, a, r, s_, done, goal_vec, obs_raw, target_id, step_source_id, ba)

        # HER transitions
        for t in range(T):
            step_data = trajectory[t]
            s, a, _, s_, done, achieved_vec, next_achieved_vec, obs_raw = step_data[:8]
            ba = step_data[8] if len(step_data) > 8 else np.zeros(self.action_dim, dtype=np.float32)
            step_source_id = self._find_node_for_goal_vec(achieved_vec)
            if step_source_id < 0:
                step_source_id = source_id

            if self.config.her_strategy == "final":
                her_indices = [T - 1]
            else:
                n_future = min(self.config.her_n_sampled_goal, T - t - 1)
                her_indices = random.sample(range(t + 1, T), n_future) if n_future > 0 else []

            for t_her in her_indices:
                her_goal_vec = trajectory[t_her][5]
                her_goal_node_id = self._find_node_for_goal_vec(her_goal_vec)

                goal_atoms_active = her_goal_vec > 0.5
                her_r = 0.0 if np.all(next_achieved_vec[goal_atoms_active] > 0.5) else -1.0
                her_done = her_r == 0.0 or done
                self.replay_buffer.add(
                    s, a, her_r, s_, her_done, her_goal_vec, obs_raw, her_goal_node_id, step_source_id, ba
                )

    def _get_base_action(self, obs: "ObsType", target_node: int) -> NDArray:
        """Get the base skill action for (obs, target_node), or zeros if unavailable."""
        if not self.config.residualize or self.first_edge_dict is None:
            return np.zeros(self.action_dim, dtype=np.float32)

        start_atoms = self.system.perceiver.step(obs)
        goal_atoms = self._node_atoms_dict.get(target_node, set())
        edge = self.first_edge_dict.get(
            (frozenset(start_atoms), frozenset(goal_atoms)), None
        )
        if edge is not None:
            operator = edge.operator
            skills = [sk for sk in self.virtual_system.skills if sk.can_execute(operator)]
            if skills:
                skill = skills[0]
                skill.reset(operator)
                try:
                    raw = skill.get_action(obs)
                    if raw is not None:
                        ba = np.array(raw, dtype=np.float32).flatten()
                        if ba.shape[0] == self.action_dim:
                            return ba
                except Exception:
                    pass
        return np.zeros(self.action_dim, dtype=np.float32)
        
    # def _get_base_action(self, obs: "ObsType", target_node: int) -> NDArray:
    #     """Get the base skill action for (obs, target_node), or zeros if unavailable."""
    #     if not self.config.residualize or self.first_edge_dict is None:
    #         return np.zeros(self.action_dim, dtype=np.float32)

    #     start_atoms = self.system.perceiver.step(obs)
    #     goal_atoms = self._node_atoms_dict.get(target_node, set())
    #     edge = self.first_edge_dict.get(
    #         (frozenset(start_atoms), frozenset(goal_atoms)), None
    #     )
    #     if edge is not None:
    #         operator = edge.operator
    #         skills = [sk for sk in self.virtual_system.skills if sk.can_execute(operator)]
    #         if skills:
    #             skill = skills[0]
    #             skill.reset(operator)
    #             t0 = time.time()
    #             try:
    #                 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
    #                     raw = ex.submit(skill.get_action, obs).result(timeout=1)
    #                 elapsed = time.time() - t0
    #                 if raw is not None:
    #                     ba = np.array(raw, dtype=np.float32).flatten()
    #                     if ba.shape[0] == self.action_dim:
    #                         print(f"[BASE_ACTION] node {target_node}: OK in {elapsed:.3f}s", flush=True)
    #                         return ba
    #                 print(f"[BASE_ACTION] node {target_node}: returned None in {elapsed:.3f}s", flush=True)
    #             except concurrent.futures.TimeoutError:
    #                 print(f"[TIMEOUT] skill.get_action hung for node {target_node} after {time.time()-t0:.3f}s, returning zeros", flush=True)
    #             except Exception as e:
    #                 print(f"[BASE_ACTION] skill.get_action failed for node {target_node} after {time.time()-t0:.3f}s: {e}", flush=True)
    #     return np.zeros(self.action_dim, dtype=np.float32)

    def _select_action(
        self, state_flat: NDArray, goal_vec: NDArray, deterministic: bool = False,
        base_action: NDArray | None = None,
    ) -> NDArray:
        device = torch.device(self.config.device)
        with torch.no_grad():
            s_t = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
            g_t = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)
            if isinstance(self.actor, ResidualContinuousActor) and base_action is not None:
                base_t = torch.FloatTensor(base_action).unsqueeze(0).to(device)
                action, _ = self.actor.sample(s_t, g_t, base_t, deterministic=deterministic)
            else:
                action, _ = self.actor.sample(s_t, g_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    # ── Distributional helpers ────────────────────────────────────────

    def _distributional_target(
        self, next_probs: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute distributional Bellman target via right-shift (γ=1, r=-1).

        Args:
            next_probs: (B, num_bins) — target network probabilities for next state
            dones:      (B, 1)        — 1.0 if goal reached, 0.0 otherwise

        Returns:
            target: (B, num_bins) — target distribution (no_grad assumed by caller)
        """
        # Right-shift: mass at bin i moves to bin i+1
        shifted = torch.zeros_like(next_probs)
        shifted[:, 1:] = next_probs[:, :-1]
        shifted[:, -1] += next_probs[:, -1]  # catch-all absorbs its own mass

        # Goal reached: all mass in bin 0
        goal_dist = torch.zeros_like(next_probs)
        goal_dist[:, 0] = 1.0

        return dones * goal_dist + (1.0 - dones) * shifted

    def _expected_distance(self, probs: torch.Tensor) -> torch.Tensor:
        """Expected distance in steps from bin probabilities.

        Args:
            probs: (B, num_bins)

        Returns:
            (B,) expected distance
        """
        bv = torch.FloatTensor(self.bin_values).to(probs.device)
        return (probs * bv).sum(dim=-1)

    # ── Network update ────────────────────────────────────────────────

    def _update_networks(self) -> tuple[float, float]:
        """One DSAC gradient step. Returns (critic_loss, actor_loss)."""
        device = torch.device(self.config.device)

        states, actions, rewards, next_states, dones, goals, obs_raws, goal_node_ids, source_node_ids, cached_base_actions = (
            self.replay_buffer.sample(self.config.batch_size)
        )
        s = torch.FloatTensor(states).to(device)
        a = torch.FloatTensor(actions).to(device)
        r = torch.FloatTensor(rewards).to(device)   # (B, 1); 0.0 at goal, -1.0 elsewhere
        s_ = torch.FloatTensor(next_states).to(device)
        d = torch.FloatTensor(dones).to(device)     # (B, 1); 1.0 if done
        g = torch.FloatTensor(goals).to(device)

        alpha = self.log_alpha.exp().detach()

        # Use cached base actions from buffer
        if isinstance(self.actor, ResidualContinuousActor):
            base_actions_tensor = torch.FloatTensor(cached_base_actions).to(device)
        else:
            base_actions_tensor = None

        # ── Critic update (distributional Bellman + KL loss) ─────────
        with torch.no_grad():
            if base_actions_tensor is not None:
                next_a, _ = self.actor.sample(s_, g, base_actions_tensor)
            else:
                next_a, _ = self.actor.sample(s_, g)

            # Pessimistic ensemble: pick distribution with higher expected distance
            next_probs_1 = self.q1_target.get_probs(s_, next_a, g)
            next_probs_2 = self.q2_target.get_probs(s_, next_a, g)
            dist_1 = self._expected_distance(next_probs_1)
            dist_2 = self._expected_distance(next_probs_2)
            # Select per-sample: use the critic predicting larger distance
            use_q1 = (dist_1 >= dist_2).unsqueeze(-1).expand_as(next_probs_1)
            next_probs = torch.where(use_q1, next_probs_1, next_probs_2)

            target = self._distributional_target(next_probs, d)  # (B, num_bins)

        # Cross-entropy loss: -Σ target_i * log(pred_i)  ≡  KL(target ‖ pred) up to constant
        log_p1 = F.log_softmax(self.q1(s, a, g), dim=-1)
        log_p2 = F.log_softmax(self.q2(s, a, g), dim=-1)
        critic_loss = (
            -(target * log_p1).sum(dim=-1).mean()
            + -(target * log_p2).sum(dim=-1).mean()
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.grad_clip,
        )
        self.critic_optimizer.step()

        # ── Actor update (minimize expected distance + entropy penalty) ──
        if base_actions_tensor is not None:
            new_a, log_p = self.actor.sample(s, g, base_actions_tensor)
        else:
            new_a, log_p = self.actor.sample(s, g)

        probs_1 = self.q1.get_probs(s, new_a, g)
        probs_2 = self.q2.get_probs(s, new_a, g)
        # Pessimistic: use max expected distance from ensemble
        expected_dist = torch.maximum(
            self._expected_distance(probs_1),
            self._expected_distance(probs_2),
        )
        # Minimize distance + entropy penalty: L = E[dist] + alpha * log_pi
        actor_loss = (expected_dist + alpha * log_p).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        # ── Entropy coefficient update ────────────────────────────────
        if self.config.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_p.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.config.log_alpha_min)

        # ── Soft target update ────────────────────────────────────────
        self.training_step += 1
        if self.training_step % self.config.target_update_interval == 0:
            self._soft_update_targets()

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
        obs_raws: list | None = None,
    ) -> NDArray:
        batch_size = len(source_node_ids)
        base_actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)

        if not self.config.residualize or self.first_edge_dict is None:
            return base_actions

        skill_cache: dict[Any, Any] = {}

        for i in range(batch_size):
            if source_node_ids[i] < 0 or goal_node_ids[i] < 0:
                continue
            start_atoms = self._node_atoms_dict.get(int(source_node_ids[i]), set())
            goal_atoms = self._node_atoms_dict.get(int(goal_node_ids[i]), set())
            edge = self.first_edge_dict.get((frozenset(start_atoms), frozenset(goal_atoms)), None)
            if edge is None:
                continue
            operator = edge.operator
            if operator not in skill_cache:
                skills = [sk for sk in self.virtual_system.skills if sk.can_execute(operator)]
                skill_cache[operator] = skills[0] if skills else None
            skill = skill_cache[operator]
            if skill is None:
                continue
            skill.reset(operator)
            obs = obs_raws[i] if obs_raws is not None else None
            action = None
            if obs is not None:
                try:
                    action = skill.get_action(obs)
                except Exception as e:
                    print(f"[BASE_ACTION] batch skill.get_action failed for pair ({source_node_ids[i]}->{goal_node_ids[i]}): {e}", flush=True)
            if action is not None:
                base_actions[i] = np.array(action, dtype=np.float32)

        return base_actions

    # ── Action + distance interface ───────────────────────────────────

    def get_action(self, obs: "ObsType", target_node: int) -> NDArray:
        state_flat = self._flatten_state(obs)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        goal_vec = self.create_atom_vector(target_atoms)
        base_action = self._get_base_action(obs, target_node)
        return self._select_action(state_flat, goal_vec, deterministic=True, base_action=base_action)

    def estimate_distance(self, source_state: "ObsType", target_node: int) -> float:
        """Pessimistic expected distance from distributional critics."""
        state_flat = self._flatten_state(source_state)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        goal_vec = self.create_atom_vector(target_atoms)

        device = torch.device(self.config.device)
        with torch.no_grad():
            s_t = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
            g_t = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)
            action, _ = self.actor.sample(s_t, g_t, deterministic=True)
            probs_1 = self.q1.get_probs(s_t, action, g_t)
            probs_2 = self.q2.get_probs(s_t, action, g_t)
            dist_1 = self._expected_distance(probs_1)
            dist_2 = self._expected_distance(probs_2)
        return torch.maximum(dist_1, dist_2).item()

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        # Multi-state averaging estimation:
        source_states = self.training_data.node_states.get(source_node, [])
        if not source_states:
            return float("inf")
        k = self.config.node_distance_samples
        samples = random.sample(source_states, min(k, len(source_states)))
        dists = [self.estimate_distance(s, target_node) for s in samples]
        return self.config.dist_scale * float(np.mean(dists))

    def _batch_estimate_raw_distances(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        """Batch pessimistic expected distances for all (src, tgt) pairs.

        For each pair, samples node_distance_samples source states and averages
        the distributional distance estimates across them to reduce variance.
        """
        device = torch.device(self.config.device)
        k = self.config.node_distance_samples

        # Build batched inputs: k samples per pair
        states_list = []
        goals_list = []
        for src, tgt in pairs:
            source_states = self.training_data.node_states[src]
            samples = random.sample(source_states, min(k, len(source_states)))
            target_atoms = self._node_atoms_dict.get(tgt, set())
            goal_vec = self.create_atom_vector(target_atoms)
            for s in samples:
                states_list.append(self._flatten_state(s))
                goals_list.append(goal_vec)

        states_t = torch.FloatTensor(np.array(states_list, dtype=np.float32)).to(device)
        goals_t = torch.FloatTensor(np.array(goals_list, dtype=np.float32)).to(device)

        with torch.no_grad():
            actions, _ = self.actor.sample(states_t, goals_t, deterministic=True)
            probs_1 = self.q1.get_probs(states_t, actions, goals_t)
            probs_2 = self.q2.get_probs(states_t, actions, goals_t)
            dist_1 = self._expected_distance(probs_1)
            dist_2 = self._expected_distance(probs_2)
        raw_all = torch.maximum(dist_1, dist_2).cpu().numpy()

        # Average over the k samples per pair
        result = np.empty(len(pairs), dtype=np.float32)
        idx = 0
        for i, (src, _) in enumerate(pairs):
            n_samples = min(k, len(self.training_data.node_states[src]))
            result[i] = raw_all[idx:idx + n_samples].mean()
            idx += n_samples
        return result

    # ── Gain methods ─────────────────────────────────────────────────

    def exact_gain(self, source_node: int, target_node: int, L: float | None = None) -> float:
        x, y = source_node, target_node
        d = self.node_pair_graph_dists
        if L is None:
            L = self.estimate_node_distance(x, y)

        L_old = d[x, y]
        U = np.where(np.isfinite(d[:, x]))[0]
        V = np.where(np.isfinite(d[y, :]))[0]

        new_paths = d[U, x][:, None] + L + d[y, V][None, :]
        old_paths = np.minimum(d[np.ix_(U, V)], self.config.max_episode_steps)
        old_paths = np.minimum(old_paths, d[U, x][:, None] + L_old + d[y, V][None, :])
        savings = np.maximum(0, old_paths - new_paths)
        total_gain = float(np.sum(savings))

        # ── Debug: dump contributors when gain looks suspicious ──
        debug = getattr(self, "_debug_exact_gain", False)
        if debug:
            d_xy_direct = d[x, y]
            print(
                f"[EXACT_GAIN] x={x} y={y} L={L:.3f} d[x,y]={d_xy_direct} "
                f"|U|={len(U)} |V|={len(V)} total_gain={total_gain:.3f}",
                flush=True,
            )
            if total_gain > 0 and savings.size > 0:
                # Top 10 contributing (u, v) pairs
                flat_idx = np.argsort(savings.ravel())[::-1][:10]
                for fi in flat_idx:
                    ui, vi = np.unravel_index(fi, savings.shape)
                    s = savings[ui, vi]
                    if s <= 0:
                        break
                    u_node = int(U[ui])
                    v_node = int(V[vi])
                    print(
                        f"  ({u_node}->{v_node}): old={old_paths[ui,vi]:.1f} "
                        f"new={new_paths[ui,vi]:.1f} (d[{u_node},{x}]={d[u_node,x]:.1f} "
                        f"+ L={L:.1f} + d[{y},{v_node}]={d[y,v_node]:.1f}) save={s:.2f}",
                        flush=True,
                    )

        return total_gain
    
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

    def _update_gains(self, normalize: bool = False) -> None:
        shortcuts = list(self.training_data.unique_shortcuts)
        if not shortcuts:
            return

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

        for (src, tgt), raw_L in zip(shortcuts, raw_dists):
            L = self.config.dist_scale * float(raw_L)
            self.node_pair_gains[src, tgt] = self.get_gain(src, tgt, L)

        if normalize:
            # Normalize gains to [0, 1] so UCB beta is on the same scale as gains.
            max_gain = float(np.max(self.node_pair_gains))
            if max_gain > 0:
                self.node_pair_gains = self.node_pair_gains / max_gain
                print(f"  Normalized gains by max={max_gain:.2f}")

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
        checkpoint_callback: Any = None,
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

        self._update_gains(normalize=True)

        for epoch in range(self.config.num_epochs_per_round):
            self._current_epoch = epoch
            for _ in range(self.config.trajectories_per_epoch):
                trajectory, goal_vec, target_id, source_id = self._collect_trajectory()
                self._add_to_buffer_with_her(trajectory, goal_vec, target_id, source_id)

                pair = (source_id, target_id)
                if self._graduation_cooldown.get(pair, 0) > 0:
                    self._graduation_cooldown[pair] -= 1
                else:
                    successes = self.node_pair_successes.get(pair, [])
                    k = self.config.num_reliability_trials
                    if k > 0 and len(successes) >= k and np.mean(successes) > success_threshold:
                        print(f"Graduating pair ({source_id} -> {target_id}) with success rate {np.mean(successes):.2f}")
                        if graduate_fn is not None:
                            graduate_fn(self, source_id, target_id)
                        self._update_graph_distances(source_id, target_id)
                        self._update_first_edge_dict(source_id, target_id)
                        self._update_gains(normalize=True)
                        self._graduation_cooldown[pair] = k

            if (epoch + 1) % self.config.learn_frequency == 0:
                if len(self.replay_buffer) >= self.config.batch_size:
                    for _ in range(self.config.iters_per_epoch):
                        c_loss, a_loss = self._update_networks()
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

            if self.config.sampling_method != "uniform" and (epoch + 1) % self.config.gain_update_frequency == 0:
                self._update_gains(normalize=True)

            if (epoch + 1) % 1 == 0 or epoch == 0:
                c_str = f"{critic_losses[-1]:.4f}" if critic_losses else "N/A"
                a_str = f"{actor_losses[-1]:.4f}" if actor_losses else "N/A"
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs_per_round} | "
                    f"Buffer: {len(self.replay_buffer)} | "
                    f"Critic: {c_str} | Actor: {a_str}"
                )

            if checkpoint_callback is not None and (epoch + 1) % 100 == 0:
                checkpoint_callback()
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

    # ── Pruning ───────────────────────────────────────────────────────

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
                pruned_pairs.append((x, y))
        return self._build_pruned_data(pruned_pairs)

    def estimate_probability(self, source_node: int, target_node: int, use_multi_rl: bool = True) -> float:
        """Estimate success probability for a shortcut.

        If use_multi_rl=True, uses Brownian motion argument from estimated distance.
        If use_multi_rl=False, uses empirical success rate from training (if available).
        """
        if not use_multi_rl:
            k = self.config.num_reliability_trials
            trials = self.node_pair_successes.get((source_node, target_node), [])
            if len(trials) >= k > 0:
                return float(np.mean(trials))
            return 0.0
        est_dist = self.estimate_node_distance(source_node, target_node)
        if est_dist <= 0:
            p_rr = 1.0
        else:
            p_rr = np.clip(np.exp(-est_dist**2 / (2 * self.config.max_episode_steps)), 0, 1)
        k = np.log(0.5) / np.log(1 - 0.05)  # threshold = 0.05
        return 1 - (1 - p_rr)**k

    def prune(self, max_shortcuts: int, use_multi_rl: bool = False) -> "GoalConditionedTrainingData":
        print(f"\nPruning greedily to max_shortcuts={max_shortcuts}")
        # Calibrate dist_scale before pruning (important when loading from checkpoint)
        self._update_gains()

        print("Estimated distances for all shortcuts:")
        for x, y in self.training_data.unique_shortcuts:
            d = self.estimate_node_distance(x, y)
            print(f"  ({x} -> {y}): {d:.2f}")

        if max_shortcuts is None or max_shortcuts >= len(self.training_data.unique_shortcuts):
            return self.training_data

        starts = np.array([x for x, _ in self.training_data.unique_shortcuts])
        ends = np.array([y for _, y in self.training_data.unique_shortcuts])

        curr_gains = self.node_pair_gains.copy()
        curr_dists = self.node_pair_graph_dists.copy()  # post-graduation dists to restore
        self.node_pair_graph_dists = self.original_node_pair_graph_dists.copy()

        pruned_pairs: list[tuple[int, int]] = []
        prev_auto = self.config.auto_dist_scale
        self.config.auto_dist_scale = False

        for i in range(max_shortcuts):
            self._update_gains()
            gains = self.node_pair_gains[starts, ends]

            # Compute success probabilities
            probs = np.zeros(len(starts))
            for j, (s, t) in enumerate(zip(starts, ends)):
                probs[j] = self.estimate_probability(int(s), int(t), use_multi_rl=use_multi_rl)

            if np.any((probs * gains) > 0):
                scores = probs * gains
            else:
                scores = gains

            best = int(np.argmax(scores))
            src, tgt = int(starts[best]), int(ends[best])
            p = probs[best]
            pruned_pairs.append((src, tgt))
            self._update_graph_distances(src, tgt)
            starts = np.delete(starts, best)
            ends = np.delete(ends, best)
            print(f"  Selected shortcut {i+1}: {src} -> {tgt} (gain={gains[best]:.2f}, p={p:.2f}, score={scores[best]:.2f})")

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
        pruned_info = [original_info[i] for i in selected_indices] if original_info else []

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
                "shortcut_info": pruned_info,
                "pruning_method": "dsac_v2",
                "threshold": self.config.threshold,
            },
        )

    # ── Atom encoding ─────────────────────────────────────────────────

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
            pickle.dump({"atom_to_index": self.atom_to_index, "next_index": self._next_atom_index}, f)
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        print(f"Saved DSAC V2 heuristic to {path}")

    def load(self, path: str) -> None:
        device = torch.device(self.config.device)
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=device))
        self.q1.load_state_dict(torch.load(os.path.join(path, "q1.pt"), map_location=device))
        self.q2.load_state_dict(torch.load(os.path.join(path, "q2.pt"), map_location=device))
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        with open(os.path.join(path, "atom_index.pkl"), "rb") as f:
            data = pickle.load(f)
            self.atom_to_index = data["atom_to_index"]
            self._next_atom_index = data["next_index"]
        print(f"Loaded DSAC V2 heuristic from {path}")
