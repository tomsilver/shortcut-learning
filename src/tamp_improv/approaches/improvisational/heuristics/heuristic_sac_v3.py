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
from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
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


class ContinuousActor(nn.Module):
    """Squashed-Gaussian actor: π(a | state, goal_atoms).

    Identical to the actor in heuristic_crl_v2.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

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
        input_dim = state_dim + atom_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(
        self, states: torch.Tensor, goals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([states, goals], dim=-1)
        features = self.trunk(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self.forward(states, goals)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # (B,)
        return action, log_prob


# ── Replay buffer ─────────────────────────────────────────────────────

# Trajectory step: (state_flat, action, reward, next_state_flat, done,
#                   achieved_atoms_vec, next_achieved_atoms_vec)
_TrajStep = tuple[NDArray, NDArray, float, NDArray, bool, NDArray, NDArray]


class SACReplayBuffer:
    """Circular replay buffer storing flat (s, a, r, s', done, goal) tuples."""

    def __init__(self, max_size: int):
        self.buffer: deque[tuple[NDArray, NDArray, float, NDArray, bool, NDArray]] = (
            deque(maxlen=max_size)
        )

    def add(
        self,
        state: NDArray,
        action: NDArray,
        reward: float,
        next_state: NDArray,
        done: bool,
        goal: NDArray,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, goal))

    def sample(
        self, batch_size: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            np.array(goals, dtype=np.float32),
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

        # UCB / gain arrays
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_gains = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist

        print(f"SAC V2: state_dim={self.state_dim}, action_dim={self.action_dim}, "
              f"num_nodes={self.num_nodes}")

        self._init_networks()
        self.replay_buffer = SACReplayBuffer(config.buffer_size)
        self.training_step = 0

    # ── Network initialisation ────────────────────────────────────────

    def _init_networks(self) -> None:
        device = torch.device(self.config.device)
        hidden_dims = self.config.hidden_dims or [256, 256]
        actor_hidden_dims = self.config.actor_hidden_dims or hidden_dims
        atom_dim = self.config.max_atom_size

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

    def train_one_round(self, **kwargs) -> dict[str, Any]:
        """Train for one round (multiple epochs)."""
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        self._update_gains()

        for epoch in range(self.config.num_epochs_per_round):
            # Collect trajectories
            for _ in range(self.config.trajectories_per_epoch):
                trajectory, goal_vec = self._collect_trajectory()
                self._add_to_buffer_with_her(trajectory, goal_vec)

            # Update networks
            if (epoch + 1) % self.config.learn_frequency == 0:
                if len(self.replay_buffer) >= self.config.batch_size:
                    for _ in range(self.config.iters_per_epoch):
                        c_loss, a_loss = self._update_networks()
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

                if self.config.sampling_method != "uniform":
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

    def _collect_trajectory(self) -> tuple[list[_TrajStep], NDArray]:
        """Collect one trajectory using current actor.

        Returns:
            trajectory: list of (s, a, r, s', done, achieved_vec, next_achieved_vec)
            goal_vec: multi-hot goal atom vector used during collection
        """
        node_ids = list(self.training_data.node_states.keys())

        # Sample (source, target) node pair
        if self.config.sampling_method == "uniform":
            source_id = int(self.rng.choice(node_ids))
            target_id = int(self.rng.choice(node_ids))
        elif self.config.sampling_method == "ucb":
            gains = self.node_pair_gains
            ucbs = np.sqrt(
                2 * np.log(max(self.total_samples, 1))
                / (self.node_pair_samples + 1e-8)
            )
            flat_idx = int(np.argmax(gains + self.config.ucb_beta * ucbs))
            source_id, target_id = flat_idx // self.num_nodes, flat_idx % self.num_nodes
        elif self.config.sampling_method == "stochastic_ucb":
            gains = self.node_pair_gains
            ucbs = np.sqrt(
                2 * np.log(max(self.total_samples, 1))
                / (self.node_pair_samples + 1e-8)
            )
            weights = gains + self.config.ucb_beta * ucbs
            exp_w = np.exp(weights - np.max(weights))
            probs = exp_w / exp_w.sum()
            flat_idx = int(
                self.rng.choice(self.num_nodes * self.num_nodes, p=probs.flatten())
            )
            source_id, target_id = flat_idx // self.num_nodes, flat_idx % self.num_nodes
        else:
            raise ValueError(f"Unknown sampling_method: {self.config.sampling_method}")

        print("Decided to sample from node pair:", source_id, target_id)
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

        for _ in range(self.config.max_episode_steps):
            state_flat = self._flatten_state(current_state)
            achieved_atoms = self.system.perceiver.step(current_state)
            achieved_vec = self.create_atom_vector(achieved_atoms)

            # Stochastic action during training
            action = self._select_action(state_flat, goal_vec, deterministic=False)

            next_state, _, terminated, truncated, _ = env.step(action)
            next_flat = self._flatten_state(next_state)
            next_atoms = self.system.perceiver.step(next_state)
            next_achieved_vec = self.create_atom_vector(next_atoms)

            goal_reached = target_atoms <= next_atoms  # superset check
            reward = 0.0 if goal_reached else -1.0
            done = terminated or truncated or goal_reached

            trajectory.append(
                (state_flat, action, reward, next_flat, done, achieved_vec, next_achieved_vec)
            )

            if done:
                break
            current_state = next_state

        return trajectory, goal_vec

    def _add_to_buffer_with_her(
        self, trajectory: list[_TrajStep], goal_vec: NDArray
    ) -> None:
        """Add real transitions and HER-relabelled transitions to the buffer."""
        if not trajectory:
            return

        T = len(trajectory)

        # Real transitions
        for s, a, r, s_, done, _, _ in trajectory:
            self.replay_buffer.add(s, a, r, s_, done, goal_vec)

        # HER transitions
        for t in range(T):
            s, a, _, s_, done, _, next_achieved_vec = trajectory[t]

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

                # Reward: 0 if next state satisfies the HER goal
                goal_atoms_active = her_goal_vec > 0.5
                her_r = (
                    0.0
                    if np.all(next_achieved_vec[goal_atoms_active] > 0.5)
                    else -1.0
                )
                her_done = her_r == 0.0 or done
                self.replay_buffer.add(s, a, her_r, s_, her_done, her_goal_vec)

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

        states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(
            self.config.batch_size
        )
        s = torch.FloatTensor(states).to(device)
        a = torch.FloatTensor(actions).to(device)
        r = torch.FloatTensor(rewards).to(device)
        s_ = torch.FloatTensor(next_states).to(device)
        d = torch.FloatTensor(dones).to(device)
        g = torch.FloatTensor(goals).to(device)

        alpha = self.log_alpha.exp().detach()

        # ── Critic update ────────────────────────────────────────────
        with torch.no_grad():
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

    # ── Action + distance interface ───────────────────────────────────

    def get_action(self, obs: "ObsType", target_node: int) -> NDArray:
        state_flat = self._flatten_state(obs)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        goal_vec = self.create_atom_vector(target_atoms)
        actor_action = self._select_action(state_flat, goal_vec, deterministic=True)

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
                base_action = skill.get_action(obs)
            else:
                base_action = np.zeros_like(actor_action)
            return actor_action + base_action

        return actor_action

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
        source_states = self.training_data.node_states[source_node]
        samples = random.sample(source_states, min(100, len(source_states)))
        dists = [self.estimate_distance(s, target_node) for s in samples]
        return self.config.dist_scale * float(np.mean(dists))

    # ── Gain methods (identical to cmd_v2 / crl_v2) ──────────────────

    def exact_gain(self, source_node: int, target_node: int) -> float:
        x, y = source_node, target_node
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(x, y)

        U = np.where(np.isfinite(d[:, x]))[0]
        V = np.where(np.isfinite(d[y, :]))[0]

        new_paths = d[U, x][:, None] + L + d[y, V][None, :]
        old_paths = np.minimum(d[np.ix_(U, V)], self.config.max_episode_steps)
        return float(np.sum(np.maximum(0, old_paths - new_paths)))

    def estimate_gain(self, source_node: int, target_node: int) -> float:
        x, y = source_node, target_node
        d = np.clip(self.node_pair_graph_dists, 0, self.config.max_episode_steps)
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

    def naive_gain(self, source_node: int, target_node: int) -> float:
        L = self.estimate_node_distance(source_node, target_node)
        d_xy = self.node_pair_graph_dists[source_node, target_node]
        return float(np.clip(d_xy - L, 0, self.config.max_episode_steps))

    def get_gain(self, source_node: int, target_node: int) -> float:
        if self.config.gain_method == "exact":
            return self.exact_gain(source_node, target_node)
        if self.config.gain_method == "estimate":
            return self.estimate_gain(source_node, target_node)
        if self.config.gain_method == "naive":
            return self.naive_gain(source_node, target_node)
        raise ValueError(f"Unknown gain_method: {self.config.gain_method}")

    def _update_gains(self) -> None:
        print("\nUpdating node pair gains...")

        if self.config.auto_dist_scale:
            self.config.dist_scale = 1.0
            shortcut_dists = np.zeros((self.num_nodes, self.num_nodes))
            for src, tgt in self.training_data.unique_shortcuts:
                shortcut_dists[src, tgt] = self.estimate_node_distance(src, tgt)
            ratios = shortcut_dists / (self.node_pair_graph_dists + 1e-8)
            valid = np.isfinite(ratios) & (self.node_pair_graph_dists > 0)
            if valid.any():
                q = self.config.dist_quantile
                ratio = 1.0 / np.quantile(ratios[valid], q)
                print(f"  Auto-scaling distances with quantile {q} ratio={ratio:.4f}")
                print(ratios[valid])
                # make sure ratio is not infinite
                if not np.isinf(ratio) and ratio > 0:
                    self.config.dist_scale = ratio
                else:
                    self.config.dist_scale = 1.0

                
            print(f"  Auto-set dist_scale={self.config.dist_scale:.4f}")

        for src, tgt in self.training_data.unique_shortcuts:
            self.node_pair_gains[src, tgt] = self.get_gain(src, tgt)

        print("Updated node pair gains.\n")

    def _update_graph_distances(self, source_node: int, target_node: int) -> None:
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)
        via_xy = d[:, source_node][:, None] + L + d[target_node, :][None, :]
        self.node_pair_graph_dists = np.minimum(d, via_xy)

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
        for x, y in self.training_data.unique_shortcuts:
            d = self.estimate_node_distance(x, y)
            prob = 1.0 if d / max_steps < success_threshold else 0.0
            print(f"  ({x} -> {y}): {d:.2f} (success prob: {prob:.2f})")
            if prob > success_threshold:
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
