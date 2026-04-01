"""Goal-conditioned distance heuristic V2 using DQN/SAC via stable-baselines3.

Uses DQN (discrete action spaces) or SAC (continuous) with Hindsight Experience
Replay to learn goal-conditioned Q-values approximating negative distances.
Shares the cmd_v2 interface for gain computation, greedy pruning, UCB sampling,
residualization, and multi-round training.

Key design:
- DQNV2Wrapper: goal-conditioned gym wrapper with UCB-aware episode sampling
- DQNv2Heuristic: BaseHeuristic subclass with manual epoch loop calling
  model.learn(num_steps_per_epoch) each epoch; gains updated between epochs
- estimate_distance: queries SB3 Q-values / critic to get -Q ≈ distance
- gain methods: exact_gain / estimate_gain / naive_gain (identical to cmd_v2)
- prune / prune_by_success: greedy gain-based (identical to cmd_v2)
"""

import os
import pickle
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from relational_structs import GroundAtom, GroundOperator
from stable_baselines3 import DQN, SAC
from stable_baselines3.her import HerReplayBuffer

from task_then_motion_planning.planning import TaskThenMotionPlanningFailure

from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
from tamp_improv.approaches.improvisational.heuristics.heuristic_dqn import (
    DQNHeuristicCallback,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class DQNV2HeuristicConfig:
    """Configuration for DQN/SAC-based goal-conditioned distance heuristic V2."""

    wandb_enabled: bool = False

    # Pruning
    threshold: float = 0.05
    beta: float = 1.0

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000
    sampling_method: str = "uniform"  # "uniform", "ucb", or "stochastic_ucb"
    ucb_beta: float = 1.0
    gain_method: str = "exact"  # "exact", "estimate", or "naive"
    dist_scale: float = 1.0
    dist_quantile: float = 1.0
    auto_dist_scale: bool = False
    residualize: bool = False

    # HER
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "final"

    # Discount
    gamma: float = 0.99

    # Epoch loop
    num_steps_per_epoch: int = 500   # SB3 env steps per epoch
    learn_frequency: int = 1         # update gains every N epochs
    num_epochs_per_round: int = 20
    max_episode_steps: int = 100

    # Atom encoding
    max_atom_size: int = 50

    # Device
    device: str = "cuda"


class DQNV2Wrapper(gym.Wrapper):
    """Goal-conditioned wrapper with UCB-aware episode sampling.

    Provides a Dict observation space {"observation", "achieved_goal",
    "desired_goal"} compatible with SB3's MultiInputPolicy and HerReplayBuffer.

    Episode sampling is controlled by an optional `sample_pair_fn` callable
    (returns `(source_state, target_node_id)`).  When not provided, sampling
    falls back to uniform over the flat `state_node_pairs` list.
    """

    def __init__(
        self,
        env: gym.Env,
        state_node_pairs: list[tuple["ObsType", int]],
        node_atoms: dict[int, set[GroundAtom]],
        perceiver: Any,
        max_episode_steps: int = 100,
        max_atom_size: int = 50,
        sample_pair_fn: Callable[[], tuple["ObsType", int]] | None = None,
    ):
        super().__init__(env)
        self.state_node_pairs = state_node_pairs
        self.node_atoms = node_atoms
        self.perceiver = perceiver
        self.max_episode_steps = max_episode_steps
        self.max_atom_size = max_atom_size
        self.sample_pair_fn = sample_pair_fn
        self.steps = 0
        self.current_goal_atoms: set | None = None

        self.atom_to_index: dict[str, int] = {}
        self._next_index = 0

        # Build observation space from first sample state
        if state_node_pairs:
            sample_state = state_node_pairs[0][0]
            if hasattr(sample_state, "nodes"):
                flat_size = sample_state.nodes.flatten().shape[0]
            else:
                flat_size = np.array(sample_state).flatten().shape[0]

            obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
            )
            goal_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(max_atom_size,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": obs_space,
                    "achieved_goal": goal_space,
                    "desired_goal": goal_space,
                }
            )

    # ------------------------------------------------------------------
    # Atom helpers (identical to DQNHeuristicWrapper)
    # ------------------------------------------------------------------

    def _get_atom_index(self, atom_str: str) -> int:
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert self._next_index < self.max_atom_size, (
            f"No more space for atom at index {self._next_index}. "
            f"Increase max_atom_size (currently {self.max_atom_size})."
        )
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def create_atom_vector(self, atoms: set) -> np.ndarray:
        vector = np.zeros(self.max_atom_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector

    def flatten_obs(self, obs: "ObsType") -> np.ndarray:
        if hasattr(obs, "nodes"):
            return obs.nodes.flatten().astype(np.float32)
        return np.array(obs).flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.steps = 0

        if self.sample_pair_fn is not None:
            source_state, goal_node = self.sample_pair_fn()
        else:
            pair_idx = np.random.randint(len(self.state_node_pairs))
            source_state, goal_node = self.state_node_pairs[pair_idx]

        self.current_goal_atoms = self.node_atoms[goal_node]

        unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        obs, info = unwrapped.reset_from_state(source_state, seed=seed)

        flat_obs = self.flatten_obs(obs)
        goal_vector = self.create_atom_vector(self.current_goal_atoms)
        current_atoms = self.perceiver.step(obs)
        achieved_vector = self.create_atom_vector(current_atoms)

        return {
            "observation": flat_obs,
            "achieved_goal": achieved_vector,
            "desired_goal": goal_vector,
        }, info

    def step(
        self, action: "ActType"
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        next_obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        flat_obs = self.flatten_obs(next_obs)
        goal_vector = self.create_atom_vector(self.current_goal_atoms)
        current_atoms = self.perceiver.step(next_obs)
        achieved_vector = self.create_atom_vector(current_atoms)

        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": achieved_vector,
            "desired_goal": goal_vector,
        }

        reward = float(self.compute_reward(achieved_vector, goal_vector, info)[0])

        goal_reached = current_atoms == self.current_goal_atoms
        info["is_success"] = bool(goal_reached)
        terminated = terminated or goal_reached
        truncated = truncated or (self.steps >= self.max_episode_steps)

        return dict_obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: list[dict[str, Any]] | dict[str, Any],
        _indices: list[int] | None = None,
    ) -> np.ndarray:
        """Sparse reward: 0 if goal satisfied (superset check), -1 otherwise."""
        if achieved_goal.ndim == 1:
            goal_indices = np.where(desired_goal > 0.5)[0]
            if len(goal_indices) == 0:
                satisfied = True
            else:
                satisfied = bool(np.all(achieved_goal[goal_indices] > 0.5))
            return np.array([0.0 if satisfied else -1.0], dtype=np.float32)
        else:
            batch_size = achieved_goal.shape[0]
            rewards = np.zeros(batch_size, dtype=np.float32)
            for i in range(batch_size):
                goal_indices = np.where(desired_goal[i] > 0.5)[0]
                if len(goal_indices) == 0:
                    satisfied = True
                else:
                    satisfied = bool(np.all(achieved_goal[i][goal_indices] > 0.5))
                rewards[i] = 0.0 if satisfied else -1.0
            return rewards

    def reconfigure(self, state_node_pairs: list[tuple["ObsType", int]]) -> None:
        """Update state-node pairs (used when training data changes between rounds)."""
        self.state_node_pairs = state_node_pairs


class DQNv2Heuristic(BaseHeuristic):
    """DQN/SAC heuristic with cmd_v2-compatible interface.

    Uses stable-baselines3 DQN (discrete) or SAC (continuous) with HER for
    training.  Gain computation, greedy pruning, UCB sampling, auto_dist_scale,
    and residualization all mirror cmd_v2 exactly.
    """

    def __init__(
        self,
        training_data: GoalConditionedTrainingData,
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
        config: DQNV2HeuristicConfig | None = None,
        first_edge_dict: (
            dict[
                tuple[frozenset[GroundAtom], frozenset[GroundAtom]], PlanningGraphEdge
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
            config = DQNV2HeuristicConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.config = config

        # Determine action space
        env = system.env
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space_type = "discrete"
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_space_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")

        # Node count and matrices (match cmd_v2 structure)
        self.num_nodes = len(training_data.node_states)
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_gains = np.zeros((self.num_nodes, self.num_nodes))
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist

        # Atoms-to-node lookup (for perceiver-based node identification)
        self._node_atoms_dict = dict(training_data.node_atoms)
        self.atoms_to_node: dict[frozenset, int] = {
            frozenset(atoms): node_id
            for node_id, atoms in self._node_atoms_dict.items()
        }

        # Build flat state-node pairs for the wrapper
        self.state_node_pairs = self._build_state_node_pairs(training_data)

        self._init_model()
        self._needs_timestep_reset = True

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_state_node_pairs(
        self, training_data: GoalConditionedTrainingData
    ) -> list[tuple["ObsType", int]]:
        """Flat list of (source_state, target_node) from training data."""
        pairs = []
        for source_node, target_node in training_data.unique_shortcuts:
            for state in training_data.node_states.get(source_node, []):
                pairs.append((state, target_node))
        return pairs

    def _init_model(self) -> None:
        """(Re-)create SB3 model and wrapper."""
        device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"

        self.wrapper = DQNV2Wrapper(
            self.system.env,
            self.state_node_pairs,
            self._node_atoms_dict,
            self.system.perceiver,
            max_episode_steps=self.config.max_episode_steps,
            max_atom_size=self.config.max_atom_size,
            sample_pair_fn=self._make_sample_pair_fn(),
        )

        her_kwargs = dict(
            n_sampled_goal=self.config.n_sampled_goal,
            goal_selection_strategy=self.config.goal_selection_strategy,
        )

        if self.action_space_type == "discrete":
            self.algorithm_used = "DQN"
            self.model: DQN | SAC = DQN(
                "MultiInputPolicy",
                self.wrapper,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                device=device,
                verbose=1,
                gamma=self.config.gamma,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=her_kwargs,
            )
        else:
            self.algorithm_used = "SAC"
            self.model = SAC(
                "MultiInputPolicy",
                self.wrapper,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                device=device,
                verbose=1,
                gamma=self.config.gamma,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=her_kwargs,
            )

        self.callback = DQNHeuristicCallback(check_freq=1000, verbose=1)
        self._needs_timestep_reset = True

    def _make_sample_pair_fn(self) -> Callable[[], tuple["ObsType", int]]:
        """Return a closure over self that selects (source_state, target_node)."""

        def sample_pair() -> tuple["ObsType", int]:
            shortcuts = self.training_data.unique_shortcuts
            if not shortcuts:
                raise RuntimeError("No shortcuts in training data — cannot sample.")

            starts = np.array([x for x, _ in shortcuts])
            ends = np.array([y for _, y in shortcuts])

            if self.config.sampling_method == "uniform":
                idx = self.rng.integers(len(shortcuts))
                source_node, target_node = shortcuts[idx]

            elif self.config.sampling_method == "ucb":
                gains = self.node_pair_gains[starts, ends]
                ucbs = np.sqrt(
                    2 * np.log(self.total_samples + 1)
                    / (self.node_pair_samples[starts, ends] + 1e-8)
                )
                flat_idx = int(np.argmax(gains + self.config.ucb_beta * ucbs))
                source_node, target_node = int(starts[flat_idx]), int(ends[flat_idx])

            elif self.config.sampling_method == "stochastic_ucb":
                gains = self.node_pair_gains[starts, ends]
                ucbs = np.sqrt(
                    2 * np.log(self.total_samples + 1)
                    / (self.node_pair_samples[starts, ends] + 1e-8)
                )
                weights = gains + self.config.ucb_beta * ucbs
                exp_w = np.exp(weights - np.max(weights))
                probs = exp_w / np.sum(exp_w)
                flat_idx = int(self.rng.choice(len(shortcuts), p=probs))
                source_node, target_node = int(starts[flat_idx]), int(ends[flat_idx])

            else:
                raise ValueError(
                    f"Unknown sampling method: {self.config.sampling_method}"
                )

            self.node_pair_samples[source_node, target_node] += 1
            self.total_samples += 1

            source_states = self.training_data.node_states.get(source_node, [])
            if not source_states:
                # Fallback: uniform over flat pairs
                idx = self.rng.integers(len(self.state_node_pairs))
                return self.state_node_pairs[idx]
            state = source_states[self.rng.integers(len(source_states))]
            return state, target_node

        return sample_pair

    # ------------------------------------------------------------------
    # System update (multi-round training)
    # ------------------------------------------------------------------

    def update_system(
        self,
        new_system: "ImprovisationalTAMPSystem",
        new_graph: "PlanningGraph",  # noqa: F841 (reserved for future use)
        new_graph_distances: dict[tuple[int, int], float],
        new_first_edge_dict: dict[tuple[int, int], PlanningGraphEdge],
    ) -> None:
        """Swap in a new system and rebuild SB3 model for the next round."""
        print("Updating system for new round of training...")
        for (i, j), dist in new_graph_distances.items():
            old = self.graph_distances.get((i, j), np.inf)
            print(f"  Node {i} -> Node {j}: {old} -> {dist}")

        self.system = new_system
        self.graph_distances = new_graph_distances
        self.node_pair_graph_dists = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j), dist in new_graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist
        self.first_edge_dict = new_first_edge_dict

        print("Rebuilding SB3 model for new round...")
        self._init_model()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_round(self, **kwargs) -> dict[str, Any]:
        """Train for one round (manual epoch loop over model.learn calls).

        Returns:
            Dictionary with critic_losses, actor_losses, buffer_size.
        """
        self.total_samples = 0
        self.node_pair_samples = np.zeros((self.num_nodes, self.num_nodes))

        # Rebuild wrapper's sample_pair_fn so it closes over fresh counters
        self.wrapper.sample_pair_fn = self._make_sample_pair_fn()

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        self._update_gains()

        for epoch in range(self.config.num_epochs_per_round):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs_per_round} ===")
            self.model.learn(
                total_timesteps=self.config.num_steps_per_epoch,
                callback=self.callback,
                reset_num_timesteps=self._needs_timestep_reset,
            )
            print("Finished learning epoch.")
            self._needs_timestep_reset = False

            if (epoch + 1) % self.config.learn_frequency == 0:
                if self.config.sampling_method != "uniform":
                    self._update_gains()

            cl, al = self._get_latest_losses()
            if cl != 0.0 or al != 0.0:
                critic_losses.append(cl)
                actor_losses.append(al)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                cl_str = f"{critic_losses[-1]:.4f}" if critic_losses else "N/A"
                al_str = f"{actor_losses[-1]:.4f}" if actor_losses else "N/A"
                buf = (
                    self.model.replay_buffer.size()
                    if self.model.replay_buffer is not None
                    else 0
                )
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs_per_round} | "
                    f"Buffer: {buf} | Critic loss: {cl_str} | Actor loss: {al_str}"
                )

        self._update_gains()

        buf_size = (
            self.model.replay_buffer.size()
            if self.model.replay_buffer is not None
            else 0
        )
        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "buffer_size": buf_size,
        }

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Stub required by BaseHeuristic interface."""
        return {"critic_losses": [], "actor_losses": [], "buffer_size": 0}

    def _get_latest_losses(self) -> tuple[float, float]:
        """Read latest logged losses from SB3 logger."""
        ntv = self.model.logger.name_to_value
        if self.algorithm_used == "DQN":
            return float(ntv.get("train/loss", 0.0)), 0.0
        else:  # SAC
            return (
                float(ntv.get("train/critic_loss", 0.0)),
                float(ntv.get("train/actor_loss", 0.0)),
            )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, obs: "ObsType", target_node: int) -> NDArray | int:
        source_flat = self._flatten_state(obs)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        target_goal_vector = self.wrapper.create_atom_vector(target_atoms)
        source_atoms = self.system.perceiver.step(obs)
        achieved_vector = self.wrapper.create_atom_vector(source_atoms)

        sb3_obs = {
            "observation": source_flat,
            "achieved_goal": achieved_vector,
            "desired_goal": target_goal_vector,
        }

        if self.algorithm_used == "SAC":
            action, _ = self.model.predict(sb3_obs, deterministic=True)
        else:  # DQN
            device = next(self.model.q_net.parameters()).device
            obs_tensor = {
                k: torch.as_tensor(v).unsqueeze(0).to(device)
                for k, v in sb3_obs.items()
            }
            with torch.no_grad():
                q_values = self.model.q_net(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        if self.config.residualize and self.action_space_type == "continuous":
            start_atoms = self.system.perceiver.step(obs)
            goal_atoms = self._node_atoms_dict.get(target_node, set())
            edge = (
                self.first_edge_dict.get(
                    (frozenset(start_atoms), frozenset(goal_atoms)), None
                )
                if self.first_edge_dict is not None
                else None
            )
            if edge is not None:
                operator = edge.operator
                skills = [s for s in self.system.skills if s.can_execute(operator)]
                if not skills:
                    raise TaskThenMotionPlanningFailure(
                        f"No skill found for operator {operator.name}"
                    )
                skill = skills[0]
                skill.reset(operator)
                base_action = skill.get_action(obs)
            else:
                base_action = np.zeros_like(action)
            return action + base_action

        return action

    # ------------------------------------------------------------------
    # Distance estimation (Q-value based)
    # ------------------------------------------------------------------

    def estimate_distance(self, source_state: "ObsType", target_node: int) -> float:
        """Estimate distance via -Q(s, a*, g)."""
        source_flat = self._flatten_state(source_state)
        target_atoms = self._node_atoms_dict.get(target_node, set())
        target_goal_vector = self.wrapper.create_atom_vector(target_atoms)
        source_atoms = self.system.perceiver.step(source_state)
        achieved_vector = self.wrapper.create_atom_vector(source_atoms)

        sb3_obs = {
            "observation": source_flat,
            "achieved_goal": achieved_vector,
            "desired_goal": target_goal_vector,
        }

        with torch.no_grad():
            if self.algorithm_used == "SAC":
                action, _ = self.model.predict(sb3_obs, deterministic=True)
                action_tensor = torch.as_tensor(action).unsqueeze(0).to(
                    self.model.device
                )
                obs_tensor = {
                    k: torch.as_tensor(v).unsqueeze(0).to(self.model.device)
                    for k, v in sb3_obs.items()
                }
                q1 = self.model.critic(obs_tensor, action_tensor)[0]
                q2 = self.model.critic(obs_tensor, action_tensor)[1]
                q_value = float(torch.min(q1, q2).item())
            else:  # DQN
                device = next(self.model.q_net.parameters()).device
                obs_tensor = {
                    k: torch.as_tensor(v).unsqueeze(0).to(device)
                    for k, v in sb3_obs.items()
                }
                q_values = self.model.q_net(obs_tensor)
                q_value = float(torch.max(q_values).item())

        return max(0.0, float(-q_value))

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Average estimate_distance over sample states from source_node."""
        source_states = self.training_data.node_states.get(source_node, [])
        if not source_states:
            return float("inf")
        sampled = random.sample(source_states, min(100, len(source_states)))
        return self.config.dist_scale * float(
            np.mean([self.estimate_distance(s, target_node) for s in sampled])
        )

    # ------------------------------------------------------------------
    # Gain methods (identical to cmd_v2)
    # ------------------------------------------------------------------

    def exact_gain(self, source_node: int, target_node: int) -> float:
        x, y = source_node, target_node
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(x, y)

        U = np.where(np.isfinite(d[:, x]))[0]
        V = np.where(np.isfinite(d[y, :]))[0]

        new_paths = d[U, x][:, None] + L + d[y, V][None, :]
        old_paths = np.minimum(d[np.ix_(U, V)], self.config.max_episode_steps)
        improvement = np.maximum(0, old_paths - new_paths)
        return float(np.sum(improvement))

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
        elif self.config.gain_method == "estimate":
            return self.estimate_gain(source_node, target_node)
        elif self.config.gain_method == "naive":
            return self.naive_gain(source_node, target_node)
        else:
            raise ValueError(f"Unknown gain method: {self.config.gain_method}")

    def estimate_weight(self, source_node: int, target_node: int) -> float:
        if source_node == target_node:
            return 0.0
        gain = self.node_pair_gains[source_node, target_node]
        ucb = np.sqrt(
            2 * np.log(self.total_samples + 1)
            / (self.node_pair_samples[source_node, target_node] + 1e-8)
        )
        return gain + self.config.ucb_beta * ucb

    # ------------------------------------------------------------------
    # Gain tracking (identical to cmd_v2)
    # ------------------------------------------------------------------

    def _update_gains(self) -> None:
        print("\nUpdating node pair gains...")

        if self.config.auto_dist_scale:
            self.config.dist_scale = 1.0
            shortcut_dists = np.zeros((self.num_nodes, self.num_nodes))
            for src, tgt in self.training_data.unique_shortcuts:
                shortcut_dists[src, tgt] = self.estimate_node_distance(src, tgt)
            ratios = shortcut_dists / self.node_pair_graph_dists
            finite = ratios[np.isfinite(ratios)]
            if finite.size > 0:
                ratio = 1.0 / np.quantile(finite, self.config.dist_quantile)
                self.config.dist_scale = ratio if ratio < 1.0 else 1.0
            print(f"  Auto-set dist_scale to {self.config.dist_scale:.4f}")

        for src, tgt in self.training_data.unique_shortcuts:
            self.node_pair_gains[src, tgt] = self.get_gain(src, tgt)

        print("Updated node pair gains.\n")

    def _update_graph_distances(self, source_node: int, target_node: int) -> None:
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)
        via_xy = d[:, source_node][:, None] + L + d[target_node, :][None, :]
        self.node_pair_graph_dists = np.minimum(d, via_xy)

    # ------------------------------------------------------------------
    # Pruning (identical to cmd_v2)
    # ------------------------------------------------------------------

    def prune_by_success(
        self, success_threshold: float, max_steps: int
    ) -> GoalConditionedTrainingData:
        """Prune shortcuts below estimated success probability."""
        print("Heuristic-Estimated Success Probs of All Shortcuts:")
        pruned_pairs = []
        for x, y in self.training_data.unique_shortcuts:
            d = self.estimate_node_distance(x, y)
            prob = 1.0 if d / max_steps < success_threshold else 0.0
            print(f"  ({x} -> {y}): {d:.2f} (success prob: {prob:.2f})")
            if prob > success_threshold:
                pruned_pairs.append((x, y))

        return self._build_pruned_data(pruned_pairs, "dqn")

    def prune(self, max_shortcuts: int) -> GoalConditionedTrainingData:
        print(f"\n[DEBUG] Pruning greedily to max_shortcuts={max_shortcuts}")

        print("Heuristic-Estimated Lengths of All Shortcuts:")
        for x, y in self.training_data.unique_shortcuts:
            print(f"  ({x} -> {y}): {self.estimate_node_distance(x, y):.2f}")

        if (
            max_shortcuts is None
            or max_shortcuts >= len(self.training_data.unique_shortcuts)
        ):
            return self.training_data

        starts = np.array([x for x, _ in self.training_data.unique_shortcuts])
        ends = np.array([y for _, y in self.training_data.unique_shortcuts])

        curr_gains = self.node_pair_gains.copy()
        curr_dists = self.node_pair_graph_dists.copy()
        pruned_pairs: list[tuple[int, int]] = []

        self.config.auto_dist_scale = False
        for i in range(max_shortcuts):
            self._update_gains()
            gains = self.node_pair_gains[starts, ends]
            max_idx = int(np.argmax(gains))
            src, tgt = int(starts[max_idx]), int(ends[max_idx])
            pruned_pairs.append((src, tgt))
            self._update_graph_distances(src, tgt)
            starts = np.delete(starts, max_idx)
            ends = np.delete(ends, max_idx)
            print(f"  Selected shortcut {i + 1}: {src} -> {tgt} with gain {gains[max_idx]:.2f}")

        self.config.auto_dist_scale = True
        self.node_pair_gains = curr_gains
        self.node_pair_graph_dists = curr_dists

        return self._build_pruned_data(pruned_pairs, "dqn")

    def _build_pruned_data(
        self,
        pruned_pairs: list[tuple[int, int]],
        pruning_method: str,
    ) -> GoalConditionedTrainingData:
        """Build a new GoalConditionedTrainingData from a selected list of node pairs."""
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
                "pruning_method": pruning_method,
                "threshold": self.config.threshold,
            },
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _flatten_state(self, state: "ObsType") -> np.ndarray:
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32)
        return np.array(state).flatten().astype(np.float32)

    def _find_node_for_atoms(self, atoms: set) -> int | None:
        return self.atoms_to_node.get(frozenset(atoms), None)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save SB3 model and atom indexing."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "distance_model"))
        with open(os.path.join(path, "algorithm.pkl"), "wb") as f:
            pickle.dump({"algorithm": self.algorithm_used}, f)
        with open(os.path.join(path, "atom_to_index.pkl"), "wb") as f:
            pickle.dump(
                {
                    "atom_to_index": self.wrapper.atom_to_index,
                    "_next_index": self.wrapper._next_index,
                    "max_atom_size": self.wrapper.max_atom_size,
                },
                f,
            )
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        print(f"Saved DQN V2 heuristic to {path}")

    def load(self, path: str) -> None:
        """Load SB3 model and restore atom indexing."""
        with open(os.path.join(path, "algorithm.pkl"), "rb") as f:
            self.algorithm_used = pickle.load(f)["algorithm"]
        with open(os.path.join(path, "atom_to_index.pkl"), "rb") as f:
            atom_data = pickle.load(f)
            self.wrapper.atom_to_index = atom_data["atom_to_index"]
            self.wrapper._next_index = atom_data["_next_index"]

        device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
        if self.algorithm_used == "DQN":
            self.model = DQN.load(
                os.path.join(path, "distance_model"),
                env=self.wrapper,
                device=device,
            )
        else:
            self.model = SAC.load(
                os.path.join(path, "distance_model"),
                env=self.wrapper,
                device=device,
            )
        print(f"Loaded DQN V2 heuristic from {path}")
