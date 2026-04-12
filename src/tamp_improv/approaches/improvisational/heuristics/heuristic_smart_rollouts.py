"""Rollout-based heuristic with exact gain computation.

Performs random rollouts to estimate shortcut distances, then uses the same
exact/estimate/naive gain methods as sac_v2/crl_v2/cmd_v2 for greedy pruning.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from tamp_improv.approaches.improvisational.heuristics.base import (
    BaseHeuristic,
    random_selection,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


@dataclass
class SmartRolloutsConfig:
    num_rollouts_per_node: int = 100
    max_steps_per_rollout: int = 100
    max_episode_steps: int = 100
    success_threshold: float = 0.01
    action_scale: float = 1.0
    gain_method: str = "exact"          # "exact", "estimate", or "naive"
    auto_dist_scale: bool = True
    dist_scale: float = 1.0
    dist_quantile: float = 0.9


class SmartRolloutsHeuristic(BaseHeuristic):
    """Random-rollout heuristic with exact gain-based greedy pruning.

    Rollouts estimate shortcut distances (mean success length). Pruning uses
    the same exact/estimate/naive gain methods as sac_v2/crl_v2/cmd_v2 so that
    shortcut selection is comparable across heuristic types.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
        seed: int = 42,
        config: SmartRolloutsConfig | None = None,
    ):
        super().__init__(training_data, graph_distances)
        self.system = system
        self.rng = rng
        self.seed = seed
        self.config = config or SmartRolloutsConfig()

        # Cache populated during train_one_round()
        self._success_counts: dict[tuple[int, int], int] | None = None
        self._success_lens: dict[tuple[int, int], list[int]] | None = None

        # Action sampling space
        raw_env = system.env
        if isinstance(raw_env.action_space, gym.spaces.Box):
            self.sampling_space = gym.spaces.Box(
                low=raw_env.action_space.low * self.config.action_scale,
                high=raw_env.action_space.high * self.config.action_scale,
                dtype=np.float32,
            )
        else:
            print("Warning: Action space is not Box, using original action space.")
            self.sampling_space = raw_env.action_space
        self.sampling_space.seed(seed)

        self._target_atoms_by_id = dict(training_data.node_atoms)

        # Pre-compute valid targets per source node as a set for O(1) lookup
        self._valid_targets: dict[int, set[int]] = {}
        for source_id, target_id in training_data.valid_shortcuts:
            self._valid_targets.setdefault(source_id, set()).add(target_id)

        # Node bookkeeping (mirrors sac_v2 layout)
        self.num_nodes = len(training_data.node_states)
        self.node_pair_graph_dists = np.full(
            (self.num_nodes, self.num_nodes), np.inf
        )
        for (i, j), dist in graph_distances.items():
            self.node_pair_graph_dists[i, j] = dist
        np.fill_diagonal(self.node_pair_graph_dists, 0.0)

        self.node_pair_gains = np.zeros((self.num_nodes, self.num_nodes))

    # ── Training ──────────────────────────────────────────────────────

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Stub — pipeline calls train_one_round() directly."""
        return {}

    def train_one_round(self, **kwargs) -> dict[str, Any]:
        """Run random rollouts to populate distance estimates."""
        print(f"\nRunning smart rollouts:")
        print(f"  Rollouts per node: {self.config.num_rollouts_per_node}")
        print(f"  Max steps per rollout: {self.config.max_steps_per_rollout}")

        shortcut_success_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        shortcut_lengths: defaultdict[tuple[int, int], list[int]] = defaultdict(list)

        raw_env = self.system.env
        total_rollouts = 0

        for source_id, source_states in self.training_data.node_states.items():
            if not source_states:
                continue

            source_atoms = self._target_atoms_by_id.get(source_id, set())
            print(
                f"\nPerforming {self.config.num_rollouts_per_node} rollouts from node {source_id} "
                f"({len(source_states)} source state(s))",
                flush=True,
            )

            for rollout_idx in range(self.config.num_rollouts_per_node):
                if rollout_idx > 0 and rollout_idx % 100 == 0:
                    print(f"  Completed {rollout_idx}/{self.config.num_rollouts_per_node} rollouts", flush=True)

                state_idx = self.rng.integers(0, len(source_states))
                raw_env.reset_from_state(source_states[state_idx])
                curr_atoms = source_atoms.copy()
                reached_in_this_rollout: set[int] = set()

                for step_idx in range(self.config.max_steps_per_rollout):
                    action = self.sampling_space.sample()
                    obs, _, terminated, truncated, _ = raw_env.step(action)
                    curr_atoms = self.system.perceiver.step(obs)

                    valid_targets = self._valid_targets.get(source_id, set())
                    for target_id in valid_targets - reached_in_this_rollout:
                        target_atoms = self._target_atoms_by_id.get(target_id)
                        if target_atoms and target_atoms == curr_atoms:
                            shortcut_success_counts[(source_id, target_id)] += 1
                            shortcut_lengths[(source_id, target_id)].append(step_idx + 1)
                            reached_in_this_rollout.add(target_id)

                    if terminated or truncated:
                        break

                total_rollouts += 1

            print(f"  Completed all {self.config.num_rollouts_per_node} rollouts from node {source_id}", flush=True)

        print("\nRollout results:")
        for (src, tgt), count in shortcut_success_counts.items():
            sr = count / self.config.num_rollouts_per_node if self.config.num_rollouts_per_node > 0 else 0.0
            avg_len = (
                float(np.mean(shortcut_lengths[(src, tgt)]))
                if shortcut_lengths[(src, tgt)]
                else self.config.max_steps_per_rollout
            )
            print(f"  ({src} -> {tgt}): {count} successes ({sr:.2%}), avg length: {avg_len:.1f}")

        self._success_counts = dict(shortcut_success_counts)
        self._success_lens = dict(shortcut_lengths)

        self._update_gains()

        return {
            "method": "smart_rollouts",
            "total_rollouts": total_rollouts,
            "success_counts": self._success_counts,
        }

    # ── Distance estimation ───────────────────────────────────────────

    def estimate_distance(self, state: "ObsType", target_node: int) -> float:
        for node_id, states in self.training_data.node_states.items():
            for s in states:
                if np.array_equal(np.array(s), np.array(state)):
                    return self.estimate_node_distance(node_id, target_node)
        return float(self.config.max_steps_per_rollout)

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Return mean rollout success length, or max_steps if never reached."""
        if self._success_lens is None:
            return float(self.config.max_steps_per_rollout)
        lengths = self._success_lens.get((source_node, target_node), [])
        if lengths:
            return self.config.dist_scale * float(np.min(lengths))
        return float(self.config.max_steps_per_rollout)

    # ── Gain methods (identical to sac_v2 / crl_v2 / cmd_v2) ─────────

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
                print(f"  Auto-scaling distances: quantile={q}, ratio={ratio:.4f}")
                if not np.isinf(ratio) and ratio > 0:
                    self.config.dist_scale = ratio
            print(f"  Auto-set dist_scale={self.config.dist_scale:.4f}")

        for src, tgt in self.training_data.unique_shortcuts:
            self.node_pair_gains[src, tgt] = self.get_gain(src, tgt)

        print("Updated node pair gains.")

    def _update_graph_distances(self, source_node: int, target_node: int) -> None:
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)
        via_xy = d[:, source_node][:, None] + L + d[target_node, :][None, :]
        self.node_pair_graph_dists = np.minimum(d, via_xy)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def estimate_probability(self, source_node: int, target_node: int, use_multi_rl: bool = True) -> float:
        """Estimate PPO success probability from empirical rollout success rate."""
        if self._success_counts is None:
            return 0.0
        count = self._success_counts.get((source_node, target_node), 0)
        n = self.config.num_rollouts_per_node
        p_rr = count / n if n > 0 else 0.0
        # Convert random rollout probability to PPO probability via step function
        k = np.log(0.5) / np.log(1 - 0.05)  # threshold = 0.05
        return 1 - (1 - p_rr)**k

    # ── Pruning ───────────────────────────────────────────────────────

    def prune_by_success(
        self, success_threshold: float, max_steps: int, **kwargs: Any
    ) -> "GoalConditionedTrainingData":
        """Keep shortcuts whose rollout success rate exceeds the threshold."""
        if self._success_counts is None:
            self._success_counts = {}
            self._success_lens = {}

        print("Rollout success rates for all shortcuts:")
        pruned_pairs = []
        for x, y in self.training_data.unique_shortcuts:
            count = self._success_counts.get((x, y), 0)
            prob = count / self.config.num_rollouts_per_node if self.config.num_rollouts_per_node > 0 else 0.0
            print(f"  ({x} -> {y}): {prob:.2f} ({count}/{self.config.num_rollouts_per_node})")
            if prob > success_threshold:
                pruned_pairs.append((x, y))

        return self._build_pruned_data(pruned_pairs)

    def prune(self, max_shortcuts: int | None, **kwargs: Any) -> "GoalConditionedTrainingData":
        """Greedy gain-based pruning, identical to sac_v2."""
        if self._success_lens is None:
            self._success_counts = {}
            self._success_lens = {}

        print(f"\nPruning greedily to max_shortcuts={max_shortcuts}")

        print("Estimated distances for all shortcuts:")
        for x, y in self.training_data.unique_shortcuts:
            print(f"  ({x} -> {y}): {self.estimate_node_distance(x, y):.2f}")

        if max_shortcuts is None or max_shortcuts >= len(self.training_data.unique_shortcuts):
            return self.training_data

        starts = np.array([x for x, _ in self.training_data.unique_shortcuts])
        ends = np.array([y for _, y in self.training_data.unique_shortcuts])

        saved_gains = self.node_pair_gains.copy()
        saved_dists = self.node_pair_graph_dists.copy()
        prev_auto = self.config.auto_dist_scale
        self.config.auto_dist_scale = False

        pruned_pairs: list[tuple[int, int]] = []
        for i in range(max_shortcuts):
            self._update_gains()
            gains = self.node_pair_gains[starts, ends]

            # Weight gains by estimated PPO success probability
            probs = np.zeros(len(starts))
            for j, (s, t) in enumerate(zip(starts, ends)):
                probs[j] = self.estimate_probability(int(s), int(t))

            if np.any(probs > 0):
                scores = probs * gains
            else:
                scores = gains  # Fall back to pure gain if all probabilities are 0

            best = int(np.argmax(scores))
            src, tgt = int(starts[best]), int(ends[best])
            p = probs[best]
            pruned_pairs.append((src, tgt))
            self._update_graph_distances(src, tgt)
            starts = np.delete(starts, best)
            ends = np.delete(ends, best)
            print(f"  Selected shortcut {i+1}: {src} -> {tgt} (gain={gains[best]:.2f}, p_ppo={p:.2f}, score={scores[best]:.2f})")

        self.config.auto_dist_scale = prev_auto
        self.node_pair_gains = saved_gains
        self.node_pair_graph_dists = saved_dists

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
                "pruning_method": "smart_rollouts",
                "gain_method": self.config.gain_method,
                "num_rollouts": self.config.num_rollouts_per_node,
            },
        )

    # ── Unsupported multi-round methods ──────────────────────────────

    def update_system(self, **kwargs: Any) -> None:
        raise NotImplementedError("SmartRolloutsHeuristic does not support multi-round training.")

    def get_action(self, obs: "ObsType", target_node: int) -> np.ndarray | int:
        raise NotImplementedError("get_action is not implemented for SmartRolloutsHeuristic.")
