"""Rollout-based heuristic for shortcut evaluation."""

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic, random_selection
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


class RolloutsHeuristic(BaseHeuristic):
    """Rollout-based heuristic for evaluating shortcuts.

    This heuristic performs random rollouts from source nodes to evaluate
    which target nodes are reachable. The rollouts are executed during
    multi_train(), and the results (success counts) are cached for use
    in estimate_node_distance() and prune().

    The success rate (0-1) indicates how often a target node is reached
    within the rollout horizon.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
        num_rollouts: int = 1000,
        max_steps_per_rollout: int = 100,
        threshold: float = 0.01,
        action_scale: float = 1.0,
        seed: int = 42,
    ):
        """Initialize rollouts heuristic.

        Args:
            training_data: Full training data from collect_total_shortcuts
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            system: TAMP system for executing rollouts
            num_rollouts: Number of rollouts per source node
            max_steps_per_rollout: Maximum steps per rollout
            threshold: Success rate threshold for pruning (0-1)
            action_scale: Scale factor for action sampling
            seed: Random seed for action sampling
        """
        super().__init__(training_data, graph_distances)
        self.system = system
        self.num_rollouts = num_rollouts
        self.max_steps_per_rollout = max_steps_per_rollout
        self.threshold = threshold
        self.action_scale = action_scale
        self.seed = seed
        self.rng = rng

        # Cache for success counts: (source_node, target_node) -> count
        # Populated during multi_train()
        self._success_counts: dict[tuple[int, int], int] | None = None

        # Setup action sampling space
        raw_env = system.env
        if isinstance(raw_env.action_space, gym.spaces.Box):
            self.sampling_space = gym.spaces.Box(
                low=raw_env.action_space.low * action_scale,
                high=raw_env.action_space.high * action_scale,
                dtype=np.float32,
            )
        else:
            print("Warning: Action space is not Box, using original action space.")
            self.sampling_space = raw_env.action_space
        self.sampling_space.seed(seed)

        # Pre-compute target node atom sets from training_data.node_atoms
        self._target_atoms_by_id = dict(training_data.node_atoms)

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Run rollouts to evaluate all shortcuts.

        This performs the actual work of the rollouts heuristic by executing
        random rollouts and counting successes.

        Returns:
            Dictionary with rollout statistics
        """
        print(f"\nRunning rollouts:")
        print(f"  Rollouts per node: {self.num_rollouts}")
        print(f"  Max steps per rollout: {self.max_steps_per_rollout}")
        print(f"  Total shortcuts: {len(self.training_data.valid_shortcuts)}")

        # Track how many times each shortcut succeeds
        shortcut_success_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

        # Get the environment for running rollouts
        raw_env = self.system.env

        # Perform rollouts from each source node
        total_rollouts = 0
        for source_id, source_states in self.training_data.node_states.items():
            if not source_states:
                continue

            source_atoms = self._target_atoms_by_id.get(source_id, set())

            print(
                f"\nPerforming {self.num_rollouts} rollouts by randomly sampling from "
                f"{len(source_states)} state(s) from node {source_id}",
                flush=True,
            )

            # Sample from source_states exactly self.num_rollouts times
            for rollout_idx in range(self.num_rollouts):
                if rollout_idx > 0 and rollout_idx % 100 == 0:
                    print(
                        f"  Completed {rollout_idx}/{self.num_rollouts} rollouts",
                        flush=True,
                    )

                # Randomly sample a source state
                state_idx = self.rng.integers(0, len(source_states))
                source_state = source_states[state_idx]

                # Reset to source state
                raw_env.reset_from_state(source_state)
                curr_atoms = source_atoms.copy()

                # Track which nodes we've reached in this rollout
                reached_in_this_rollout: set[int] = set()

                # Execute random rollout
                for step_idx in range(self.max_steps_per_rollout):
                    action = self.sampling_space.sample()
                    obs, _, terminated, truncated, _ = raw_env.step(action)
                    curr_atoms = self.system.perceiver.step(obs)

                    # Check if we've reached any target nodes
                    for target_id in self.training_data.node_states.keys():
                        # Skip if not a valid shortcut
                        if (source_id, target_id) not in self.training_data.valid_shortcuts:
                            continue

                        # Skip if already reached in this rollout
                        if target_id in reached_in_this_rollout:
                            continue

                        # Check if atoms match target node
                        target_atoms = self._target_atoms_by_id.get(target_id)
                        if target_atoms and target_atoms == curr_atoms:
                            shortcut_success_counts[(source_id, target_id)] += 1
                            reached_in_this_rollout.add(target_id)

                    if terminated or truncated:
                        break

                total_rollouts += 1

            # Print progress after completing all rollouts for this node
            print(
                f"  Completed all {self.num_rollouts} rollouts from node {source_id}",
                flush=True,
            )

        print("\nRollout results:")
        for (source_id, target_id), count in shortcut_success_counts.items():
            success_rate = count / self.num_rollouts if self.num_rollouts > 0 else 0.0
            print(f"  Shortcut ({source_id} -> {target_id}): {count} successes ({success_rate:.2%})")

        # Store results
        self._success_counts = dict(shortcut_success_counts)
        print("Success counts:", self._success_counts)

        # Return training history
        return {
            "method": "rollouts",
            "total_rollouts": total_rollouts,
            "num_shortcuts_evaluated": len(self.training_data.valid_shortcuts),
            "success_counts": self._success_counts,
        }

    def estimate_distance(self, state: "ObsType", target_node: int) -> float:
        """Estimate distance from state to target node using rollouts.

        For rollouts, we return (1 - success_rate) * max_steps as a distance proxy.
        Higher success rate = lower distance.

        Note: This uses node-level estimates since rollouts are run per-node.

        Args:
            state: Source state
            target_node: Target node ID

        Returns:
            Distance estimate
        """
        # Find source node for this state (linear search, inefficient)
        for node_id, states in self.training_data.node_states.items():
            for s in states:
                # Simple equality check - may not work for all state types
                if np.array_equal(np.array(s), np.array(state)):
                    return self.estimate_node_distance(node_id, target_node)

        # No match found, assume far away
        return float(self.max_steps_per_rollout)

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Estimate distance between nodes using rollout success rate.

        Returns (1 - success_rate) * max_steps, so higher success = lower distance.

        Args:
            source_node: Source node ID
            target_node: Target node ID

        Returns:
            Distance estimate
        """
        # print(self._success_counts, self._success_counts is None)
        if self._success_counts is None:
            raise RuntimeError("Must call multi_train() before estimate_node_distance()")

        # Get success count
        success_count = self._success_counts.get((source_node, target_node), 0)

        # Compute success rate
        success_rate = success_count / self.num_rollouts if self.num_rollouts > 0 else 0.0

        # Return inverse scaled by max_steps (higher success = lower distance)
        distance = (1.0 - success_rate) * self.max_steps_per_rollout

        return distance

    def prune(self, max_shortcuts: int | None, **kwargs: Any) -> "GoalConditionedTrainingData":
        """Prune shortcuts based on rollout success rate.

        Keeps only shortcuts where success_rate >= threshold.

        Args:
            **kwargs: Can override threshold with 'threshold' parameter

        Returns:
            Pruned training data
        """
        if self._success_counts is None:
            raise RuntimeError("Must call multi_train() before prune()")

        # Allow overriding threshold
        threshold = kwargs.get("threshold", self.threshold)

        print(f"\nPruning with rollouts (threshold={threshold})")

        # Compute success rates and select shortcuts (use unique_shortcuts for node-node pairs)
        selected_unique_shortcuts = []
        for source_id, target_id in self.training_data.unique_shortcuts:
            success_count = self._success_counts.get((source_id, target_id), 0)
            success_rate = success_count / self.num_rollouts if self.num_rollouts > 0 else 0.0

            if success_rate >= threshold:
                selected_unique_shortcuts.append((source_id, target_id))

        pruned_count = len(self.training_data.unique_shortcuts) - len(selected_unique_shortcuts)
        print(f"Pruned {pruned_count} unique shortcuts")
        print(f"Kept {len(selected_unique_shortcuts)} unique shortcuts")

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
                "pruning_method": "rollouts",
                "rollout_threshold": threshold,
                "num_rollouts": self.num_rollouts,
            },
        )

        return random_selection(
            pruned_data,
            max_shortcuts=max_shortcuts,
            rng=self.rng,
        )
