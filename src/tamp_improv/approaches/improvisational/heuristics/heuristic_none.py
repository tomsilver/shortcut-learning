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


class NoneHeuristic(BaseHeuristic):
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
        rng: np.random.Generator,
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
        self.training_data = training_data
        self.graph_distances = graph_distances
        self.rng = rng

    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Run rollouts to evaluate all shortcuts.

        This performs the actual work of the rollouts heuristic by executing
        random rollouts and counting successes.

        Returns:
            Dictionary with rollout statistics
        """
        print("None Heuristic: no training performed.")
        return {}

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
        return 0

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
        return 0

    def prune(self, max_shortcuts: int | None, **kwargs: Any) -> "GoalConditionedTrainingData":
        """Prune shortcuts based on rollout success rate.

        Keeps only shortcuts where success_rate >= threshold.

        Args:
            **kwargs: Can override threshold with 'threshold' parameter

        Returns:
            Pruned training data
        """

        if max_shortcuts is None:
            return self.training_data

        pruned_training_data = random_selection(self.training_data,
                                                max_shortcuts=max_shortcuts,
                                                rng=self.rng)


        return pruned_training_data
