"""Pass-through heuristic that performs no training."""

from typing import TYPE_CHECKING, Any

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


class NoneHeuristic(BaseHeuristic):
    """Pass-through heuristic that does no training and keeps all shortcuts."""

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        rng: np.random.Generator,
    ):
        super().__init__(training_data, graph_distances)
        self.system = system
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

    def prune(
        self, max_shortcuts: int | None, **kwargs: Any
    ) -> "GoalConditionedTrainingData":
        """Prune shortcuts based on rollout success rate.

        Keeps only shortcuts where success_rate >= threshold.

        Args:
            **kwargs: Can override threshold with 'threshold' parameter

        Returns:
            Pruned training data
        """

        if max_shortcuts is None:
            return self.training_data

        pruned_training_data = random_selection(
            self.training_data, max_shortcuts=max_shortcuts, rng=self.rng
        )

        return pruned_training_data

    def train_one_round(self, **kwargs) -> dict[str, Any]:
        return self.multi_train()

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def prune_by_success(
        self, success_threshold: float, max_steps: int, **kwargs: Any
    ) -> "GoalConditionedTrainingData":
        raise NotImplementedError("NoneHeuristic does not support multi-round pruning.")

    def update_system(self, **kwargs: Any) -> None:
        raise NotImplementedError("NoneHeuristic does not support update_system.")

    def get_action(self, obs: "ObsType", target_node: int) -> np.ndarray | int:
        raise NotImplementedError("get_action is not implemented for NoneHeuristic.")