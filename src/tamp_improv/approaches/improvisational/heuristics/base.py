"""Base class for shortcut heuristics."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData, ObsType


class BaseHeuristic(ABC):
    """Base class for all shortcut heuristics.

    All heuristics follow a unified interface:
    1. multi_train() - Train the heuristic (or do nothing for non-learning methods)
    2. estimate_distance() - Estimate distance from state to node
    3. estimate_node_distance() - Estimate distance between nodes
    4. prune() - Filter training data to promising shortcuts

    This allows the pipeline to treat all heuristic methods uniformly.

    The training data and graph distances are stored at initialization and
    used by all methods.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
    ):
        """Initialize heuristic with data.

        Args:
            training_data: Full training data from collect_total_shortcuts
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
        """
        self.training_data = training_data
        self.graph_distances = graph_distances

    @abstractmethod
    def multi_train(self, **kwargs: Any) -> dict[str, Any]:
        """Train the heuristic on shortcut data.

        For non-learning methods (e.g., rollouts), this may do nothing.

        Args:
            **kwargs: Additional training parameters (epochs, rounds, etc.)

        Returns:
            Dictionary with training history/metadata
        """
        pass

    @abstractmethod
    def estimate_distance(self, state: "ObsType", target_node: int) -> float:
        """Estimate distance from a state to a target node.

        Args:
            state: Source state
            target_node: Target node ID

        Returns:
            Estimated distance (in steps)
        """
        pass

    @abstractmethod
    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Estimate distance between two nodes.

        Args:
            source_node: Source node ID
            target_node: Target node ID

        Returns:
            Estimated distance (in steps)
        """
        pass

    @abstractmethod
    def prune(self, max_shortcuts: int | None, **kwargs: Any) -> 'GoalConditionedTrainingData':
        """Prune training data to promising shortcuts.

        Uses estimate_node_distance to evaluate shortcuts and filters
        to those worth training on.

        Args:
            **kwargs: Additional pruning parameters (threshold, keep_fraction, etc.)

        Returns:
            New GoalConditionedTrainingData with filtered shortcuts
        """
        pass

    def save(self, path: str) -> None:
        """Save heuristic to disk (optional, for learned heuristics)."""
        raise NotImplementedError("This heuristic does not support saving")

    def load(self, path: str) -> None:
        """Load heuristic from disk (optional, for learned heuristics)."""
        raise NotImplementedError("This heuristic does not support loading")


def random_selection(
    training_data: 'GoalConditionedTrainingData',
    max_shortcuts: int,
    rng: np.random.Generator,
) -> 'GoalConditionedTrainingData':
    """Stage 3.5: Randomly select up to max_shortcuts from pruned data.

    If max_shortcuts is 0, returns empty training data (pure planning mode).
    If max_shortcuts >= num shortcuts, returns all shortcuts unchanged.

    Note: We select from unique_shortcuts (node-node pairs) but keep all
    corresponding state-node pairs in valid_shortcuts for MultiRL training.

    Args:
        training_data: Pruned training data
        max_shortcuts: Maximum number of shortcuts to keep (0 = pure planning)
        rng: Random number generator

    Returns:
        Training data with randomly selected shortcuts
    """
    print("\n" + "=" * 80)
    print("STAGE 3.5: RANDOM SELECTION")
    print("=" * 80)

    num_shortcuts = len(training_data.unique_shortcuts)

    # Handle edge cases
    if max_shortcuts == 0:
        print("max_shortcuts_per_graph = 0: Using pure planning (no shortcuts)")
        # Return empty training data
        return GoalConditionedTrainingData(
            states=[],
            current_atoms=[],
            goal_atoms=[],
            valid_shortcuts=[],
            unique_shortcuts=[],
            node_states=training_data.node_states,
            node_atoms=training_data.node_atoms,
            graph=training_data.graph,
            config={
                **training_data.config,
                "random_selection": True,
                "max_shortcuts_per_graph": max_shortcuts,
            },
        )

    if max_shortcuts >= num_shortcuts:
        print(f"max_shortcuts_per_graph ({max_shortcuts}) >= num shortcuts ({num_shortcuts}): Keeping all shortcuts")
        return training_data

    # Random selection
    print(f"Randomly selecting {max_shortcuts} from {num_shortcuts} unique shortcuts")

    # Randomly select unique shortcuts (node-node pairs)
    print(training_data.unique_shortcuts)
    sample = (rng.choice(training_data.unique_shortcuts, size=max_shortcuts, replace=False).tolist())

    selected_unique_shortcuts = set(
        (x, y) for x, y in sample
    )

    # Filter valid_shortcuts to keep only those matching selected unique shortcuts
    # valid_shortcuts has one entry per state-node pair, so we filter by node pairs
    selected_indices = []
    for i, (source_id, target_id) in enumerate(training_data.valid_shortcuts):
        if (source_id, target_id) in selected_unique_shortcuts:
            selected_indices.append(i)

    # Filter training data
    original_shortcut_info = training_data.config.get("shortcut_info", [])
    selected_shortcut_info = (
        [original_shortcut_info[i] for i in selected_indices]
        if original_shortcut_info
        else []
    )

    selected_data = GoalConditionedTrainingData(
        states=[training_data.states[i] for i in selected_indices],
        current_atoms=[training_data.current_atoms[i] for i in selected_indices],
        goal_atoms=[training_data.goal_atoms[i] for i in selected_indices],
        valid_shortcuts=[training_data.valid_shortcuts[i] for i in selected_indices],
        unique_shortcuts=list(selected_unique_shortcuts),
        node_states=training_data.node_states,
        node_atoms=training_data.node_atoms,
        graph=training_data.graph,
        config={
            **training_data.config,
            "shortcut_info": selected_shortcut_info,
            "random_selection": True,
            "max_shortcuts_per_graph": max_shortcuts,
        },
    )

    print(f"Selected {len(selected_data.unique_shortcuts)} unique shortcuts")
    print(f"  ({len(selected_data.valid_shortcuts)} state-node pairs)")

    return selected_data