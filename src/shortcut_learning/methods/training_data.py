"""Training data."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from relational_structs import GroundAtom

from shortcut_learning.methods.graph_utils import (
    PlanningGraphNode,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class ShortcutSignature:
    """Domain-agnostic signature of a shortcut for matching purposes."""

    source_predicates: set[str]
    target_predicates: set[str]
    source_types: set[str]
    target_types: set[str]

    @classmethod
    def from_context(
        cls, source_atoms: set[GroundAtom], target_atoms: set[GroundAtom]
    ) -> ShortcutSignature:
        """Create signature from context."""
        source_preds = {atom.predicate.name for atom in source_atoms}
        target_preds = {atom.predicate.name for atom in target_atoms}

        source_types = set()
        for atom in source_atoms:
            for obj in atom.objects:
                source_types.add(obj.type.name)

        target_types = set()
        for atom in target_atoms:
            for obj in atom.objects:
                target_types.add(obj.type.name)

        return cls(source_preds, target_preds, source_types, target_types)

    def similarity(self, other: ShortcutSignature) -> float:
        """Calculate similarity score between signatures."""
        # Predicate similarity (Jaccard)
        source_pred_sim = len(self.source_predicates & other.source_predicates) / max(
            len(self.source_predicates | other.source_predicates), 1
        )
        target_pred_sim = len(self.target_predicates & other.target_predicates) / max(
            len(self.target_predicates | other.target_predicates), 1
        )

        # Object type similarity (Jaccard)
        source_type_sim = len(self.source_types & other.source_types) / max(
            len(self.source_types | other.source_types), 1
        )
        target_type_sim = len(self.target_types & other.target_types) / max(
            len(self.target_types | other.target_types), 1
        )

        # Overall similarity - weighted average
        return (
            0.3 * source_pred_sim
            + 0.3 * target_pred_sim
            + 0.2 * source_type_sim
            + 0.2 * target_type_sim
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another ShortcutSignature."""
        if not isinstance(other, ShortcutSignature):
            return False

        return (
            self.source_predicates == other.source_predicates
            and self.target_predicates == other.target_predicates
            and self.source_types == other.source_types
            and self.target_types == other.target_types
        )

    def __hash__(self) -> int:
        """Hash function for ShortcutSignature."""
        # Convert sets to frozensets for hashing
        return hash(
            (
                frozenset(self.source_predicates),
                frozenset(self.target_predicates),
                frozenset(self.source_types),
                frozenset(self.target_types),
            )
        )


@dataclass
class ShortcutCandidate:
    """Represents a potential shortcut in the planning graph."""

    source_node: PlanningGraphNode
    target_node: PlanningGraphNode
    source_atoms: set[GroundAtom]
    target_atoms: set[GroundAtom]
    source_state: Any


@dataclass
class TrainingData:
    """Container for policy training data."""

    states: list[Any]  # List of states where intervention needed
    current_atoms: list[set[GroundAtom]]
    goal_atoms: list[set[GroundAtom]]
    config: dict[str, Any]

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: Path) -> None:
        """Save training data to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save states
        states_path = path / "states.pkl"
        with open(states_path, "wb") as f:
            pickle.dump(self.states, f)

        # Save current atoms and goal atoms as pickle
        data_paths = {
            "current_atoms": self.current_atoms,
            "goal_atoms": self.goal_atoms,
        }
        for name, obj in data_paths.items():
            file_path = path / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        # Save config as JSON
        serializable_config = dict(self.config)

        if "atom_to_index" in serializable_config:
            # Convert keys to strings if they aren't already
            atom_to_index = {
                str(k): v for k, v in serializable_config["atom_to_index"].items()
            }
            serializable_config["atom_to_index"] = atom_to_index

        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TrainingData:
        """Load training data from disk."""
        # Load states
        states_path = path / "states.pkl"
        with open(states_path, "rb") as f:
            states = pickle.load(f)

        # Load current atoms and goal atoms
        data = {}
        data_names = [
            "current_atoms",
            "goal_atoms",
        ]
        for name in data_names:
            file_path = path / f"{name}.pkl"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    data[name] = pickle.load(f)
            else:
                print(f"Warning: {file_path} not found, using empty list.")
                data[name] = []

        # Load config
        config_path = path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "atom_to_index" in config:
            config["atom_to_index"] = {
                k: int(v) if isinstance(v, str) else v
                for k, v in config["atom_to_index"].items()
            }

        return cls(
            states=states,
            current_atoms=data["current_atoms"],
            goal_atoms=data["goal_atoms"],
            config=config,
        )


@dataclass
class GoalConditionedTrainingData(TrainingData, Generic[ObsType]):
    """Training data for goal-conditioned learning."""

    node_states: dict[int, ObsType] = field(default_factory=dict)
    valid_shortcuts: list[tuple[int, int]] = field(default_factory=list)
    node_atoms: dict[int, set[GroundAtom]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save training data including node states."""
        super().save(path)
        if self.node_states:
            with open(path / "node_states.pkl", "wb") as f:
                pickle.dump(self.node_states, f)
        if self.valid_shortcuts:
            with open(path / "valid_shortcuts.pkl", "wb") as f:
                pickle.dump(self.valid_shortcuts, f)
        if self.node_atoms:
            with open(path / "node_atoms.pkl", "wb") as f:
                pickle.dump(self.node_atoms, f)

    @classmethod
    def load(cls, path: Path) -> GoalConditionedTrainingData:
        """Load training data including node states."""
        train_data = super().load(path)
        node_states: dict[int, ObsType] = {}
        if (path / "node_states.pkl").exists():
            with open(path / "node_states.pkl", "rb") as f:
                node_states = pickle.load(f)
        valid_shortcuts: list[tuple[int, int]] = []
        if (path / "valid_shortcuts.pkl").exists():
            with open(path / "valid_shortcuts.pkl", "rb") as f:
                valid_shortcuts = pickle.load(f)
        node_atoms: dict[int, set[GroundAtom]] = {}
        if (path / "node_atoms.pkl").exists():
            with open(path / "node_atoms.pkl", "rb") as f:
                node_atoms = pickle.load(f)
        return cls(
            states=train_data.states,
            current_atoms=train_data.current_atoms,
            goal_atoms=train_data.goal_atoms,
            config=train_data.config,
            node_states=node_states,
            valid_shortcuts=valid_shortcuts,
            node_atoms=node_atoms,
        )
