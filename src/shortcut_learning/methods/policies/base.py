"""Base policy interface for improvisational approaches."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
from relational_structs import GroundAtom

from shortcut_learning.methods.training_data import TrainingData

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class PolicyContext(Generic[ObsType, ActType]):
    """Context information passed from approach to policy."""

    goal_atoms: set[GroundAtom]
    current_atoms: set[GroundAtom]
    info: dict[str, Any] = field(default_factory=dict)


class Policy(Generic[ObsType, ActType], ABC):
    """Base class for policies."""

    node_states: dict[int, ObsType]

    def __init__(self, seed: int) -> None:
        """Initialize policy with environment."""
        self._seed = seed

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""

    @abstractmethod
    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""

    @abstractmethod
    def can_initiate(self) -> bool:
        """Check whether the policy can be executed given the current
        context."""

    @abstractmethod
    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure policy with context information."""

    def train(self, env: gym.Env, train_data: TrainingData | None) -> None:
        """Train the policy if needed.

        Default implementation just initializes the policy and updates
        context. Policies that need training should override this.
        """
        self.initialize(env)
        if hasattr(env, "configure_training"):
            env.configure_training(train_data)

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from disk."""
