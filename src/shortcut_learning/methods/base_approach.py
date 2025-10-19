"""Base class for all approaches."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from shortcut_learning.configs import (
    TrainingConfig,
)
from shortcut_learning.methods.training_data import TrainingData
from shortcut_learning.problems.base_tamp import BaseTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class ApproachStepResult(Generic[ActType]):
    """Result from an approach step."""

    action: ActType
    terminate: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseApproach(Generic[ObsType, ActType], ABC):
    """Base class for all approaches."""

    def __init__(
        self, system: BaseTAMPSystem[ObsType, ActType], seed: int, name: str
    ) -> None:
        """Initialize approach.

        Args:
            system: The TAMP system to use
            seed: Random seed
        """
        self.system = system
        self._seed = seed
        self._training_mode = False
        self.name = name

    @property
    def training_mode(self) -> bool:
        """Whether the approach is in training mode."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool) -> None:
        """Set training mode."""
        self._training_mode = value

    @abstractmethod
    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""

    @abstractmethod
    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""

    @abstractmethod
    def train(self, train_data: TrainingData | None, config: TrainingConfig) -> None:
        """Train approach with optional training data."""
