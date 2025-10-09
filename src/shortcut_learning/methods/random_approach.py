"""Random action approach."""

from typing import Any

from shortcut_learning.configs import (
<<<<<<< HEAD
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
=======
>>>>>>> 5ea7ae89167f8c1d86addab3512337e6d04bc515
    TrainingConfig,
)
from shortcut_learning.methods.base_approach import (
    ActType,
    ApproachStepResult,
    BaseApproach,
    ObsType,
)
from shortcut_learning.methods.training_data import TrainingData


class RandomApproach(BaseApproach[ObsType, ActType]):
    """An approach that takes random actions."""

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach."""
        return ApproachStepResult(action=self.system.env.action_space.sample())

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Take random action."""
        return ApproachStepResult(action=self.system.env.action_space.sample())

    def train(self, train_data: TrainingData | None, config: TrainingConfig) -> None:
        """Train approach with optional training data."""
        return
