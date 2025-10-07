"""Random action approach."""

from typing import Any

from shortcut_learning.methods.base_approach import (
    ActType,
    ApproachStepResult,
    BaseApproach,
    ObsType,
)

from shortcut_learning.methods.training_data import TrainingData

from shortcut_learning.configs import (
    ApproachConfig,
    PolicyConfig,
    CollectionConfig,
    TrainingConfig,
    EvaluationConfig
)

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

    def train(
        self,
        train_data: TrainingData | None,
        config: TrainingConfig
    ) -> None:
        """Train approach with optional training data."""
        return
    
