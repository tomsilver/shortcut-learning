"""Pure RL baseline approach without using TAMP structure."""

from pathlib import Path
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from shortcut_learning.configs import (
    TrainingConfig,
)
from shortcut_learning.methods.base_approach import ApproachStepResult, BaseApproach
from shortcut_learning.methods.policies.base import Policy
from shortcut_learning.methods.training_data import TrainingData
from shortcut_learning.problems.base_tamp import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class PureRLWrapper(gym.Wrapper):
    """Wrapper for training pure RL baselines without TAMP structure."""

    def __init__(
        self,
        env: gym.Env,
        perceiver: Perceiver[ObsType],
        goal_atoms: set[GroundAtom],
        *,
        max_episode_steps: int = 100,
        step_penalty: float = -0.1,
        achievement_bonus: float = 1.0,
        action_scale: float = 1.0,
    ) -> None:
        """Initialize wrapper for pure RL training."""
        super().__init__(env)
        self.perceiver = perceiver
        self.goal_atoms = goal_atoms
        self.max_episode_steps = max_episode_steps
        self.step_penalty = step_penalty
        self.achievement_bonus = achievement_bonus
        self.action_scale = action_scale
        self.steps = 0
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=env.action_space.low * action_scale,
                high=env.action_space.high * action_scale,
                dtype=np.float32,
            )
        else:
            print("Warning: Action space is not Box, using original action space.")
            self.action_space = env.action_space
        self._render_mode = getattr(env, "render_mode", None)

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        self.steps = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1

        current_atoms = self.perceiver.step(obs)
        achieved = self.goal_atoms.issubset(current_atoms)

        reward = self.step_penalty
        if achieved:
            reward += self.achievement_bonus

        terminated = achieved
        truncated = truncated or self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, info


class PureRLApproach(BaseApproach[ObsType, ActType]):
    """Pure RL approach that doesn't use TAMP structure."""

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        seed: int,
        name: str,
        policy: Policy[ObsType, ActType],
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed, name)
        self.policy = policy

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""
        action = self.policy.get_action(obs)
        return ApproachStepResult(action=action)

    def train(self, train_data: TrainingData | None, config: TrainingConfig):

        obs, info = self.system.reset()
        _, _, goal_atoms = self.system.perceiver.reset(obs, info)

        wrapped_env = PureRLWrapper(
            env=self.system.env,
            perceiver=self.system.perceiver,
            goal_atoms=goal_atoms,
            max_episode_steps=config.max_steps,
            step_penalty=config.step_penalty,
            achievement_bonus=config.success_reward,
            action_scale=config.action_scale,
        )

        render_mode = getattr(wrapped_env, "_render_mode", None)
        can_render = render_mode is not None
        if config.record_training and can_render:
            video_folder = Path(f"videos/{self.system.name}_{self.name}_train")
            video_folder.mkdir(parents=True, exist_ok=True)
            wrapped_env = RecordVideo(
                wrapped_env,  # type: ignore[assignment]
                str(video_folder),
                episode_trigger=lambda x: x % config.training_record_interval == 0,
                name_prefix="training",
            )

        self.policy.initialize(wrapped_env)

        self.policy.train(wrapped_env, train_data=None)
