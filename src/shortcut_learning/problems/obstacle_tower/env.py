"""Obstacle Tower environment wrapper.

This module provides a wrapper around the PyBullet ObstacleTower environment
from pybullet_blocks, adapting it for use in the shortcut learning framework.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray
from pybullet_blocks.envs.obstacle_tower_env import (
    GraphObstacleTowerPyBulletBlocksEnv,
    ObstacleTowerSceneDescription,
)


class ObstacleTowerEnv(gym.Env):
    """Wrapper for the PyBullet ObstacleTower environment.

    This environment requires clearing obstacle blocks from a target area
    before placing a goal block in that area. It uses PyBullet for 3D
    physics simulation with a robotic arm.

    The observation space is a graph where:
    - Nodes represent: robot, blocks (lettered), and target area
    - Each node contains relevant state information

    The action space is continuous (8D):
    - 7D for robot joint movements
    - 1D for gripper open/close
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        num_obstacle_blocks: int = 3,
        stack_blocks: bool = True,
        render_mode: str | None = None,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        """Initialize the ObstacleTower environment.

        Args:
            num_obstacle_blocks: Number of obstacle blocks to place in target area
            stack_blocks: Whether to stack blocks vertically or place side-by-side
            render_mode: Rendering mode (None or "rgb_array")
            use_gui: Whether to display PyBullet GUI
            seed: Random seed for reproducibility
        """
        self._scene_description = ObstacleTowerSceneDescription(
            num_obstacle_blocks=num_obstacle_blocks,
            num_irrelevant_blocks=0,
            stack_blocks=stack_blocks,
        )

        self._env = GraphObstacleTowerPyBulletBlocksEnv(
            scene_description=self._scene_description,
            render_mode=render_mode,
            use_gui=use_gui,
            seed=seed,
        )

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options for reset

        Returns:
            observation: Graph observation of initial state
            info: Additional information dictionary
        """
        return self._env.reset(seed=seed, options=options)

    def reset_from_state(
        self,
        state: GraphInstance,
        *,
        seed: int | None = None,
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to specific state.

        Args:
            state: Graph instance representing the state
            seed: Random seed

        Returns:
            observation: Graph observation
            info: Additional information dictionary
        """
        return self._env.reset_from_state(state, seed=seed)

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[GraphInstance, float, bool, bool, dict[str, Any]]:
        """Take environment step with given action.

        Args:
            action: 8D array (7D joint movements + 1D gripper)

        Returns:
            observation: Graph observation after step
            reward: Reward (1.0 if goal achieved, 0.0 otherwise)
            terminated: Whether episode is done (goal achieved)
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        return self._env.step(action)

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        return self._env.render()

    def close(self) -> None:
        """Clean up environment resources."""
        self._env.close()

    def clone(self) -> ObstacleTowerEnv:
        """Clone the environment with current state.

        Returns:
            New ObstacleTowerEnv instance with same state
        """
        clone_env = ObstacleTowerEnv(
            num_obstacle_blocks=self._scene_description.num_obstacle_blocks,
            stack_blocks=self._scene_description.stack_blocks,
            render_mode=self.render_mode,
            use_gui=False,
        )
        # Copy the state from the underlying PyBullet environment
        clone_env._env.set_state(self._env.get_state())
        return clone_env

    @property
    def unwrapped(self) -> GraphObstacleTowerPyBulletBlocksEnv:
        """Get the underlying PyBullet environment.

        Returns:
            The wrapped GraphObstacleTowerPyBulletBlocksEnv instance
        """
        return self._env
