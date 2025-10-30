"""Obstacle Tower TAMP system implementation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray
from pybullet_blocks.envs.obstacle_tower_env import (
    GraphObstacleTowerPyBulletBlocksEnv,
    ObstacleTowerSceneDescription,
)
from pybullet_blocks.planning_models.action import OPERATORS, SKILLS
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    GraphObstacleTowerPyBulletBlocksPerceiver,
)
from relational_structs import PDDLDomain
from task_then_motion_planning.structs import Skill

from shortcut_learning.problems.base_tamp import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from shortcut_learning.problems.obstacle_tower.env import ObstacleTowerEnv


class BaseObstacleTowerTAMPSystem(
    BaseTAMPSystem[GraphInstance, NDArray]
):
    """Base TAMP system for ObstacleTower environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        num_obstacle_blocks: int = 3,
        stack_blocks: bool = True,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize ObstacleTower TAMP system.

        Args:
            planning_components: Planning components (types, predicates, operators, skills, perceiver)
            num_obstacle_blocks: Number of obstacle blocks
            stack_blocks: Whether to stack blocks vertically
            seed: Random seed
            render_mode: Rendering mode
        """
        self._num_obstacle_blocks = num_obstacle_blocks
        self._stack_blocks = stack_blocks
        self._render_mode = render_mode
        super().__init__(
            planning_components, name="ObstacleTowerTAMPSystem", seed=seed
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        return ObstacleTowerEnv(
            num_obstacle_blocks=self._num_obstacle_blocks,
            stack_blocks=self._stack_blocks,
            render_mode=self._render_mode,
            use_gui=False,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "obstacle-tower-domain"

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container,
            self.components.types,
        )

    @classmethod
    def create_default(
        cls,
        num_obstacle_blocks: int = 3,
        stack_blocks: bool = True,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BaseObstacleTowerTAMPSystem:
        """Factory method for creating system with default components.

        Args:
            num_obstacle_blocks: Number of obstacle blocks
            stack_blocks: Whether to stack blocks vertically
            seed: Random seed
            render_mode: Rendering mode

        Returns:
            Configured BaseObstacleTowerTAMPSystem instance
        """
        scene_description = ObstacleTowerSceneDescription(
            num_obstacle_blocks=num_obstacle_blocks,
            num_irrelevant_blocks=0,
            stack_blocks=stack_blocks,
        )

        # Create a temporary environment for skill initialization
        sim = GraphObstacleTowerPyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )

        # Initialize skills with the simulator
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1) for s in SKILLS  # type: ignore
        }
        skills: set[Skill[GraphInstance, NDArray]] = pybullet_skills  # type: ignore

        # Initialize perceiver
        perceiver = GraphObstacleTowerPyBulletBlocksPerceiver(sim)

        # Create planning components
        planning_components = PlanningComponents(
            types=set(TYPES),
            predicate_container=set(PREDICATES),
            operators=OPERATORS,
            skills=skills,
            perceiver=perceiver,
        )

        # Create and return the system
        system = cls(
            planning_components,
            num_obstacle_blocks=num_obstacle_blocks,
            stack_blocks=stack_blocks,
            seed=seed,
            render_mode=render_mode,
        )
        return system


class ObstacleTowerTAMPSystem(
    ImprovisationalTAMPSystem[GraphInstance, NDArray],
    BaseObstacleTowerTAMPSystem,
):
    """TAMP system for ObstacleTower environment with improvisational policy
    learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        num_obstacle_blocks: int = 3,
        stack_blocks: bool = True,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize ObstacleTower TAMP system with improvisation.

        Args:
            planning_components: Planning components
            num_obstacle_blocks: Number of obstacle blocks
            stack_blocks: Whether to stack blocks vertically
            seed: Random seed
            render_mode: Rendering mode
        """
        self._num_obstacle_blocks = num_obstacle_blocks
        self._stack_blocks = stack_blocks
        self._render_mode = render_mode
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_wrapped_env(
        self, components: PlanningComponents[GraphInstance]
    ) -> gym.Env:
        """Create wrapped environment for training.

        For now, we return the base environment without additional wrapping.
        In the future, this could include an ImprovWrapper similar to the
        original implementation.

        Args:
            components: Planning components

        Returns:
            Wrapped environment for policy training
        """
        # TODO: Consider adding ImprovWrapper if needed
        # return ImprovWrapper(
        #     base_env=self.env,
        #     perceiver=components.perceiver,
        #     step_penalty=-1.0,
        #     achievement_bonus=100.0,
        #     action_scale=0.015,
        # )
        return self.env

    @classmethod
    def create_default(
        cls,
        num_obstacle_blocks: int = 3,
        stack_blocks: bool = True,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> ObstacleTowerTAMPSystem:
        """Factory method for creating improvisational system with default components.

        Args:
            num_obstacle_blocks: Number of obstacle blocks
            stack_blocks: Whether to stack blocks vertically
            seed: Random seed
            render_mode: Rendering mode

        Returns:
            Configured ObstacleTowerTAMPSystem instance
        """
        scene_description = ObstacleTowerSceneDescription(
            num_obstacle_blocks=num_obstacle_blocks,
            num_irrelevant_blocks=0,
            stack_blocks=stack_blocks,
        )

        # Create a temporary environment for skill initialization
        sim = GraphObstacleTowerPyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )

        # Initialize skills with the simulator
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1) for s in SKILLS  # type: ignore
        }
        skills: set[Skill[GraphInstance, NDArray]] = pybullet_skills  # type: ignore

        # Initialize perceiver
        perceiver = GraphObstacleTowerPyBulletBlocksPerceiver(sim)

        # Create planning components
        planning_components = PlanningComponents(
            types=set(TYPES),
            predicate_container=set(PREDICATES),
            operators=OPERATORS,
            skills=skills,
            perceiver=perceiver,
        )

        # Create and return the system
        system = cls(
            planning_components,
            num_obstacle_blocks=num_obstacle_blocks,
            stack_blocks=stack_blocks,
            seed=seed,
            render_mode=render_mode,
        )
        return system
