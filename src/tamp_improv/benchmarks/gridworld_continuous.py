"""Continuous gridworld environment for distance heuristic learning.

This environment provides a continuous version of the gridworld where:
- High-level: C×C grid of cells (abstract nodes)
- Low-level: Continuous (x, y) coordinates in [0, grid_size] x [0, grid_size]
- Observation: GraphInstance with single node containing robot position [x, y]
- Actions: Continuous velocity (vx, vy) clipped to maximum magnitude
- Skills: Move horizontally/vertically between adjacent cells (no diagonals)
- Portals: Point pairs where being within radius r triggers teleportation

Key differences from gridworld_fixed:
- Continuous state space (float x, y)
- Continuous action space (velocity vx, vy)
- Portal activation based on proximity (radius) instead of exact position
- Shortcuts can include diagonal movement (not just horizontal/vertical)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Graph, GraphInstance
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    PDDLDomain,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver, Skill

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
    PredicateContainer,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper

# ============================================================================
# Gridworld Continuous Gymnasium Environment
# ============================================================================


class GridworldContinuousEnv(gym.Env):
    """A continuous hierarchical gridworld with portals.

    State space:
    - High-level: C×C cells
    - Low-level: Continuous (x, y) in [0, grid_size] x [0, grid_size]

    Key features:
    - Continuous position and velocity
    - Portals activated by proximity (within radius)
    - Abstract nodes are grid cells
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_cells: int = 3,
        grid_size: float = 10.0,
        max_velocity: float = 1.0,
        portal_radius: float = 0.5,
        num_teleporters: int = 1,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
    ):
        """Initialize continuous gridworld.

        Args:
            num_cells: Number of cells in each dimension (C)
            grid_size: Size of the grid in each dimension
            max_velocity: Maximum velocity magnitude
            portal_radius: Radius within which portal activates
            num_teleporters: Number of portal pairs
            render_mode: Rendering mode
            max_episode_steps: Maximum steps before episode ends
            seed: Random seed for initializing portal locations
        """
        super().__init__()
        self.num_cells = num_cells
        self.grid_size = grid_size
        self.max_velocity = max_velocity
        self.portal_radius = portal_radius
        self.num_teleporters = num_teleporters
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Cell size
        self.cell_size = grid_size / num_cells

        # Observation: GraphInstance with single node [x, y]
        self.observation_space = Graph(
            node_space=Box(
                low=0.0, high=grid_size, shape=(2,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Actions: continuous velocity (vx, vy)
        self.action_space = spaces.Box(
            low=-max_velocity,
            high=max_velocity,
            shape=(2,),
            dtype=np.float32,
        )

        # Initialize FIXED portal locations
        self.portal_positions: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = (
            []
        )

        # Use temporary RNG for initialization
        init_rng = np.random.default_rng(seed)

        # Get all cells
        all_cells = [(i, j) for i in range(num_cells) for j in range(num_cells)]

        # Randomly select cell pairs for portals
        # Constraint: no two portal cells can be adjacent (Manhattan dist <= 1)
        available_cells = all_cells.copy()
        init_rng.shuffle(available_cells)

        occupied_cells = set()
        portals_found = 0

        while portals_found < num_teleporters and len(available_cells) >= 2:
            # Find first valid cell1
            c1_idx = -1
            cell1 = None
            for idx, c in enumerate(available_cells):
                if any(
                    abs(c[0] - oc[0]) + abs(c[1] - oc[1]) <= 1 for oc in occupied_cells
                ):
                    continue
                c1_idx = idx
                cell1 = c
                break

            if cell1 is None:
                break

            # Find first valid cell2
            c2_idx = -1
            cell2 = None
            for idx, c in enumerate(available_cells):
                if idx == c1_idx:
                    continue
                if any(
                    abs(c[0] - oc[0]) + abs(c[1] - oc[1]) <= 1 for oc in occupied_cells
                ):
                    continue
                if abs(c[0] - cell1[0]) + abs(c[1] - cell1[1]) <= 1:
                    continue
                c2_idx = idx
                cell2 = c
                break

            if cell2 is not None:
                occupied_cells.add(cell1)
                occupied_cells.add(cell2)

                # Place portals at center of each cell
                portal1_pos = np.array(
                    [
                        (cell1[0] + 0.5) * self.cell_size,
                        (cell1[1] + 0.5) * self.cell_size,
                    ],
                    dtype=np.float32,
                )

                portal2_pos = np.array(
                    [
                        (cell2[0] + 0.5) * self.cell_size,
                        (cell2[1] + 0.5) * self.cell_size,
                    ],
                    dtype=np.float32,
                )

                self.portal_positions.append((portal1_pos, portal2_pos))

                # Remove used cells
                if c1_idx > c2_idx:
                    available_cells.pop(c1_idx)
                    available_cells.pop(c2_idx)
                else:
                    available_cells.pop(c2_idx)
                    available_cells.pop(c1_idx)

                portals_found += 1
            else:
                available_cells.pop(c1_idx)

        # State variables (randomized each reset)
        self.robot_pos: NDArray[np.float32] | None = None
        self.goal_cell: tuple[int, int] | None = None
        self.step_count: int = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment.

        Robot starts at a random position. Goal is a random cell.
        Portals remain fixed at their initialized locations.
        """
        super().reset(seed=seed)

        # Robot starts randomly anywhere in the grid
        self.robot_pos = self.np_random.uniform(
            low=0.0, high=self.grid_size, size=2
        ).astype(np.float32)

        # Goal is a random cell
        self.goal_cell = tuple(self.np_random.integers(0, self.num_cells, size=2))

        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def reset_from_state(
        self, state: GraphInstance, seed: int | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to match a specific low-level state.

        Args:
            state: GraphInstance observation containing robot position [x, y]

        Returns:
            Observation and info dict
        """
        # Extract robot position from GraphInstance
        robot_node = state.nodes[0]
        self.robot_pos = np.array(
            [float(robot_node[0]), float(robot_node[1])], dtype=np.float32
        )

        # Goal cell remains the same (set by previous reset)
        if self.goal_cell is None:
            self.goal_cell = (self.num_cells - 1, self.num_cells - 1)

        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[GraphInstance, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Velocity (vx, vy) - will be clipped to max magnitude
        """
        assert self.robot_pos is not None
        assert self.goal_cell is not None

        # Clip action to max velocity magnitude
        action = np.array(action, dtype=np.float32)
        velocity_magnitude = np.linalg.norm(action)
        if velocity_magnitude > self.max_velocity:
            action = action * (self.max_velocity / velocity_magnitude)

        # Apply velocity to position
        new_pos = self.robot_pos + action

        # Clip to grid bounds
        new_pos = np.clip(new_pos, 0.0, self.grid_size - 1e-6).astype(np.float32)

        self.robot_pos = new_pos

        # Check for portal teleportation (unidirectional: portal1 -> portal2 only)
        for portal1, portal2 in self.portal_positions:
            dist_to_p1 = np.linalg.norm(self.robot_pos - portal1)

            if dist_to_p1 < self.portal_radius:
                self.robot_pos = portal2.copy()
                break

        self.step_count += 1

        # Check if goal cell reached
        robot_cell = self._get_cell(self.robot_pos)
        terminated = robot_cell == self.goal_cell

        # Check if max steps exceeded
        truncated = self.step_count >= self.max_episode_steps

        # Reward: -1 per step, +100 for reaching goal cell
        reward = 100.0 if terminated else -1.0

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> GraphInstance:
        """Get current observation as a graph with single node."""
        assert self.robot_pos is not None

        robot_node = np.array(
            [self.robot_pos[0], self.robot_pos[1]],
            dtype=np.float32,
        )

        return GraphInstance(
            nodes=robot_node.reshape(1, -1), edges=None, edge_links=None
        )

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        robot_cell = self._get_cell(self.robot_pos)
        return {
            "robot_cell": robot_cell,
            "goal_cell": self.goal_cell,
            "step_count": self.step_count,
        }

    def extract_relevant_object_features(
        self, obs: GraphInstance, relevant_object_names: set[str]
    ) -> NDArray[np.float32]:
        """Extract features from observation."""
        if not hasattr(obs, "nodes"):
            return obs
        return obs.nodes.flatten()

    def _get_cell(self, pos: NDArray[np.float32] | None) -> tuple[int, int]:
        """Get cell coordinates for a position."""
        if pos is None:
            return (-1, -1)
        cell_x = int(pos[0] / self.cell_size)
        cell_y = int(pos[1] / self.cell_size)
        # Clamp to valid range
        cell_x = max(0, min(self.num_cells - 1, cell_x))
        cell_y = max(0, min(self.num_cells - 1, cell_y))
        return (cell_x, cell_y)

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode != "human":
            return

        # Simple text-based rendering
        resolution = 20  # Characters per cell
        total_chars = self.num_cells * resolution

        print("\n" + "=" * (total_chars + 3))
        for y_idx in range(total_chars - 1, -1, -1):
            row = "|"
            y = (y_idx + 0.5) / resolution * self.cell_size
            for x_idx in range(total_chars):
                x = (x_idx + 0.5) / resolution * self.cell_size
                pos = np.array([x, y])
                cell = self._get_cell(pos)

                # Check if robot is here
                if self.robot_pos is not None:
                    robot_x_idx = int(self.robot_pos[0] / self.cell_size * resolution)
                    robot_y_idx = int(self.robot_pos[1] / self.cell_size * resolution)
                    if x_idx == robot_x_idx and y_idx == robot_y_idx:
                        row += "R"
                        continue

                # Check if goal cell
                if cell == self.goal_cell:
                    row += "g"
                    continue

                # Check portals
                is_portal = False
                for portal1, portal2 in self.portal_positions:
                    if (
                        np.linalg.norm(pos - portal1) < self.portal_radius
                        or np.linalg.norm(pos - portal2) < self.portal_radius
                    ):
                        row += "P"
                        is_portal = True
                        break
                if is_portal:
                    continue

                # Cell boundaries
                if x_idx % resolution == 0 or y_idx % resolution == 0:
                    row += "+"
                else:
                    row += "."

            row += "|"
            print(row)
        print("=" * (total_chars + 3))
        robot_cell = self._get_cell(self.robot_pos)
        print(
            f"Step: {self.step_count}, Robot: ({self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}), "
            f"Cell: {robot_cell}, Goal Cell: {self.goal_cell}"
        )


# ============================================================================
# PDDL Types and Predicates (same as gridworld_fixed)
# ============================================================================


class GridworldContinuousTypes:
    """Types for gridworld continuous."""

    robot = Type("robot")


class GridworldContinuousPredicates(PredicateContainer):
    """Predicates for gridworld continuous - only spatial predicates, no goal."""

    def __init__(self, num_cells: int):
        """Initialize predicates.

        Creates:
        - InRow0, InRow1, ..., InRow(C-1)
        - InCol0, InCol1, ..., InCol(C-1)
        """
        self.num_cells = num_cells

        # Row predicates
        self.row_preds = []
        for i in range(num_cells):
            pred = Predicate(f"InRow{i}", [GridworldContinuousTypes.robot])
            setattr(self, f"InRow{i}", pred)
            self.row_preds.append(pred)

        # Column predicates
        self.col_preds = []
        for j in range(num_cells):
            pred = Predicate(f"InCol{j}", [GridworldContinuousTypes.robot])
            setattr(self, f"InCol{j}", pred)
            self.col_preds.append(pred)

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return getattr(self, key)

    def as_set(self) -> set[Predicate]:
        """Convert to set."""
        return set(self.row_preds + self.col_preds)


# ============================================================================
# Perceiver
# ============================================================================


class GridworldContinuousPerceiver(Perceiver[GraphInstance]):
    """Perceiver for gridworld continuous that maps observations to cell-based
    atoms."""

    def __init__(self, num_cells: int, cell_size: float):
        """Initialize perceiver.

        Args:
            num_cells: Number of cells in each dimension
            cell_size: Size of each cell
        """
        self.num_cells = num_cells
        self.cell_size = cell_size
        self.predicates = GridworldContinuousPredicates(num_cells)
        self.robot_obj = Object("robot0", GridworldContinuousTypes.robot)

    def reset(
        self, obs: GraphInstance, info: dict[str, Any]
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver and get initial objects, atoms, and goal."""
        objects = {self.robot_obj}

        # Get current atoms based on robot cell
        atoms = self._get_atoms_from_obs(obs)

        # Goal: robot in the goal cell (from info)
        goal_cell = info["goal_cell"]
        goal_atoms = {
            GroundAtom(self.predicates.row_preds[goal_cell[1]], [self.robot_obj]),
            GroundAtom(self.predicates.col_preds[goal_cell[0]], [self.robot_obj]),
        }

        return objects, atoms, goal_atoms

    def step(self, obs: GraphInstance) -> set[GroundAtom]:
        """Get atoms from observation."""
        return self._get_atoms_from_obs(obs)

    def _get_atoms_from_obs(self, obs: GraphInstance) -> set[GroundAtom]:
        """Convert graph observation to ground atoms."""
        robot_node = obs.nodes[0]
        robot_x, robot_y = robot_node[0], robot_node[1]

        atoms = set()

        # Determine cell
        cell_col = int(robot_x / self.cell_size)
        cell_row = int(robot_y / self.cell_size)

        # Clamp to valid range
        cell_col = max(0, min(self.num_cells - 1, cell_col))
        cell_row = max(0, min(self.num_cells - 1, cell_row))

        # Add row and column predicates
        atoms.add(GroundAtom(self.predicates.row_preds[cell_row], [self.robot_obj]))
        atoms.add(GroundAtom(self.predicates.col_preds[cell_col], [self.robot_obj]))

        return atoms


# ============================================================================
# Skills (move in straight lines - horizontal or vertical only)
# ============================================================================


class BaseGridworldContinuousSkill(LiftedOperatorSkill[GraphInstance, NDArray]):
    """Base class for gridworld continuous skills."""

    def __init__(self, components: PlanningComponents[GraphInstance]):
        """Initialize skill."""
        super().__init__()
        self._components = components
        self._lifted_operator = self._get_lifted_operator()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        op_name = self._get_operator_name()
        return next(op for op in self._components.operators if op.name == op_name)

    def _get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError


class MoveUpContinuousSkill(BaseGridworldContinuousSkill):
    """Move up one cell (vertical line movement)."""

    def __init__(
        self,
        components: PlanningComponents[GraphInstance],
        from_row: int,
        cell_size: float,
        max_velocity: float,
    ):
        """Initialize skill."""
        self.from_row = from_row
        self.cell_size = cell_size
        self.max_velocity = max_velocity
        super().__init__(components)

    def _get_operator_name(self) -> str:
        return f"MoveUp_from_row{self.from_row}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,
    ) -> NDArray[np.float32]:
        """Move up by applying positive y velocity."""
        robot_node = obs.nodes[0]
        robot_y = robot_node[1]

        # Target is the next cell boundary (moving up)
        target_y = (self.from_row + 1) * self.cell_size + 0.1

        if robot_y < target_y:
            # Move up (only vertical, no horizontal)
            return np.array([0.0, self.max_velocity], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)


class MoveDownContinuousSkill(BaseGridworldContinuousSkill):
    """Move down one cell (vertical line movement)."""

    def __init__(
        self,
        components: PlanningComponents[GraphInstance],
        from_row: int,
        cell_size: float,
        max_velocity: float,
    ):
        """Initialize skill."""
        self.from_row = from_row
        self.cell_size = cell_size
        self.max_velocity = max_velocity
        super().__init__(components)

    def _get_operator_name(self) -> str:
        return f"MoveDown_from_row{self.from_row}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,
    ) -> NDArray[np.float32]:
        """Move down by applying negative y velocity."""
        robot_node = obs.nodes[0]
        robot_y = robot_node[1]

        # Target is the cell below
        target_y = (self.from_row - 1) * self.cell_size + self.cell_size - 0.1

        if robot_y > target_y:
            # Move down (only vertical, no horizontal)
            return np.array([0.0, -self.max_velocity], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)


class MoveRightContinuousSkill(BaseGridworldContinuousSkill):
    """Move right one cell (horizontal line movement)."""

    def __init__(
        self,
        components: PlanningComponents[GraphInstance],
        from_col: int,
        cell_size: float,
        max_velocity: float,
    ):
        """Initialize skill."""
        self.from_col = from_col
        self.cell_size = cell_size
        self.max_velocity = max_velocity
        super().__init__(components)

    def _get_operator_name(self) -> str:
        return f"MoveRight_from_col{self.from_col}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,
    ) -> NDArray[np.float32]:
        """Move right by applying positive x velocity."""
        robot_node = obs.nodes[0]
        robot_x = robot_node[0]

        # Target is the next cell boundary (moving right)
        target_x = (self.from_col + 1) * self.cell_size + 0.1

        if robot_x < target_x:
            # Move right (only horizontal, no vertical)
            return np.array([self.max_velocity, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)


class MoveLeftContinuousSkill(BaseGridworldContinuousSkill):
    """Move left one cell (horizontal line movement)."""

    def __init__(
        self,
        components: PlanningComponents[GraphInstance],
        from_col: int,
        cell_size: float,
        max_velocity: float,
    ):
        """Initialize skill."""
        self.from_col = from_col
        self.cell_size = cell_size
        self.max_velocity = max_velocity
        super().__init__(components)

    def _get_operator_name(self) -> str:
        return f"MoveLeft_from_col{self.from_col}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,
    ) -> NDArray[np.float32]:
        """Move left by applying negative x velocity."""
        robot_node = obs.nodes[0]
        robot_x = robot_node[0]

        # Target is the cell to the left
        target_x = (self.from_col - 1) * self.cell_size + self.cell_size - 0.1

        if robot_x > target_x:
            # Move left (only horizontal, no vertical)
            return np.array([-self.max_velocity, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)


# ============================================================================
# TAMP System
# ============================================================================


class GridworldContinuousTAMPSystem(ImprovisationalTAMPSystem[GraphInstance, NDArray]):
    """Continuous gridworld TAMP system for distance heuristic learning."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        num_cells: int = 3,
        grid_size: float = 10.0,
        max_velocity: float = 1.0,
        portal_radius: float = 0.5,
        num_teleporters: int = 1,
        seed: int | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
    ):
        """Initialize gridworld continuous system."""
        self.num_cells = num_cells
        self.grid_size = grid_size
        self.max_velocity = max_velocity
        self.portal_radius = portal_radius
        self.num_teleporters = num_teleporters
        self.max_episode_steps = max_episode_steps
        self._env_seed = seed
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_env(self) -> gym.Env:
        """Create base gridworld continuous environment."""
        return GridworldContinuousEnv(
            num_cells=self.num_cells,
            grid_size=self.grid_size,
            max_velocity=self.max_velocity,
            portal_radius=self.portal_radius,
            num_teleporters=self.num_teleporters,
            render_mode=self._render_mode,
            max_episode_steps=self.max_episode_steps,
            seed=self._env_seed,
        )

    def _create_wrapped_env(
        self, components: PlanningComponents[GraphInstance]
    ) -> gym.Env:
        """Create wrapped environment for training shortcuts."""
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            max_episode_steps=self.max_episode_steps,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "gridworld_continuous"

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @classmethod
    def create_default(
        cls,
        num_cells: int = 3,
        grid_size: float = 10.0,
        max_velocity: float = 1.0,
        portal_radius: float = 0.5,
        num_teleporters: int = 1,
        seed: int = 42,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
    ) -> GridworldContinuousTAMPSystem:
        """Create default gridworld continuous system."""
        cell_size = grid_size / num_cells

        predicates = GridworldContinuousPredicates(num_cells)
        perceiver = GridworldContinuousPerceiver(num_cells, cell_size)

        # Create operators
        robot = Variable("?r", GridworldContinuousTypes.robot)
        operators = set()

        # MoveUp operators (one for each row except the top)
        for row in range(num_cells - 1):
            operator = LiftedOperator(
                name=f"MoveUp_from_row{row}",
                parameters=[robot],
                preconditions={predicates.row_preds[row]([robot])},
                add_effects={predicates.row_preds[row + 1]([robot])},
                delete_effects={predicates.row_preds[row]([robot])},
            )
            operators.add(operator)

        # MoveDown operators (one for each row except the bottom)
        for row in range(1, num_cells):
            operator = LiftedOperator(
                name=f"MoveDown_from_row{row}",
                parameters=[robot],
                preconditions={predicates.row_preds[row]([robot])},
                add_effects={predicates.row_preds[row - 1]([robot])},
                delete_effects={predicates.row_preds[row]([robot])},
            )
            operators.add(operator)

        # MoveRight operators (one for each column except the rightmost)
        for col in range(num_cells - 1):
            operator = LiftedOperator(
                name=f"MoveRight_from_col{col}",
                parameters=[robot],
                preconditions={predicates.col_preds[col]([robot])},
                add_effects={predicates.col_preds[col + 1]([robot])},
                delete_effects={predicates.col_preds[col]([robot])},
            )
            operators.add(operator)

        # MoveLeft operators (one for each column except the leftmost)
        for col in range(1, num_cells):
            operator = LiftedOperator(
                name=f"MoveLeft_from_col{col}",
                parameters=[robot],
                preconditions={predicates.col_preds[col]([robot])},
                add_effects={predicates.col_preds[col - 1]([robot])},
                delete_effects={predicates.col_preds[col]([robot])},
            )
            operators.add(operator)

        # Create planning components
        components = PlanningComponents(
            types={GridworldContinuousTypes.robot},
            predicate_container=predicates,
            skills=set(),
            perceiver=perceiver,
            operators=operators,
        )

        # Create skills
        skills = set()
        for row in range(num_cells - 1):
            skills.add(
                MoveUpContinuousSkill(components, from_row=row, cell_size=cell_size, max_velocity=max_velocity)
            )
        for row in range(1, num_cells):
            skills.add(
                MoveDownContinuousSkill(components, from_row=row, cell_size=cell_size, max_velocity=max_velocity)
            )
        for col in range(num_cells - 1):
            skills.add(
                MoveRightContinuousSkill(components, from_col=col, cell_size=cell_size, max_velocity=max_velocity)
            )
        for col in range(1, num_cells):
            skills.add(
                MoveLeftContinuousSkill(components, from_col=col, cell_size=cell_size, max_velocity=max_velocity)
            )

        # Update components with skills
        components.skills = skills

        return cls(
            planning_components=components,
            num_cells=num_cells,
            grid_size=grid_size,
            max_velocity=max_velocity,
            portal_radius=portal_radius,
            num_teleporters=num_teleporters,
            seed=seed,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
