"""Gridworld environment for testing SLAP with portals.

This environment provides a hierarchical gridworld where:
- High-level: C×C grid of cells
- Low-level: S×S states within each cell
- Observation: GraphInstance with nodes for robot, goal, and portals
- Actions: up, down, left, right, teleport
- Skills: MoveUp (one cell up), MoveRight (one cell right), NavigateToGoal
- Portals: Random pairs that enable teleportation shortcuts

Example with C=2, S=5:
- 4 high-level cells (2×2) = quadrants
- 25 low-level states (5×5) per cell
- Predicates: InRow0, InRow1, InCol0, InCol1
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
# Gridworld Gymnasium Environment
# ============================================================================


class GridworldEnv(gym.Env):
    """A hierarchical gridworld with portals for testing SLAP.

    State space:
    - High-level: C×C cells
    - Low-level: S×S positions within each cell
    - Total positions: (C*S) × (C*S)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_cells: int = 2,
        num_states_per_cell: int = 5,
        num_teleporters: int = 1,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        seed: int | None = None,
    ):
        """Initialize gridworld.

        Args:
            num_cells: Number of cells in each dimension (C)
            num_states_per_cell: Number of low-level states per cell (S)
            num_teleporters: Number of portal pairs
            render_mode: Rendering mode
            max_episode_steps: Maximum steps before episode ends
            seed: Random seed for initializing portal cell pairs
        """
        super().__init__()
        self.num_cells = num_cells
        self.num_states_per_cell = num_states_per_cell
        self.num_teleporters = num_teleporters
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Total grid size
        self.grid_size = num_cells * num_states_per_cell

        # Observation: GraphInstance with nodes for robot, goal, and portals
        # Each node: [type, x, y, cell_x, cell_y, id]
        self.observation_space = Graph(
            node_space=Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            edge_space=None,
        )

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=teleport
        self.action_space = spaces.Discrete(5)

        # Initialize portal cell pairs (locked for entire environment lifetime)
        # Each pair has one "end portal" in goal cell (C-1, C-1)
        # and one "start portal" in a random non-goal cell
        self.portal_cell_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

        # Use temporary RNG for initialization
        init_rng = np.random.default_rng(seed)
        goal_cell = (num_cells - 1, num_cells - 1)

        # Get all non-goal cells for start portals
        all_cells = [(i, j) for i in range(num_cells) for j in range(num_cells)]
        non_goal_cells = [c for c in all_cells if c != goal_cell]

        # Randomly select start cells for portals (without replacement)
        start_cells = init_rng.choice(len(non_goal_cells), size=num_teleporters, replace=False)

        for idx in start_cells:
            start_cell = non_goal_cells[idx]
            # Each portal pair: (start_cell, goal_cell)
            self.portal_cell_pairs.append((start_cell, goal_cell))

        # State variables (positions within cells, randomized each reset)
        self.robot_pos: NDArray[np.int32] | None = None
        self.goal_pos: NDArray[np.int32] | None = None
        self.portals: list[tuple[NDArray[np.int32], NDArray[np.int32]]] = []  # Pairs
        self.step_count: int = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment.

        Portal locations are randomized within their locked cell pairs,
        ensuring shortcuts remain consistent across resets.
        """
        super().reset(seed=seed)

        cell_size = self.num_states_per_cell

        # Robot starts randomly in cell (0, 0)
        self.robot_pos = self.np_random.integers(0, cell_size, size=2)

        # Goal randomly in cell (C-1, C-1)
        goal_cell_start = (self.num_cells - 1) * cell_size
        goal_cell_end = self.num_cells * cell_size
        self.goal_pos = self.np_random.integers(goal_cell_start, goal_cell_end, size=2)

        # Place portals within their locked cell pairs
        self.portals = []
        for start_cell, end_cell in self.portal_cell_pairs:
            # Start portal: random position within start_cell
            start_cell_min = np.array(start_cell) * cell_size
            start_cell_max = start_cell_min + cell_size
            start_portal = self.np_random.integers(start_cell_min, start_cell_max, size=2)

            # End portal: random position within end_cell (goal cell)
            end_cell_min = np.array(end_cell) * cell_size
            end_cell_max = end_cell_min + cell_size
            end_portal = self.np_random.integers(end_cell_min, end_cell_max, size=2)

            # Ensure portals don't overlap with robot or goal
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts and (
                np.array_equal(start_portal, self.robot_pos) or
                np.array_equal(start_portal, self.goal_pos)
            ):
                start_portal = self.np_random.integers(start_cell_min, start_cell_max, size=2)
                attempts += 1

            attempts = 0
            while attempts < max_attempts and (
                np.array_equal(end_portal, self.robot_pos) or
                np.array_equal(end_portal, self.goal_pos) or
                np.array_equal(end_portal, start_portal)
            ):
                end_portal = self.np_random.integers(end_cell_min, end_cell_max, size=2)
                attempts += 1

            self.portals.append((start_portal, end_portal))

        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def reset_from_state(
        self, state: GraphInstance, seed: int | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to match a specific low-level state.

        Args:
            state: GraphInstance observation containing positions of robot, goal, and portals

        Returns:
            Observation and info dict
        """
        # Extract positions from GraphInstance nodes
        # Node format: [type, x, y, cell_x, cell_y, id/pair_id]

        robot_node = state.nodes[0]  # type=0
        self.robot_pos = np.array([int(robot_node[1]), int(robot_node[2])], dtype=np.int32)

        goal_node = state.nodes[1]  # type=1
        self.goal_pos = np.array([int(goal_node[1]), int(goal_node[2])], dtype=np.int32)

        # Extract portal positions
        self.portals = []
        portal_nodes = state.nodes[2:]  # All remaining nodes are portals

        # Portals come in pairs, so process them two at a time
        for i in range(0, len(portal_nodes), 2):
            if i + 1 < len(portal_nodes):
                portal1_node = portal_nodes[i]
                portal2_node = portal_nodes[i + 1]

                portal1 = np.array([int(portal1_node[1]), int(portal1_node[2])], dtype=np.int32)
                portal2 = np.array([int(portal2_node[1]), int(portal2_node[2])], dtype=np.int32)

                self.portals.append((portal1, portal2))

        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[GraphInstance, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: 0=up, 1=down, 2=left, 3=right, 4=teleport
        """
        assert self.robot_pos is not None

        # Execute action
        new_pos = self.robot_pos.copy()

        if action == 0:  # Up
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 1:  # Down
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 2:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 4:  # Teleport
            # Check if on any portal
            for portal1, portal2 in self.portals:
                if np.array_equal(self.robot_pos, portal1):
                    new_pos = portal2.copy()
                    break
                elif np.array_equal(self.robot_pos, portal2):
                    new_pos = portal1.copy()
                    break
            # If not on a portal, teleport has no effect (wastes a step)

        self.robot_pos = new_pos
        self.step_count += 1

        # Check if goal reached
        terminated = np.array_equal(self.robot_pos, self.goal_pos)

        # Check if max steps exceeded
        truncated = self.step_count >= self.max_episode_steps

        # Reward: -1 per step, +100 for reaching goal
        reward = 100.0 if terminated else -1.0

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> GraphInstance:
        """Get current observation as a graph."""
        assert self.robot_pos is not None
        assert self.goal_pos is not None

        nodes = []

        # Robot node: [type=0, x, y, cell_x, cell_y, id=0]
        robot_cell = self.robot_pos // self.num_states_per_cell
        robot_node = np.array([
            0,  # type
            float(self.robot_pos[0]),
            float(self.robot_pos[1]),
            float(robot_cell[0]),
            float(robot_cell[1]),
            0,  # id
        ], dtype=np.float32)
        nodes.append(robot_node)

        # Goal node: [type=1, x, y, cell_x, cell_y, id=1]
        goal_cell = self.goal_pos // self.num_states_per_cell
        goal_node = np.array([
            1,  # type
            float(self.goal_pos[0]),
            float(self.goal_pos[1]),
            float(goal_cell[0]),
            float(goal_cell[1]),
            1,  # id
        ], dtype=np.float32)
        nodes.append(goal_node)

        # Portal nodes: [type=2, x, y, cell_x, cell_y, portal_pair_id]
        for pair_id, (portal1, portal2) in enumerate(self.portals):
            portal1_cell = portal1 // self.num_states_per_cell
            portal1_node = np.array([
                2,  # type
                float(portal1[0]),
                float(portal1[1]),
                float(portal1_cell[0]),
                float(portal1_cell[1]),
                float(pair_id * 2),  # id
            ], dtype=np.float32)
            nodes.append(portal1_node)

            portal2_cell = portal2 // self.num_states_per_cell
            portal2_node = np.array([
                2,  # type
                float(portal2[0]),
                float(portal2[1]),
                float(portal2_cell[0]),
                float(portal2_cell[1]),
                float(pair_id * 2 + 1),  # id
            ], dtype=np.float32)
            nodes.append(portal2_node)

        return GraphInstance(nodes=np.stack(nodes), edges=None, edge_links=None)

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        robot_cell = self._get_cell(self.robot_pos)
        return {
            "robot_cell": robot_cell,
            "goal_cell": self._get_cell(self.goal_pos),
            "step_count": self.step_count,
        }

    def extract_relevant_object_features(
        self, obs: GraphInstance, relevant_object_names: set[str]
    ) -> NDArray[np.float32]:
        """Extract features from relevant objects in the observation.

        In gridworld, the only object is 'robot0'. This function extracts
        the robot's position features (x, y, cell_x, cell_y), the goal's
        position features, and all portal features for the policy to use.

        Args:
            obs: Graph observation with nodes [type, x, y, cell_x, cell_y, id]
            relevant_object_names: Set of object names (should contain "robot0")

        Returns:
            Feature vector containing robot, goal, and portal features
        """
        if not hasattr(obs, "nodes"):
            return obs  # Not a graph observation

        nodes = obs.nodes
        robot_features = None
        goal_features = None
        portal_features = []

        for node in nodes:
            node_type = int(node[0])
            if node_type == 0:  # Robot
                # Extract: x, y, cell_x, cell_y (skip type and id)
                robot_features = node[1:5]
            elif node_type == 1:  # Goal
                # Extract: x, y, cell_x, cell_y (skip type and id)
                goal_features = node[1:5]
            elif node_type == 2:  # Portal
                # Extract: x, y, cell_x, cell_y (skip type and id)
                portal_features.append(node[1:5])

        # Build feature vector: robot + goal + all portals
        features = []
        if robot_features is not None:
            features.extend(robot_features)
        if goal_features is not None:
            features.extend(goal_features)
        # Add portals in order (sorted by id to ensure consistency)
        for portal_feat in portal_features:
            features.extend(portal_feat)

        return np.array(features, dtype=np.float32)

    def _get_cell(self, pos: NDArray[np.int32] | None) -> tuple[int, int]:
        """Get cell coordinates for a position."""
        if pos is None:
            return (-1, -1)
        cell = pos // self.num_states_per_cell
        return (int(cell[0]), int(cell[1]))

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode != "human":
            return

        print("\n" + "=" * (self.grid_size * 2 + 3))
        for y in range(self.grid_size - 1, -1, -1):
            row = "|"
            for x in range(self.grid_size):
                pos = np.array([x, y])
                if np.array_equal(pos, self.robot_pos):
                    row += " R"
                elif np.array_equal(pos, self.goal_pos):
                    row += " G"
                else:
                    # Check if it's a portal
                    is_portal = False
                    for portal1, portal2 in self.portals:
                        if np.array_equal(pos, portal1) or np.array_equal(pos, portal2):
                            row += " P"
                            is_portal = True
                            break
                    if not is_portal:
                        # Show cell boundaries
                        if x % self.num_states_per_cell == 0 or y % self.num_states_per_cell == 0:
                            row += " +"
                        else:
                            row += " ."
            row += " |"
            print(row)
        print("=" * (self.grid_size * 2 + 3))
        cell = self._get_cell(self.robot_pos)
        print(f"Step: {self.step_count}, Cell: ({cell[0]}, {cell[1]})")


# ============================================================================
# PDDL Types and Predicates
# ============================================================================


class GridworldTypes:
    """Types for gridworld."""

    robot = Type("robot")


class GridworldPredicates(PredicateContainer):
    """Predicates for gridworld."""

    def __init__(self, num_cells: int):
        """Initialize predicates.

        Creates:
        - InRow0, InRow1, ..., InRow(C-1)
        - InCol0, InCol1, ..., InCol(C-1)
        - GoalReached
        """
        self.num_cells = num_cells

        # Row predicates
        self.row_preds = []
        for i in range(num_cells):
            pred = Predicate(f"InRow{i}", [GridworldTypes.robot])
            setattr(self, f"InRow{i}", pred)
            self.row_preds.append(pred)

        # Column predicates
        self.col_preds = []
        for j in range(num_cells):
            pred = Predicate(f"InCol{j}", [GridworldTypes.robot])
            setattr(self, f"InCol{j}", pred)
            self.col_preds.append(pred)

        # Goal predicate
        self.GoalReached = Predicate("GoalReached", [GridworldTypes.robot])

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return getattr(self, key)

    def as_set(self) -> set[Predicate]:
        """Convert to set."""
        return set(self.row_preds + self.col_preds + [self.GoalReached])


# ============================================================================
# Perceiver
# ============================================================================


class GridworldPerceiver(Perceiver[GraphInstance]):
    """Perceiver for gridworld that maps graph observations to high-level atoms."""

    def __init__(self, num_cells: int, num_states_per_cell: int):
        """Initialize perceiver.

        Args:
            num_cells: Number of cells in each dimension
            num_states_per_cell: Number of states per cell
        """
        self.num_cells = num_cells
        self.num_states_per_cell = num_states_per_cell
        self.predicates = GridworldPredicates(num_cells)
        self.robot_obj = Object("robot0", GridworldTypes.robot)

    def reset(
        self, obs: GraphInstance, info: dict[str, Any]
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver and get initial objects, atoms, and goal."""
        objects = {self.robot_obj}

        # Get current atoms based on robot cell
        atoms = self._get_atoms_from_obs(obs)

        # Goal: robot in cell (C-1, C-1) and goal reached
        goal_atoms = {
            GroundAtom(self.predicates.row_preds[self.num_cells - 1], [self.robot_obj]),
            GroundAtom(self.predicates.col_preds[self.num_cells - 1], [self.robot_obj]),
            GroundAtom(self.predicates.GoalReached, [self.robot_obj]),
        }

        return objects, atoms, goal_atoms

    def step(self, obs: GraphInstance) -> set[GroundAtom]:
        """Get atoms from observation."""
        return self._get_atoms_from_obs(obs)

    def _get_atoms_from_obs(self, obs: GraphInstance) -> set[GroundAtom]:
        """Convert graph observation to ground atoms."""
        # Robot is always first node
        robot_node = obs.nodes[0]
        robot_x, robot_y = robot_node[1], robot_node[2]
        goal_node = obs.nodes[1]
        goal_x, goal_y = goal_node[1], goal_node[2]

        atoms = set()

        # Determine cell
        cell_row = int(robot_y // self.num_states_per_cell)
        cell_col = int(robot_x // self.num_states_per_cell)
        # print("Coords:", robot_x, robot_y, cell_row, cell_col)

        # Add row and column predicates
        atoms.add(GroundAtom(self.predicates.row_preds[cell_row], [self.robot_obj]))
        atoms.add(GroundAtom(self.predicates.col_preds[cell_col], [self.robot_obj]))

        # Check if goal reached
        if np.isclose(robot_x, goal_x) and np.isclose(robot_y, goal_y):
            atoms.add(GroundAtom(self.predicates.GoalReached, [self.robot_obj]))

        return atoms


# ============================================================================
# Skills (that ignore portals)
# ============================================================================


class BaseGridworldSkill(LiftedOperatorSkill[GraphInstance, int]):
    """Base class for gridworld skills."""

    def __init__(self, components: PlanningComponents[GraphInstance]):
        """Initialize skill."""
        super().__init__()
        self._components = components
        self._lifted_operator = self._get_lifted_operator()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        op_name = self._get_operator_name()
        return next(
            op
            for op in self._components.operators
            if op.name == op_name
        )

    def _get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError


class MoveUpSkill(BaseGridworldSkill):
    """Move up one cell (ignores portals)."""

    def __init__(self, components: PlanningComponents[GraphInstance], from_row: int):
        """Initialize skill.

        Args:
            components: Planning components containing operators
            from_row: The row this skill moves from (0 to num_cells-2)
        """
        self.from_row = from_row
        super().__init__(components)
        # Extract env info from components
        env = components.perceiver  # type: ignore
        self.num_states_per_cell = env.num_states_per_cell

    def _get_operator_name(self) -> str:
        return f"MoveUp_from_row{self.from_row}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> int:
        """Move up by navigating to the cell above."""
        robot_node = obs.nodes[0]
        robot_y = robot_node[2]

        # Target is the cell above
        target_y = ((int(robot_y) // self.num_states_per_cell) + 1) * self.num_states_per_cell

        # Move up towards target
        if robot_y < target_y - 0.5:
            return 0  # Up
        return 0  # Default


class MoveRightSkill(BaseGridworldSkill):
    """Move right one cell (ignores portals)."""

    def __init__(self, components: PlanningComponents[GraphInstance], from_col: int):
        """Initialize skill.

        Args:
            components: Planning components containing operators
            from_col: The column this skill moves from (0 to num_cells-2)
        """
        self.from_col = from_col
        super().__init__(components)
        # Extract env info from components
        env = components.perceiver  # type: ignore
        self.num_states_per_cell = env.num_states_per_cell

    def _get_operator_name(self) -> str:
        return f"MoveRight_from_col{self.from_col}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> int:
        """Move right by navigating to the cell to the right."""
        robot_node = obs.nodes[0]
        robot_x = robot_node[1]

        # Target is the cell to the right
        target_x = ((int(robot_x) // self.num_states_per_cell) + 1) * self.num_states_per_cell

        # Move right towards target
        if robot_x < target_x - 0.5:
            return 3  # Right
        return 3  # Default


class NavigateToGoalSkill(BaseGridworldSkill):
    """Navigate to goal within the final cell."""

    def _get_operator_name(self) -> str:
        return "NavigateToGoal"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> int:
        """Navigate to goal using Manhattan distance."""
        robot_node = obs.nodes[0]
        robot_x, robot_y = robot_node[1], robot_node[2]
        goal_node = obs.nodes[1]
        goal_x, goal_y = goal_node[1], goal_node[2]

        # Move towards goal (prioritize y, then x)
        if robot_y < goal_y - 0.5:
            return 0  # Up
        elif robot_y > goal_y + 0.5:
            return 1  # Down
        elif robot_x < goal_x - 0.5:
            return 3  # Right
        elif robot_x > goal_x + 0.5:
            return 2  # Left
        else:
            return 0  # Already at goal


# ============================================================================
# TAMP System
# ============================================================================


class GridworldTAMPSystem(ImprovisationalTAMPSystem[GraphInstance, int]):
    """Gridworld TAMP system for testing SLAP."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        num_cells: int = 2,
        num_states_per_cell: int = 5,
        num_teleporters: int = 1,
        seed: int | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
    ):
        """Initialize gridworld system."""
        self.num_cells = num_cells
        self.num_states_per_cell = num_states_per_cell
        self.num_teleporters = num_teleporters
        self.max_episode_steps = max_episode_steps
        self._env_seed = seed
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_env(self) -> gym.Env:
        """Create base gridworld environment."""
        return GridworldEnv(
            num_cells=self.num_cells,
            num_states_per_cell=self.num_states_per_cell,
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
        return "gridworld"

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
        num_cells: int = 2,
        num_states_per_cell: int = 5,
        num_teleporters: int = 1,
        seed: int = 42,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
    ) -> GridworldTAMPSystem:
        """Create default gridworld system."""
        predicates = GridworldPredicates(num_cells)
        perceiver = GridworldPerceiver(num_cells, num_states_per_cell)

        # Create operators first
        robot = Variable("?r", GridworldTypes.robot)
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

        # NavigateToGoal operator
        navigate_op = LiftedOperator(
            name="NavigateToGoal",
            parameters=[robot],
            preconditions={
                predicates.row_preds[num_cells - 1]([robot]),
                predicates.col_preds[num_cells - 1]([robot]),
            },
            add_effects={predicates.GoalReached([robot])},
            delete_effects=set(),
        )
        operators.add(navigate_op)

        # Create planning components (needed for skill initialization)
        components = PlanningComponents(
            types={GridworldTypes.robot},
            predicate_container=predicates,
            skills=set(),  # Will be populated below
            perceiver=perceiver,
            operators=operators,
        )

        # Create skills (they need components to find their operators)
        skills = set()
        for row in range(num_cells - 1):
            skills.add(MoveUpSkill(components, from_row=row))
        for col in range(num_cells - 1):
            skills.add(MoveRightSkill(components, from_col=col))
        skills.add(NavigateToGoalSkill(components))

        # Update components with skills
        components.skills = skills

        return cls(
            planning_components=components,
            num_cells=num_cells,
            num_states_per_cell=num_states_per_cell,
            num_teleporters=num_teleporters,
            seed=seed,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
