"""Simplified gridworld environment for distance heuristic learning.

This environment provides a simplified gridworld where:
- High-level: C×C grid of cells (nodes are just cells, no goal predicate)
- Low-level: S×S states within each cell
- Observation: GraphInstance with single node containing just robot position
- Actions: up, down, left, right, teleport
- Skills: MoveUp (one cell up), MoveRight (one cell right)
- Portals: Fixed at cell centers, randomly placed at initialization

Key differences from gridworld.py:
- No GoalReached predicate (nodes are just spatial cells)
- Random start state and random goal node each reset
- Fixed portal locations (placed at initialization, not randomized on reset)
- Minimal observation space (just 2D robot position)
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
# Gridworld Fixed Gymnasium Environment
# ============================================================================


class GridworldFixedEnv(gym.Env):
    """A simplified hierarchical gridworld with fixed portals.

    State space:
    - High-level: C×C cells
    - Low-level: S×S positions within each cell
    - Total positions: (C*S) × (C*S)

    Key simplifications:
    - Nodes are just cells (no goal predicate)
    - Portals are fixed at cell centers (set at initialization)
    - Observation is just 2D robot position
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

        # Observation: GraphInstance with single node [x, y]
        # Minimal observation - just robot position
        self.observation_space = Graph(
            node_space=Box(low=0, high=self.grid_size, shape=(2,), dtype=np.float32),
            edge_space=None,
        )

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=teleport
        self.action_space = spaces.Discrete(5)

        # Initialize FIXED portal locations (locked for entire environment lifetime)
        # Portals are placed at the CENTER of randomly selected cell pairs
        self.portal_positions: list[tuple[NDArray[np.int32], NDArray[np.int32]]] = []

        # Use temporary RNG for initialization
        init_rng = np.random.default_rng(seed)
        cell_size = num_states_per_cell

        # Get all cells
        all_cells = [(i, j) for i in range(num_cells) for j in range(num_cells)]

        # Randomly select cell pairs for portals (without replacement)
        # obeying constraint: no two portal cells can be adjacent (Manhattan dist <= 1)
        available_cells = all_cells.copy()
        init_rng.shuffle(available_cells)
        
        occupied_cells = set()
        portals_found = 0

        while portals_found < num_teleporters and len(available_cells) >= 2:
            # Find first valid cell1
            c1_idx = -1
            cell1 = None
            for idx, c in enumerate(available_cells):
                # Check if c is adjacent to any occupied cell
                if any(abs(c[0] - oc[0]) + abs(c[1] - oc[1]) <= 1 for oc in occupied_cells):
                    continue
                c1_idx = idx
                cell1 = c
                break
            
            if cell1 is None:
                # No valid cell1 found remaining in available_cells
                break
                
            # Find first valid cell2
            c2_idx = -1
            cell2 = None
            for idx, c in enumerate(available_cells):
                if idx == c1_idx:
                    continue
                    
                # Check if c is adjacent to occupied cells
                if any(abs(c[0] - oc[0]) + abs(c[1] - oc[1]) <= 1 for oc in occupied_cells):
                    continue
                    
                # Check if c is adjacent to cell1
                if abs(c[0] - cell1[0]) + abs(c[1] - cell1[1]) <= 1:
                    continue
                    
                c2_idx = idx
                cell2 = c
                break
            
            if cell2 is not None:
                # Found a pair
                occupied_cells.add(cell1)
                occupied_cells.add(cell2)
                
                # Place portals at center of each cell
                portal1_pos = np.array([
                    cell1[0] * cell_size + cell_size // 2,
                    cell1[1] * cell_size + cell_size // 2
                ], dtype=np.int32)

                portal2_pos = np.array([
                    cell2[0] * cell_size + cell_size // 2,
                    cell2[1] * cell_size + cell_size // 2
                ], dtype=np.int32)

                self.portal_positions.append((portal1_pos, portal2_pos))
                
                # Remove used cells from available_cells
                # Remove larger index first to avoid index shifting issues
                if c1_idx > c2_idx:
                    available_cells.pop(c1_idx)
                    available_cells.pop(c2_idx)
                else:
                    available_cells.pop(c2_idx)
                    available_cells.pop(c1_idx)
                
                portals_found += 1
            else:
                # Found cell1 but no cell2. cell1 cannot be a start point with current remaining cells.
                available_cells.pop(c1_idx)

        # State variables (randomized each reset)
        self.robot_pos: NDArray[np.int32] | None = None
        self.goal_cell: tuple[int, int] | None = None  # Goal is just a cell, not a position
        self.step_count: int = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment.

        Robot starts at a random position in the entire grid.
        Goal is a random cell in the C×C grid.
        Portals remain fixed at their initialized locations.
        """
        print("reset")
        super().reset(seed=seed)

        # Robot starts randomly anywhere in the grid
        self.robot_pos = self.np_random.integers(0, self.grid_size, size=2, dtype=np.int32)

        # Goal is a random cell (not a specific position, just a cell)
        self.goal_cell = tuple(self.np_random.integers(0, self.num_cells, size=2))

        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
        # return obs

    def reset_from_state(
        self, state: GraphInstance, seed: int | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to match a specific low-level state.

        Args:
            state: GraphInstance observation containing robot position [x, y]

        Returns:
            Observation and info dict
        """
        # print("env reset from state")
        # Extract robot position from GraphInstance
        # Node format: [x, y]
        robot_node = state.nodes[0]
        self.robot_pos = np.array([int(robot_node[0]), int(robot_node[1])], dtype=np.int32)

        # Goal cell remains the same (set by previous reset)
        # If not set, default to a cell
        if self.goal_cell is None:
            self.goal_cell = (self.num_cells - 1, self.num_cells - 1)

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
        # print("env step")
        assert self.robot_pos is not None
        assert self.goal_cell is not None

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
            # Check if on any portal (fixed positions)
            for portal1, portal2 in self.portal_positions:
                if np.array_equal(self.robot_pos, portal1):
                    new_pos = portal2.copy()
                    break
                elif np.array_equal(self.robot_pos, portal2):
                    new_pos = portal1.copy()
                    break
            # If not on a portal, teleport has no effect (wastes a step)

        # print(f"env step: new_pos={new_pos}, about to assign to robot_pos={self.robot_pos}")
        self.robot_pos = new_pos
        # print(f"env step: robot_pos AFTER={self.robot_pos}")
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

        # Single node: [x, y]
        robot_node = np.array([
            float(self.robot_pos[0]),
            float(self.robot_pos[1]),
        ], dtype=np.float32)

        return GraphInstance(nodes=robot_node.reshape(1, -1), edges=None, edge_links=None)

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
        """Extract features from observation.

        In gridworld_fixed, the observation is already minimal (just x, y).
        We return it as-is for the policy to use.

        Args:
            obs: Graph observation with single node [x, y]
            relevant_object_names: Set of object names (should contain "robot0")

        Returns:
            Feature vector containing [x, y]
        """
        if not hasattr(obs, "nodes"):
            return obs  # Not a graph observation

        # Return flattened position
        return obs.nodes.flatten()

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
                cell = self._get_cell(pos)

                if np.array_equal(pos, self.robot_pos):
                    row += " R"
                elif cell == self.goal_cell:
                    # Show goal cell with 'g' (lowercase to distinguish from exact goal position)
                    row += " g"
                else:
                    # Check if it's a fixed portal
                    is_portal = False
                    for portal1, portal2 in self.portal_positions:
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
        robot_cell = self._get_cell(self.robot_pos)
        print(f"Step: {self.step_count}, Robot Cell: {robot_cell}, Goal Cell: {self.goal_cell}")


# ============================================================================
# PDDL Types and Predicates
# ============================================================================


class GridworldFixedTypes:
    """Types for gridworld fixed."""

    robot = Type("robot")


class GridworldFixedPredicates(PredicateContainer):
    """Predicates for gridworld fixed - only spatial predicates, no goal."""

    def __init__(self, num_cells: int):
        """Initialize predicates.

        Creates:
        - InRow0, InRow1, ..., InRow(C-1)
        - InCol0, InCol1, ..., InCol(C-1)

        Note: No GoalReached predicate - nodes are just cells
        """
        self.num_cells = num_cells

        # Row predicates
        self.row_preds = []
        for i in range(num_cells):
            pred = Predicate(f"InRow{i}", [GridworldFixedTypes.robot])
            setattr(self, f"InRow{i}", pred)
            self.row_preds.append(pred)

        # Column predicates
        self.col_preds = []
        for j in range(num_cells):
            pred = Predicate(f"InCol{j}", [GridworldFixedTypes.robot])
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


class GridworldFixedPerceiver(Perceiver[GraphInstance]):
    """Perceiver for gridworld fixed that maps observations to cell-based atoms."""

    def __init__(self, num_cells: int, num_states_per_cell: int):
        """Initialize perceiver.

        Args:
            num_cells: Number of cells in each dimension
            num_states_per_cell: Number of states per cell
        """
        self.num_cells = num_cells
        self.num_states_per_cell = num_states_per_cell
        self.predicates = GridworldFixedPredicates(num_cells)
        self.robot_obj = Object("robot0", GridworldFixedTypes.robot)

    def reset(
        self, obs: GraphInstance, info: dict[str, Any]
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver and get initial objects, atoms, and goal.

        Goal is determined by the environment's goal_cell stored in info.
        """
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
        # Robot node: [x, y]
        robot_node = obs.nodes[0]
        robot_x, robot_y = robot_node[0], robot_node[1]

        atoms = set()

        # Determine cell
        cell_col = int(robot_x // self.num_states_per_cell)
        cell_row = int(robot_y // self.num_states_per_cell)

        # Add row and column predicates
        atoms.add(GroundAtom(self.predicates.row_preds[cell_row], [self.robot_obj]))
        atoms.add(GroundAtom(self.predicates.col_preds[cell_col], [self.robot_obj]))

        return atoms


# ============================================================================
# Skills
# ============================================================================


class BaseGridworldFixedSkill(LiftedOperatorSkill[GraphInstance, int]):
    """Base class for gridworld fixed skills."""

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


class MoveUpFixedSkill(BaseGridworldFixedSkill):
    """Move up one cell."""

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
        robot_y = robot_node[1]

        # Target is the cell above
        target_y = ((int(robot_y) // self.num_states_per_cell) + 1) * self.num_states_per_cell

        # Move up towards target
        if robot_y < target_y - 0.5:
            return 0  # Up
        return 0  # Default


class MoveRightFixedSkill(BaseGridworldFixedSkill):
    """Move right one cell."""

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
        robot_x = robot_node[0]

        # Target is the cell to the right
        target_x = ((int(robot_x) // self.num_states_per_cell) + 1) * self.num_states_per_cell

        # Move right towards target
        if robot_x < target_x - 0.5:
            return 3  # Right
        return 3  # Default


class MoveDownFixedSkill(BaseGridworldFixedSkill):
    """Move down one cell."""

    def __init__(self, components: PlanningComponents[GraphInstance], from_row: int):
        """Initialize skill.

        Args:
            components: Planning components containing operators
            from_row: The row this skill moves from (1 to num_cells-1)
        """
        self.from_row = from_row
        super().__init__(components)
        # Extract env info from components
        env = components.perceiver  # type: ignore
        self.num_states_per_cell = env.num_states_per_cell

    def _get_operator_name(self) -> str:
        return f"MoveDown_from_row{self.from_row}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> int:
        """Move down by navigating to the cell below."""
        robot_node = obs.nodes[0]
        robot_y = robot_node[1]

        # Target is the cell below
        target_y = ((int(robot_y) // self.num_states_per_cell) - 1) * self.num_states_per_cell + (self.num_states_per_cell - 1)

        # Move down towards target
        if robot_y > target_y + 0.5:
            return 1  # Down
        return 1  # Default


class MoveLeftFixedSkill(BaseGridworldFixedSkill):
    """Move left one cell."""

    def __init__(self, components: PlanningComponents[GraphInstance], from_col: int):
        """Initialize skill.

        Args:
            components: Planning components containing operators
            from_col: The column this skill moves from (1 to num_cells-1)
        """
        self.from_col = from_col
        super().__init__(components)
        # Extract env info from components
        env = components.perceiver  # type: ignore
        self.num_states_per_cell = env.num_states_per_cell

    def _get_operator_name(self) -> str:
        return f"MoveLeft_from_col{self.from_col}"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> int:
        """Move left by navigating to the cell to the left."""
        robot_node = obs.nodes[0]
        robot_x = robot_node[0]

        # Target is the cell to the left
        target_x = ((int(robot_x) // self.num_states_per_cell) - 1) * self.num_states_per_cell + (self.num_states_per_cell - 1)

        # Move left towards target
        if robot_x > target_x + 0.5:
            return 2  # Left
        return 2  # Default


# ============================================================================
# TAMP System
# ============================================================================


class GridworldFixedTAMPSystem(ImprovisationalTAMPSystem[GraphInstance, int]):
    """Simplified gridworld TAMP system for distance heuristic learning."""

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
        """Initialize gridworld fixed system."""
        self.num_cells = num_cells
        self.num_states_per_cell = num_states_per_cell
        self.num_teleporters = num_teleporters
        self.max_episode_steps = max_episode_steps
        self._env_seed = seed
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_env(self) -> gym.Env:
        """Create base gridworld fixed environment."""
        return GridworldFixedEnv(
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
        return "gridworld_fixed"

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
    ) -> GridworldFixedTAMPSystem:
        """Create default gridworld fixed system."""
        predicates = GridworldFixedPredicates(num_cells)
        perceiver = GridworldFixedPerceiver(num_cells, num_states_per_cell)

        # Create operators first
        robot = Variable("?r", GridworldFixedTypes.robot)
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

        # Note: No NavigateToGoal operator - nodes are just cells

        # Create planning components (needed for skill initialization)
        components = PlanningComponents(
            types={GridworldFixedTypes.robot},
            predicate_container=predicates,
            skills=set(),  # Will be populated below
            perceiver=perceiver,
            operators=operators,
        )

        # Create skills (they need components to find their operators)
        skills = set()
        for row in range(num_cells - 1):
            skills.add(MoveUpFixedSkill(components, from_row=row))
        for row in range(1, num_cells):
            skills.add(MoveDownFixedSkill(components, from_row=row))
        for col in range(num_cells - 1):
            skills.add(MoveRightFixedSkill(components, from_col=col))
        for col in range(1, num_cells):
            skills.add(MoveLeftFixedSkill(components, from_col=col))

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
