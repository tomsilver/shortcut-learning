"""Analysis utilities for improvisational TAMP approaches."""

from typing import TypeVar, TYPE_CHECKING

import numpy as np
from relational_structs import GroundAtom, GroundOperator

from tamp_improv.approaches.improvisational.graph import PlanningGraph, PlanningGraphEdge
from tamp_improv.approaches.improvisational.policies.base import Policy, PolicyContext
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem, GraphInstance
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedTAMPSystem
ObsType = TypeVar("ObsType")


def execute_shortcut_once(
    policy: Policy,
    system: ImprovisationalTAMPSystem,
    start_state: ObsType,
    goal_atoms: set[GroundAtom],
    max_steps: int = 100,
    source_node_id: int | None = None,
    target_node_id: int | None = None,
) -> tuple[bool, int, list[ObsType]]:
    """Execute a shortcut policy once from start state to goal atoms.

    Args:
        policy: The policy to execute
        system: The TAMP system (provides env and perceiver)
        start_state: The initial state to start from
        goal_atoms: The goal atoms to reach
        max_steps: Maximum number of steps to execute
        source_node_id: Optional source node ID for policy key matching
        target_node_id: Optional target node ID for policy key matching

    Returns:
        Tuple of (success, num_steps, path)
            - success: Whether the goal atoms were reached
            - num_steps: Number of steps taken
            - path: List of observations (low-level states) during the shortcut execution
    """
    # Get the base environment
    env = system.env

    # Reset environment to start state
    obs, _ = env.reset_from_state(start_state)

    # Store the path
    path = [obs]

    # Get start atoms from perceiver
    start_atoms = system.perceiver.step(obs)

    # Configure policy with context
    info = {}
    if source_node_id is not None and target_node_id is not None:
        info["source_node_id"] = source_node_id
        info["target_node_id"] = target_node_id

    policy_context = PolicyContext(
        current_atoms=start_atoms,
        goal_atoms=goal_atoms,
        info=info,
    )
    policy.configure_context(policy_context)

    # Execute policy
    num_steps = 0
    success = False

    for _ in range(max_steps):
        # Get action from policy
        if not policy.can_initiate():
            break

        action = policy.get_action(obs)
        if action is None:
            break

        # Step environment
        obs, _, _, _, _ = env.step(action)
        num_steps += 1
        path.append(obs)

        # Check if goal atoms are reached
        current_atoms = system.perceiver.step(obs)
        if goal_atoms == frozenset(current_atoms):
            success = True
            break

    return success, num_steps, path


def execute_edge_once(
    system: ImprovisationalTAMPSystem,
    edge: PlanningGraphEdge,
    start_state: ObsType,
    max_steps: int = 100,
) -> tuple[bool, int]:
    """Execute an edge (operator) once from a start state.

    Args:
        system: The TAMP system (provides env and perceiver)
        edge: The edge to execute (must have an operator)
        start_state: The initial state to start from
        max_steps: Maximum number of steps to execute

    Returns:
        Tuple of (success, num_steps)
            - success: Whether the target atoms were reached
            - num_steps: Number of steps taken (max_steps if failed)
    """
    # print("Executing edge", edge, "from", start_state)

    if edge.operator is None or edge.is_shortcut:
        # Can't execute edge without operator, or if it's a shortcut
        return False, max_steps

    env = system.env

    # Reset environment to start state
    obs, info = env.reset_from_state(start_state)

    # Get target atoms from edge
    target_atoms = edge.target.atoms

    # Get the skill that can execute this operator
    skills = [s for s in system.skills if s.can_execute(edge.operator)]
    if not skills:
        return False, max_steps
    skill = skills[0]
    skill.reset(edge.operator)

    # Execute the operator using the skill
    num_steps = 0
    success = False

    for _ in range(max_steps):
        # Get action from skill
        action = skill.get_action(obs)
        # print(action)
        if action is None:
            break

        # Step environment
        obs, _, _, _, _ = env.step(action)

        num_steps += 1

        # Check if we reached target atoms
        current_atoms = system.perceiver.step(obs)
        if frozenset(current_atoms) == target_atoms:
            success = True
            break


    return success, num_steps


def compute_average_edge_cost(
    system: ImprovisationalTAMPSystem,
    edge: PlanningGraphEdge,
    start_states: list[ObsType],
    num_samples: int = 5,
    max_steps: int = 100,
) -> float:
    """Compute average cost of an edge by executing it from multiple start states.

    Args:
        system: The TAMP system
        edge: The edge to compute cost for
        start_states: List of start states to sample from
        num_samples: Number of samples to use (or all if fewer available)
        max_steps: Maximum steps per execution attempt

    Returns:
        Average cost (num_steps) across successful executions, or inf if all failed
    """
    if not start_states:
        return float("inf")

    # Sample states (or use all if fewer than num_samples)
    if len(start_states) <= num_samples:
        sampled_states = start_states
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(start_states), size=num_samples, replace=False)
        sampled_states = [start_states[i] for i in indices]

    # Execute edge from each sampled state
    costs = []
    for start_state in sampled_states:
        success, num_steps = execute_edge_once(system, edge, start_state, max_steps)
        if success:
            costs.append(num_steps)
        else:
            costs.append(max_steps)  # Penalty for failure

    # Return average cost
    # print("Costs of edge", edge, "are", costs)
    if costs:
        return float(np.max(costs))
    return float("inf")


def compute_all_edge_costs(
    system: ImprovisationalTAMPSystem,
    graph: PlanningGraph,
    node_states: dict[frozenset, list],
    num_samples: int = 5,
    max_steps: int = 100,
) -> None:
    """Compute costs for all edges in a planning graph.

    This updates the edge costs in-place by executing each edge multiple times
    from different start states sampled from node_states.

    Args:
        system: The TAMP system
        graph: Planning graph with edges to compute costs for
        node_states: Dict mapping frozenset[atoms] -> list of states
        num_samples: Number of start states to sample per edge
        max_steps: Maximum steps allowed for edge execution
    """
    print(f"\nComputing edge costs using {num_samples} samples per edge...")

    for edge_idx, edge in enumerate(graph.edges):
        # Skip shortcut edges (they don't have operators)
        if edge.is_shortcut or edge.operator is None:
            edge.cost = 1.0  # Default cost for shortcuts
            continue

        # Get states for the source node
        source_atoms = edge.source.atoms
        if source_atoms not in node_states or not node_states[source_atoms]:
            edge.cost = float("inf")  # No states available
            continue

        # Compute average cost
        edge.cost = compute_average_edge_cost(
            system, edge, node_states[source_atoms], num_samples, max_steps
        )


        # Print progress every 10 edges
        if (edge_idx + 1) % 10 == 0:
            print(f"  Processed {edge_idx + 1}/{len(graph.edges)} edges")

    print(f"Edge cost computation complete!")

def compute_true_distance_gridworld_fixed(
    system: "GridworldFixedTAMPSystem",
    start_state: "GraphInstance",
    goal_node_atoms: set[GroundAtom],
) -> float:
    """Compute true minimum distance from start state to goal node using Dijkstra's algorithm.

    This function considers teleporters when computing the shortest path at the state level.

    Args:
        system: GridworldFixedTAMPSystem instance
        start_state: Low-level state (GraphInstance observation)
        goal_node_atoms: Set of atoms defining the goal node

    Returns:
        Minimum distance from start_state to any state satisfying goal_node_atoms
    """
    # Extract robot position from start state
    robot_node = start_state.nodes[0]  # type=0 is robot
    robot_x, robot_y = robot_node[0], robot_node[1]

    # Parse goal atoms to determine target cell
    target_row = None
    target_col = None

    for atom in goal_node_atoms:
        atom_str = str(atom)
        if "InRow" in atom_str or "Row" in atom_str:
            for i in range(system.env.num_cells):
                if f"Row{i}" in atom_str or f"InRow{i}" in atom_str:
                    target_row = i
                    break
        elif "InCol" in atom_str or "Col" in atom_str:
            for i in range(system.env.num_cells):
                if f"Col{i}" in atom_str or f"InCol{i}" in atom_str:
                    target_col = i
                    break

    if target_row is None or target_col is None:
        return float("inf")

    # Compute target cell boundaries
    num_states_per_cell = system.env.num_states_per_cell
    target_x_min = target_col * num_states_per_cell
    target_x_max = (target_col + 1) * num_states_per_cell
    target_y_min = target_row * num_states_per_cell
    target_y_max = (target_row + 1) * num_states_per_cell

    # Find closest point in target cell to robot position
    closest_x = np.clip(robot_x, target_x_min, target_x_max)
    closest_y = np.clip(robot_y, target_y_min, target_y_max)

    # Use Dijkstra considering teleporters at state level
    return _dijkstra_state_level(
        system, robot_x, robot_y, closest_x, closest_y
    )


def _dijkstra_state_level(
    system: "GridworldFixedTAMPSystem",
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
) -> float:
    """Compute shortest path using Dijkstra's algorithm at state level with teleporters.

    The graph consists of:
    - Start position
    - Goal position
    - All teleporter portal positions

    Edges:
    - Direct Manhattan distance between any two positions
    - Cost 1 for teleporting between paired portals

    Args:
        system: GridworldFixedTAMPSystem instance
        start_x, start_y: Starting position
        goal_x, goal_y: Goal position

    Returns:
        Minimum distance considering teleporters
    """
    import heapq

    # If no teleporters, use direct Manhattan distance
    if not hasattr(system.env, 'portal_positions') or not system.env.portal_positions:
        return float(abs(start_x - goal_x) + abs(start_y - goal_y))

    # Build graph of important positions (start, goal, and all portal positions)
    positions = [(start_x, start_y)]  # Index 0: start
    positions.append((goal_x, goal_y))  # Index 1: goal

    # Add all portal positions
    portal_indices = []  # List of (portal1_idx, portal2_idx) pairs
    for portal1_pos, portal2_pos in system.env.portal_positions:
        idx1 = len(positions)
        positions.append((float(portal1_pos[0]), float(portal1_pos[1])))

        idx2 = len(positions)
        positions.append((float(portal2_pos[0]), float(portal2_pos[1])))

        portal_indices.append((idx1, idx2))

    # Dijkstra's algorithm
    # State: position index in positions list
    distances = {0: 0.0}  # Start at position 0 with distance 0
    pq = [(0.0, 0)]  # (distance, position_idx)
    visited = set()

    while pq:
        curr_dist, curr_idx = heapq.heappop(pq)

        if curr_idx in visited:
            continue
        visited.add(curr_idx)

        if curr_idx == 1:  # Reached goal
            return curr_dist

        curr_x, curr_y = positions[curr_idx]

        # Explore all neighbors
        for next_idx in range(len(positions)):
            if next_idx == curr_idx or next_idx in visited:
                continue

            next_x, next_y = positions[next_idx]

            # Check if this is a teleporter edge
            is_teleporter = False
            for portal1_idx, portal2_idx in portal_indices:
                if (curr_idx == portal1_idx and next_idx == portal2_idx) or \
                   (curr_idx == portal2_idx and next_idx == portal1_idx):
                    is_teleporter = True
                    break

            if is_teleporter:
                # Teleporter cost is 1
                edge_cost = 1.0
            else:
                # Normal movement: Manhattan distance
                edge_cost = abs(curr_x - next_x) + abs(curr_y - next_y)

            new_dist = curr_dist + edge_cost

            if next_idx not in distances or new_dist < distances[next_idx]:
                distances[next_idx] = new_dist
                heapq.heappush(pq, (new_dist, next_idx))

    # If goal not reached (shouldn't happen), return direct Manhattan distance
    return float(abs(start_x - goal_x) + abs(start_y - goal_y))


def compute_true_distance_gridworld(
    system: "GridworldTAMPSystem",
    start_state: "GraphInstance",
    goal_node_atoms: set[GroundAtom],
) -> float:
    """Compute true minimum distance from start state to goal node using Dijkstra's algorithm.

    This function considers teleporters when computing the shortest path at the state level.

    Args:
        system: GridworldTAMPSystem instance
        start_state: Low-level state (GraphInstance observation)
        goal_node_atoms: Set of atoms defining the goal node

    Returns:
        Minimum Manhattan distance from start_state to any state satisfying goal_node_atoms

    Example:
        Grid is 3x3 cells with 10x10 states per cell (total 30x30 continuous space)
        Start state: robot at (0, 2)
        Goal node: InRow1, InCol1 (cell (1,1))
        The closest state in cell (1,1) is at position (10, 10)
        True distance = |0-10| + |2-10| = 10 + 8 = 18
    """
    # Extract robot position from start state
    robot_node = start_state.nodes[0]  # type=0 is robot
    robot_x, robot_y = robot_node[1], robot_node[2]

    # Check if goal is GoalReached predicate
    has_goal_reached = any("GoalReached" in str(atom) for atom in goal_node_atoms)

    if has_goal_reached:
        # Goal is to reach the goal position - extract from environment
        goal_node = start_state.nodes[1]  # type=1 is goal
        goal_x, goal_y = goal_node[1], goal_node[2]

        # Manhattan distance to goal
        distance = abs(robot_x - goal_x) + abs(robot_y - goal_y)
        return float(distance)

    # Parse goal atoms to determine target cell
    # Atoms are like: InRow1(robot0), InCol2(robot0)
    target_row = None
    target_col = None

    for atom in goal_node_atoms:
        atom_str = str(atom)
        if "InRow" in atom_str or "Row" in atom_str:
            # Extract row number from atom like "InRow1(robot0)" or "Row1(robot0)"
            for i in range(system.env.num_cells):
                if f"Row{i}" in atom_str or f"InRow{i}" in atom_str:
                    target_row = i
                    break
        elif "InCol" in atom_str or "Col" in atom_str:
            # Extract col number from atom like "InCol2(robot0)" or "Col2(robot0)"
            for i in range(system.env.num_cells):
                if f"Col{i}" in atom_str or f"InCol{i}" in atom_str:
                    target_col = i
                    break

    if target_row is None or target_col is None:
        # Incomplete goal specification
        return float("inf")

    # Compute cell boundaries
    # Each cell spans [row*states_per_cell, (row+1)*states_per_cell]
    num_states_per_cell = system.env.num_states_per_cell

    # Target cell boundaries in continuous space
    target_x_min = target_col * num_states_per_cell
    target_x_max = (target_col + 1) * num_states_per_cell
    target_y_min = target_row * num_states_per_cell
    target_y_max = (target_row + 1) * num_states_per_cell

    # Find closest point in target cell to robot position
    # Clamp robot position to cell boundaries
    closest_x = np.clip(robot_x, target_x_min, target_x_max)
    closest_y = np.clip(robot_y, target_y_min, target_y_max)

    # Manhattan distance
    distance = abs(robot_x - closest_x) + abs(robot_y - closest_y)

    return float(distance)

def compute_graph_node_distance(
    planning_graph: PlanningGraph,
    start_node_atoms: set[GroundAtom],
    goal_node_atoms: set[GroundAtom]
):
    path = planning_graph.find_shortest_path(start_node_atoms, goal_node_atoms)
    dist = 0
    for edge in path:
        dist += edge.cost
    return dist

    

def compute_true_distance(
    system: ImprovisationalTAMPSystem,
    start_state: ObsType,
    goal_node_atoms: set[GroundAtom],
) -> float:
    """Compute true minimum distance from start state to goal node.

    Dispatches to domain-specific implementations.

    Args:
        system: TAMP system instance
        start_state: Low-level state observation
        goal_node_atoms: Set of atoms defining the goal node

    Returns:
        Minimum distance from start_state to any state satisfying goal_node_atoms
    """
    from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem

    if isinstance(system, GridworldTAMPSystem):
        return compute_true_distance_gridworld(system, start_state, goal_node_atoms)
    elif isinstance(system, GridworldFixedTAMPSystem):
        return compute_true_distance_gridworld_fixed(system, start_state, goal_node_atoms)
    else:
        print(f"True distance computation not implemented for {type(system).__name__}")
        return -42

def compute_true_node_distance(
    system: ImprovisationalTAMPSystem,
    start_states: list[ObsType],
    goal_node_atoms: set[GroundAtom],
) -> float:
    dists = []
    for s in start_states:
        dists.append(compute_true_distance(system, s, goal_node_atoms))
    return np.max(dists)