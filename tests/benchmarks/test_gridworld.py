"""Tests for gridworld environment."""

import numpy as np
import pytest

from tamp_improv.benchmarks.gridworld import (
    GridworldEnv,
    GridworldTAMPSystem,
)


def test_gridworld_env_initialization():
    """Test that gridworld environment initializes correctly."""
    print("\n" + "=" * 80)
    print("TESTING GRIDWORLD ENVIRONMENT INITIALIZATION")
    print("=" * 80)

    # Create environment with default parameters
    env = GridworldEnv(num_cells=2, num_states_per_cell=5, num_teleporters=1, seed=42)

    print(f"\nEnvironment parameters:")
    print(f"  num_cells: {env.num_cells}")
    print(f"  num_states_per_cell: {env.num_states_per_cell}")
    print(f"  grid_size: {env.grid_size}")
    print(f"  num_teleporters: {env.num_teleporters}")

    # Check portal cell pairs are initialized
    assert len(env.portal_cell_pairs) == 1, "Should have 1 portal pair"
    start_cell, end_cell = env.portal_cell_pairs[0]
    print(f"\nPortal cell pair:")
    print(f"  Start cell: {start_cell}")
    print(f"  End cell: {end_cell}")

    # End portal should always be in goal cell (1, 1) for num_cells=2
    assert end_cell == (1, 1), "End portal should be in goal cell"
    # Start portal should be in a different cell
    assert start_cell != end_cell, "Start and end portals should be in different cells"

    # Reset environment
    obs, info = env.reset(seed=123)

    print(f"\nAfter reset:")
    print(f"  Robot position: {env.robot_pos}")
    print(f"  Goal position: {env.goal_pos}")
    print(f"  Robot cell: {info['robot_cell']}")
    print(f"  Goal cell: {info['goal_cell']}")

    # Check robot starts in cell (0, 0)
    robot_cell = env.robot_pos // env.num_states_per_cell
    assert tuple(robot_cell) == (0, 0), "Robot should start in cell (0, 0)"

    # Check goal is in cell (1, 1)
    goal_cell = env.goal_pos // env.num_states_per_cell
    assert tuple(goal_cell) == (1, 1), "Goal should be in cell (1, 1)"

    # Check portals are placed
    assert len(env.portals) == 1, "Should have 1 portal pair after reset"
    portal_start, portal_end = env.portals[0]

    print(f"\nPortal positions:")
    print(f"  Start portal: {portal_start}")
    print(f"  End portal: {portal_end}")

    # Check portal positions are within their assigned cells
    portal_start_cell = portal_start // env.num_states_per_cell
    portal_end_cell = portal_end // env.num_states_per_cell
    assert tuple(portal_start_cell) == start_cell, "Start portal should be in start cell"
    assert tuple(portal_end_cell) == end_cell, "End portal should be in end cell"

    # Check observation is a GraphInstance
    assert hasattr(obs, 'nodes'), "Observation should be a GraphInstance"
    print(f"\nObservation graph:")
    print(f"  Number of nodes: {len(obs.nodes)}")
    print(f"  Node shapes: {obs.nodes.shape}")

    # Should have robot, goal, and 2 portal nodes
    assert len(obs.nodes) == 4, "Should have 4 nodes (robot, goal, 2 portals)"

    print("\n✓ Environment initialization test passed!")


def test_gridworld_portal_consistency():
    """Test that portal cell pairs remain consistent across resets."""
    print("\n" + "=" * 80)
    print("TESTING PORTAL CONSISTENCY ACROSS RESETS")
    print("=" * 80)

    env = GridworldEnv(num_cells=3, num_states_per_cell=4, num_teleporters=2, seed=42)

    # Get initial portal cell pairs
    initial_pairs = env.portal_cell_pairs.copy()
    print(f"\nInitial portal cell pairs:")
    for i, (start, end) in enumerate(initial_pairs):
        print(f"  Portal {i}: {start} → {end}")

    # Reset multiple times and check cell pairs don't change
    for episode in range(5):
        obs, info = env.reset(seed=100 + episode)

        # Portal cell pairs should remain the same
        assert env.portal_cell_pairs == initial_pairs, \
            f"Portal cell pairs changed on episode {episode}"

        # But portal positions should vary
        portal_positions = env.portals
        print(f"\nEpisode {episode} portal positions:")
        for i, (start_pos, end_pos) in enumerate(portal_positions):
            start_cell = tuple(start_pos // env.num_states_per_cell)
            end_cell = tuple(end_pos // env.num_states_per_cell)
            print(f"  Portal {i}: {start_pos} (cell {start_cell}) → {end_pos} (cell {end_cell})")

            # Verify positions are in correct cells
            expected_start_cell, expected_end_cell = initial_pairs[i]
            assert start_cell == expected_start_cell, \
                f"Start portal {i} not in expected cell"
            assert end_cell == expected_end_cell, \
                f"End portal {i} not in expected cell"

    print("\n✓ Portal consistency test passed!")


def test_gridworld_teleport_action():
    """Test that teleport action works correctly."""
    print("\n" + "=" * 80)
    print("TESTING TELEPORT ACTION")
    print("=" * 80)

    env = GridworldEnv(num_cells=2, num_states_per_cell=5, num_teleporters=1, seed=42)
    obs, info = env.reset(seed=123)

    portal_start, portal_end = env.portals[0]
    print(f"\nPortal locations:")
    print(f"  Start: {portal_start}")
    print(f"  End: {portal_end}")

    # Move robot to start portal position
    env.robot_pos = portal_start.copy()
    initial_pos = env.robot_pos.copy()
    print(f"\nRobot at start portal: {initial_pos}")

    # Use teleport action (action 4)
    obs, reward, terminated, truncated, info = env.step(4)

    print(f"After teleport: {env.robot_pos}")
    print(f"Expected: {portal_end}")

    # Robot should now be at end portal
    assert np.array_equal(env.robot_pos, portal_end), \
        "Robot should teleport to end portal"

    # Teleport back
    obs, reward, terminated, truncated, info = env.step(4)
    print(f"After teleport back: {env.robot_pos}")

    # Robot should be back at start portal
    assert np.array_equal(env.robot_pos, portal_start), \
        "Robot should teleport back to start portal"

    # Test teleport from non-portal location (should have no effect)
    env.robot_pos = np.array([2, 2])
    non_portal_pos = env.robot_pos.copy()
    print(f"\nRobot at non-portal position: {non_portal_pos}")

    obs, reward, terminated, truncated, info = env.step(4)
    print(f"After teleport attempt: {env.robot_pos}")

    # Robot should not move
    assert np.array_equal(env.robot_pos, non_portal_pos), \
        "Teleport from non-portal position should have no effect"

    print("\n✓ Teleport action test passed!")


def test_gridworld_system_initialization():
    """Test that GridworldTAMPSystem initializes correctly."""
    print("\n" + "=" * 80)
    print("TESTING GRIDWORLD TAMP SYSTEM")
    print("=" * 80)

    system = GridworldTAMPSystem.create_default(
        num_cells=2,
        num_states_per_cell=5,
        num_teleporters=1,
        seed=42,
    )

    print(f"\nSystem components:")
    print(f"  Types: {len(system.types)}")
    print(f"  Predicates: {len(system.predicates)}")
    print(f"  Operators: {len(system.operators)}")
    print(f"  Skills: {len(system.skills)}")

    # Check predicates
    predicates = system.predicates
    pred_names = {p.name for p in predicates}
    print(f"\nPredicates: {sorted(pred_names)}")

    # Should have InRow0, InRow1, InCol0, InCol1, GoalReached
    assert 'InRow0' in pred_names, "Should have InRow0 predicate"
    assert 'InRow1' in pred_names, "Should have InRow1 predicate"
    assert 'InCol0' in pred_names, "Should have InCol0 predicate"
    assert 'InCol1' in pred_names, "Should have InCol1 predicate"
    assert 'GoalReached' in pred_names, "Should have GoalReached predicate"

    # Check skills
    skill_names = {s._lifted_operator.name.split('_')[0] for s in system.skills}
    print(f"\nSkill types: {sorted(skill_names)}")

    assert 'MoveUp' in skill_names, "Should have MoveUp skill"
    assert 'MoveRight' in skill_names, "Should have MoveRight skill"
    assert 'NavigateToGoal' in skill_names, "Should have NavigateToGoal skill"

    # Reset and check perceiver
    obs, info = system.reset(seed=123)
    objects, atoms, goal = system.perceiver.reset(obs, info)

    print(f"\nPerceiver output:")
    print(f"  Objects: {objects}")
    print(f"  Initial atoms: {sorted(str(a) for a in atoms)}")
    print(f"  Goal atoms: {sorted(str(a) for a in goal)}")

    # Check initial atoms
    atom_strs = {str(a) for a in atoms}
    assert '(InRow0 robot0)' in atom_strs, "Robot should be in row 0"
    assert '(InCol0 robot0)' in atom_strs, "Robot should be in col 0"

    # Check goal atoms
    goal_strs = {str(a) for a in goal}
    assert '(InRow1 robot0)' in goal_strs, "Goal should be in row 1"
    assert '(InCol1 robot0)' in goal_strs, "Goal should be in col 1"
    assert '(GoalReached robot0)' in goal_strs, "Goal should include GoalReached"

    print("\n✓ System initialization test passed!")


def test_gridworld_pure_planning_solution():
    """Test that we can find a pure planning solution (without shortcuts)."""
    print("\n" + "=" * 80)
    print("TESTING PURE PLANNING SOLUTION")
    print("=" * 80)

    system = GridworldTAMPSystem.create_default(
        num_cells=10,
        num_states_per_cell=10,
        num_teleporters=1,
        seed=42,
    )

    # Reset environment
    obs, info = system.reset(seed=123)
    objects, atoms, goal = system.perceiver.reset(obs, info)

    print(f"\nInitial state: {sorted(str(a) for a in atoms)}")
    print(f"Goal: {sorted(str(a) for a in goal)}")

    # Create planning graph using the approach
    from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
    from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy

    policy = MultiRLPolicy(seed=42)
    approach = ImprovisationalTAMPApproach(system, policy, seed=42)
    approach.training_mode = True
    approach.reset(obs, info)

    # Check planning graph
    graph = approach.planning_graph
    print(f"\nPlanning graph:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")

    # Print node atoms
    print(f"\nGraph nodes:")
    for node in graph.nodes:
        print(f"  Node {node.id}: {sorted(str(a) for a in node.atoms)}")

    # Find shortest path
    path = graph.find_shortest_path(atoms, goal)
    print(f"\nShortest path found with {len(path)} edges:")
    for i, edge in enumerate(path):
        source_atoms = sorted(str(a) for a in edge.source.atoms)
        target_atoms = sorted(str(a) for a in edge.target.atoms)
        op_name = edge.operator.name if edge.operator else "None"
        print(f"  {i+1}. {op_name}: {source_atoms} → {target_atoms}")

    # Expected path: (Row0,Col0) → (Row0,Col1) → (Row1,Col1,GoalReached)
    # Should use MoveRight, then MoveUp, then NavigateToGoal
    # So minimum 3 edges
    assert len(path) >= 3, f"Expected at least 3 edges in path, got {len(path)}"

    # Verify path ends at goal
    final_atoms = path[-1].target.atoms
    assert goal.issubset(final_atoms), "Path should reach goal"

    print("\n✓ Pure planning solution test passed!")


if __name__ == "__main__":
    # test_gridworld_env_initialization()
    # test_gridworld_portal_consistency()
    # test_gridworld_teleport_action()
    # test_gridworld_system_initialization()
    test_gridworld_pure_planning_solution()
    print("\n" + "=" * 80)
    print("ALL GRIDWORLD TESTS PASSED!")
    print("=" * 80)
