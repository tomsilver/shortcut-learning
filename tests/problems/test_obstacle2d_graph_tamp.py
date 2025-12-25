"""Tests for GraphObstacle2D environment with TAMP."""

import pytest
from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from shortcut_learning.problems.obstacle2d.system_graph import (
    BaseGraphObstacle2DTAMPSystem,
    GraphObstacle2DTAMPSystem,
)


@pytest.mark.parametrize(
    "system_cls", [BaseGraphObstacle2DTAMPSystem, GraphObstacle2DTAMPSystem]
)
def test_graph_obstacle2d_tamp_system(system_cls):
    """Test GraphObstacle2D environment with TAMP planner."""
    # Create TAMP system
    tamp_system = system_cls.create_default(render_mode="rgb_array", seed=42)

    # Verify the environment uses graph observations
    assert hasattr(tamp_system.env.observation_space, "node_space")

    # Create environment with time limit
    env = TimeLimit(tamp_system.env, max_episode_steps=50)

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        planner_id="pyperplan",
    )

    obs, info = env.reset()

    # Verify observation is a GraphInstance
    assert hasattr(obs, "nodes"), "Observation should be a GraphInstance"
    assert hasattr(obs, "edges"), "Observation should be a GraphInstance"

    objects, atoms, goal = tamp_system.perceiver.reset(obs, info)
    print("Objects:", objects)
    print("Initial atoms:", atoms)
    print("Goal:", goal)

    try:
        planner.reset(obs, info)
    except Exception as e:
        print("Error during planner reset:", str(e))
        print(
            "Current problem:",
            planner._current_problem,  # pylint: disable=protected-access
        )
        print("Current domain:", planner._domain)  # pylint: disable=protected-access
        raise

    total_reward = 0
    for step in range(100):
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 100 steps")

    env.close()

    # Verify we got some reward (planning should succeed)
    assert total_reward > 0, "Planning should have achieved the goal"


def test_graph_obstacle2d_system_instantiation():
    """Test that GraphObstacle2DTAMPSystem can be instantiated without errors."""
    # This specifically tests the fix for the _create_wrapped_env abstract method
    system = GraphObstacle2DTAMPSystem.create_default(seed=42)

    assert system is not None
    assert hasattr(system.env, "observation_space")
    assert hasattr(system.env.observation_space, "node_space")

    # Test reset
    obs, info = system.reset()
    assert hasattr(obs, "nodes")
    assert hasattr(obs, "edges")

    # Verify nodes have the correct structure
    # Each node should be: [type, x, y, w, h, id/gripper]
    for node in obs.nodes:
        assert len(node) == 6, "Each node should have 6 features"

    print(f"System instantiated successfully with {len(obs.nodes)} nodes")
