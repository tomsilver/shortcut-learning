"""Tests for obstacle2d_hard environment with continuous random positions."""

import numpy as np
from gymnasium.wrappers import TimeLimit

from shortcut_learning.problems.obstacle2d_hard.env import Obstacle2DEnv


def test_obstacle2d_hard_env_loads():
    """Test that the obstacle2d_hard environment loads properly."""
    env = Obstacle2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    obs, info = env.reset()

    # Check observation shape
    assert obs.shape == (15,), f"Expected obs shape (15,), got {obs.shape}"

    # Check that all values are within bounds [0, 1]
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Observation values out of bounds"

    print(f"Initial observation: {obs}")
    print(f"Robot position: {obs[0:2]}")
    print(f"Block 1 position: {obs[4:6]}")
    print(f"Block 2 position: {obs[6:8]}")
    print(f"Target area: x={obs[11]}, y={obs[12]}, w={obs[13]}, h={obs[14]}")

    env.close()


def test_obstacle2d_hard_env_randomization():
    """Test that reset produces different initial states (continuous randomization)."""
    env = Obstacle2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # Collect 10 initial states
    initial_states = []
    target_positions = []
    for i in range(10):
        obs, info = env.reset(seed=i)
        initial_states.append(obs.copy())
        target_positions.append(obs[11])  # Target x position

    # Check that we have diverse initial states
    # Robot x positions should be different
    robot_x_positions = [obs[0] for obs in initial_states]
    unique_robot_x = len(set(np.round(robot_x_positions, 4)))
    print(f"Unique robot x positions: {unique_robot_x}/10")
    assert unique_robot_x >= 8, f"Expected at least 8 different robot positions, got {unique_robot_x}"

    # Block 1 x positions should be different
    block1_x_positions = [obs[4] for obs in initial_states]
    unique_block1_x = len(set(np.round(block1_x_positions, 4)))
    print(f"Unique block 1 x positions: {unique_block1_x}/10")
    assert unique_block1_x >= 8, f"Expected at least 8 different block 1 positions, got {unique_block1_x}"

    # Target area x positions should be different
    unique_target_x = len(set(np.round(target_positions, 4)))
    print(f"Unique target x positions: {unique_target_x}/10")
    assert unique_target_x >= 8, f"Expected at least 8 different target positions, got {unique_target_x}"

    # Check that target positions are within valid bounds [0.1, 0.9]
    # (with width=0.2, center must be at least 0.1 from edges)
    for target_x in target_positions:
        assert 0.1 <= target_x <= 0.9, f"Target x position {target_x} out of valid bounds [0.1, 0.9]"

    print("✓ Continuous randomization working correctly!")

    env.close()


def test_obstacle2d_hard_env_basic_step():
    """Test basic stepping in the environment."""
    env = Obstacle2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    obs, info = env.reset()
    env.action_space.seed(123)

    # Take a few random actions
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check observation shape
        assert obs.shape == (15,), f"Expected obs shape (15,), got {obs.shape}"

        # Check that position/size values are within bounds [0, 1]
        # (gripper status at index 10 can be in range [-1, 1])
        positions = np.concatenate([obs[0:10], obs[11:15]])
        assert np.all(positions >= 0.0) and np.all(positions <= 1.0), "Position values out of bounds"

        # Check gripper status is in valid range
        assert -1.0 <= obs[10] <= 1.0, f"Gripper status {obs[10]} out of range [-1, 1]"

        if terminated or truncated:
            break

    print("✓ Basic stepping works correctly!")

    env.close()
