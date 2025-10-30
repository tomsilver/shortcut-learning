"""Tests for obstacle tower environment."""

import numpy as np
import pytest

from shortcut_learning.problems.obstacle_tower.env import ObstacleTowerEnv
from shortcut_learning.problems.obstacle_tower.system import (
    BaseObstacleTowerTAMPSystem,
    ObstacleTowerTAMPSystem,
)


def test_obstacle_tower_env_basic():
    """Test basic obstacle tower environment functionality."""
    env = ObstacleTowerEnv(num_obstacle_blocks=3, stack_blocks=True)

    # Test reset
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert obs.nodes.shape[0] == 6  # robot + 3 obstacles + target + target_area
    assert obs.nodes.shape[1] > 0  # Has features

    # Test action space
    assert env.action_space.shape == (8,)  # 7 joints + 1 gripper

    # Test step (without executing, just checking the interface)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()


def test_obstacle_tower_system_initialization():
    """Test TAMP system initialization."""
    system = BaseObstacleTowerTAMPSystem.create_default(
        num_obstacle_blocks=3,
        stack_blocks=True,
        seed=42,
    )

    # Check system properties
    assert system.name == "ObstacleTowerTAMPSystem"
    assert system._get_domain_name() == "obstacle-tower-domain"

    # Check planning components
    assert len(system.components.types) == 2  # robot_type, object_type
    assert len(system.components.operators) == 6  # Pick, PickFromTarget, Place, PlaceInTarget, Unstack, Stack
    assert len(system.components.skills) == 6

    # Test reset
    obs, info = system.reset(seed=42)
    assert obs is not None
    assert obs.nodes.shape[0] == 6


def test_obstacle_tower_improvisational_system():
    """Test improvisational TAMP system initialization."""
    system = ObstacleTowerTAMPSystem.create_default(
        num_obstacle_blocks=3,
        stack_blocks=True,
        seed=42,
    )

    assert system.name == "ObstacleTowerTAMPSystem"

    # Test reset
    obs, info = system.reset(seed=42)
    assert obs is not None


def test_obstacle_tower_different_configurations():
    """Test different environment configurations."""
    # Test with 2 obstacle blocks
    env2 = ObstacleTowerEnv(num_obstacle_blocks=2, stack_blocks=True)
    obs2, _ = env2.reset(seed=42)
    assert obs2.nodes.shape[0] == 5  # robot + 2 obstacles + target + target_area
    env2.close()

    # Test with 4 obstacle blocks
    env4 = ObstacleTowerEnv(num_obstacle_blocks=4, stack_blocks=True)
    obs4, _ = env4.reset(seed=42)
    assert obs4.nodes.shape[0] == 7  # robot + 4 obstacles + target + target_area
    env4.close()

    # Test with blocks side-by-side instead of stacked
    env_side = ObstacleTowerEnv(num_obstacle_blocks=3, stack_blocks=False)
    obs_side, _ = env_side.reset(seed=42)
    assert obs_side.nodes.shape[0] == 6
    env_side.close()


if __name__ == "__main__":
    print("Testing obstacle tower environment...")
    test_obstacle_tower_env_basic()
    print("✓ Basic environment test passed")

    test_obstacle_tower_system_initialization()
    print("✓ System initialization test passed")

    test_obstacle_tower_improvisational_system()
    print("✓ Improvisational system test passed")

    test_obstacle_tower_different_configurations()
    print("✓ Configuration test passed")

    print("\n✓ All tests passed!")
