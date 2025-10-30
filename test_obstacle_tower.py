"""Test script for obstacle tower environment."""

from shortcut_learning.problems.obstacle_tower.env import ObstacleTowerEnv
from shortcut_learning.problems.obstacle_tower.system import ObstacleTowerTAMPSystem


def test_env():
    """Test basic environment functionality."""
    print("Testing ObstacleTowerEnv...")
    env = ObstacleTowerEnv(num_obstacle_blocks=3, stack_blocks=True)

    obs, info = env.reset(seed=42)
    print(f"Observation type: {type(obs)}")
    print(f"Observation shape: nodes={obs.nodes.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Take a random step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step completed - reward: {reward}, terminated: {terminated}")

    env.close()
    print("✓ Environment test passed!")


def test_system():
    """Test TAMP system initialization."""
    print("\nTesting ObstacleTowerTAMPSystem...")
    system = ObstacleTowerTAMPSystem.create_default(
        num_obstacle_blocks=3,
        stack_blocks=True,
        seed=42
    )

    print(f"System name: {system.name}")
    print(f"Domain name: {system._get_domain_name()}")
    print(f"Number of operators: {len(system.components.operators)}")
    print(f"Number of skills: {len(system.components.skills)}")
    print(f"Number of types: {len(system.components.types)}")

    # Test reset
    obs, info = system.reset(seed=42)
    print(f"System reset successful - observation shape: {obs.nodes.shape}")

    print("✓ System test passed!")


if __name__ == "__main__":
    test_env()
    test_system()
    print("\n✓ All tests passed!")
