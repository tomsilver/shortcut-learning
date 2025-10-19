"""Test base random approach."""

import pytest

from shortcut_learning.configs import *
from shortcut_learning.methods.pipeline import *
from shortcut_learning.methods.random_approach import RandomApproach
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem


def run_episode(system, approach, max_steps):
    """Run single episode with approach."""
    obs, info = system.reset()
    step_result = approach.reset(obs, info)

    # Process first step
    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    if terminated or truncated:
        return 1

    # Process remaining steps
    for step in range(1, max_steps):
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        if terminated or truncated:
            return step + 1
    return max_steps


@pytest.mark.parametrize(
    "system_cls,max_steps",
    [(BaseObstacle2DTAMPSystem, 100)],
)
def test_pure_rl_approach(system_cls, max_steps):
    """Test random approach on different environments."""
    system = system_cls.create_default(seed=42)

    approach_config = ApproachConfig(approach_type="pure_rl", approach_name="example")

    policy_config = PolicyConfig()

    collect_config = CollectionConfig()
    train_config = TrainingConfig(num_episodes=1)
    eval_config = EvaluationConfig()

    approach = initialize_approach(system, approach_config, policy_config)

    print(approach)

    train_data = collect_approach(approach, collect_config)

    trained_approach = train_approach(approach, train_config, train_data)
    # trained_approach = approach

    metrics = evaluate_approach(system, trained_approach, eval_config)

    print(metrics)

    assert metrics is not None

    # approach = RandomApproach(system, seed=42)

    # steps = run_episode(system, approach, max_steps)
    # assert steps <= max_steps


if __name__ == "__main__":
    test_pure_rl_approach(BaseObstacle2DTAMPSystem, 100)
