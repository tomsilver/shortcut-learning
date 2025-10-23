"""Test base random approach."""

import pytest

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)
from shortcut_learning.methods.pipeline import (
    pipeline_from_configs,
    Metrics
)
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem
from shortcut_learning.problems.base_tamp import ImprovisationalTAMPSystem

@pytest.mark.parametrize(
    "system_cls",
    [BaseObstacle2DTAMPSystem],
)
def test_multi_slap_approach(system_cls):
    """Test random approach on different environments."""
    system: ImprovisationalTAMPSystem = system_cls.create_default(seed=42)

    print(system)
    print(isinstance(system, ImprovisationalTAMPSystem))
    approach_config = ApproachConfig(
        approach_type="slap", approach_name="example", debug_videos=False
    )

    policy_config = PolicyConfig(policy_type="multi_rl_ppo")

    collect_config = CollectionConfig(max_shortcuts_per_graph=2)
    train_config = TrainingConfig(runs_per_shortcut=1, max_env_steps=2)
    eval_config = EvaluationConfig(num_episodes=1)

    metrics = pipeline_from_configs(
        system,
        approach_config,
        policy_config,
        collect_config,
        train_config,
        eval_config,
    )

    print(metrics)

    assert isinstance(metrics, Metrics)

    # approach = RandomApproach(system, seed=42)

    # steps = run_episode(system, approach, max_steps)
    # assert steps <= max_steps


if __name__ == "__main__":
    test_multi_slap_approach(BaseObstacle2DTAMPSystem)
