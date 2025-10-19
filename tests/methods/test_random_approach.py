"""Test base random approach."""

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)
from shortcut_learning.methods.pipeline import (
    Metrics,
    collect_approach,
    evaluate_approach,
    initialize_approach,
    train_approach,
)
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem


def test_random_approach():
    """Test random approach on different environments."""
    system = BaseObstacle2DTAMPSystem.create_default(seed=42)

    approach_config = ApproachConfig(approach_type="random", approach_name="example")

    policy_config = PolicyConfig()

    collect_config = CollectionConfig()
    train_config = TrainingConfig()
    eval_config = EvaluationConfig(num_episodes=1)

    approach = initialize_approach(system, approach_config, policy_config)

    train_data = collect_approach(  # pylint: disable=assignment-from-none
        approach, collect_config
    )

    trained_approach = train_approach(approach, train_config, train_data)

    metrics = evaluate_approach(system, trained_approach, eval_config)
    assert isinstance(metrics, Metrics)
