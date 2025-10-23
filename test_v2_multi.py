from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)
from shortcut_learning.methods.pipeline import pipeline_from_configs
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem

system = BaseObstacle2DTAMPSystem.create_default(seed=42)

approach_config = ApproachConfig(
    approach_type='slap_v2', approach_name='example', debug_videos=False, seed=42
)

policy_config = PolicyConfig(policy_type='multi_rl_ppo_v2')

collect_config = CollectionConfig()
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

print('SUCCESS:', metrics)
