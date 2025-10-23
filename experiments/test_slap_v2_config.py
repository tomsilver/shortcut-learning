"""Quick test to verify SLAP V2 configs work with Hydra."""

import hydra
from omegaconf import DictConfig

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def test_config(cfg: DictConfig) -> None:
    """Test that config instantiation works."""
    print("\n=== Testing SLAP V2 Configuration ===\n")

    # Instantiate all configs
    approach_config: ApproachConfig = hydra.utils.instantiate(cfg.approach)
    policy_config: PolicyConfig = hydra.utils.instantiate(cfg.policy)
    collect_config: CollectionConfig = hydra.utils.instantiate(cfg.collection)
    train_config: TrainingConfig = hydra.utils.instantiate(cfg.training)
    eval_config: EvaluationConfig = hydra.utils.instantiate(cfg.evaluation)

    # Print key settings
    print(f"Approach Type: {approach_config.approach_type}")
    print(f"Policy Type: {policy_config.policy_type}")
    print(f"Collection States/Node: {collect_config.states_per_node}")
    print(f"Collection Perturbation Steps: {collect_config.perturbation_steps}")
    print(f"Training Runs/Shortcut: {train_config.runs_per_shortcut}")
    print(f"Training Max Steps: {train_config.max_env_steps}")
    print(f"Policy Epochs: {policy_config.n_epochs}")
    print(f"Evaluation Episodes: {eval_config.num_episodes}")

    print("\n=== Configuration Valid! ===\n")


if __name__ == "__main__":
    test_config()  # pylint: disable=no-value-for-parameter
