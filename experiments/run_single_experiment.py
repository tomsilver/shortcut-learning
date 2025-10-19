"""Run a single experiment."""

import logging
import time

import hydra
from omegaconf import DictConfig

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)
from shortcut_learning.methods.base_approach import BaseApproach
from shortcut_learning.methods.pipeline import (
    collect_approach,
    evaluate_approach,
    initialize_approach,
    train_approach,
)
from shortcut_learning.problems.base_tamp import BaseTAMPSystem


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(
        f"Running seed={cfg.seed}, system={cfg.system}, approach={cfg.approach}"
    )

    # Create the approach.
    approach_config: ApproachConfig = hydra.utils.instantiate(cfg.approach)
    policy_config: PolicyConfig = hydra.utils.instantiate(cfg.policy)
    collect_config: CollectionConfig = hydra.utils.instantiate(cfg.collection)
    train_config: TrainingConfig = hydra.utils.instantiate(cfg.training)
    eval_config: EvaluationConfig = hydra.utils.instantiate(cfg.evaluation)

    # Create the system.
    system = hydra.utils.instantiate(cfg.system).create_default(seed=cfg.seed)
    assert isinstance(system, BaseTAMPSystem)
    approach = initialize_approach(system, approach_config, policy_config)
    assert isinstance(approach, BaseApproach)

    start_time = time.time()

    train_data = collect_approach(  # pylint: disable=assignment-from-none
        approach, collect_config
    )

    collect_time = time.time()

    trained_approach = train_approach(approach, train_config, train_data)

    train_time = time.time()

    metrics = evaluate_approach(system, trained_approach, eval_config)

    eval_time = time.time()

    metrics.collection_time = collect_time - start_time
    metrics.training_time = train_time - collect_time
    metrics.evaluation_time = eval_time - train_time

    logging.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
