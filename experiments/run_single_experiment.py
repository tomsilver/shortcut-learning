"""Run a single experiment."""

import logging

import hydra
from omegaconf import DictConfig

from shortcut_learning.methods.base_approach import BaseApproach
from shortcut_learning.problems.base_tamp import BaseTAMPSystem


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(
        f"Running seed={cfg.seed}, system={cfg.system}, approach={cfg.approach}"
    )

    # Create the system.
    system = hydra.utils.instantiate(cfg.system).create_default(seed=cfg.seed)
    assert isinstance(system, BaseTAMPSystem)

    # Create the approach.
    approach = hydra.utils.instantiate(cfg.approach, system, cfg.seed)
    assert isinstance(approach, BaseApproach)


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
