"""Hydra experiment script for contrastive learning pipeline.

This script provides a Hydra-based interface to the contrastive learning
pipeline for TAMP shortcuts.
"""

from pathlib import Path
from typing import Any, Type

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.contrastive_pipeline import (
    ContrastivePipelineConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedTAMPSystem
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem


SYSTEM_CLASSES: dict[str, Type[ImprovisationalTAMPSystem[Any, Any]]] = {
    "GraphObstacle2DTAMPSystem": GraphObstacle2DTAMPSystem,
    "GridworldTAMPSystem": GridworldTAMPSystem,
    "GridworldFixedTAMPSystem": GridworldFixedTAMPSystem,
}


@hydra.main(version_base=None, config_path="configs", config_name="gridworld_fixed")
def main(cfg: DictConfig) -> float:
    """Main function for contrastive pipeline experiment.

    Args:
        cfg: Hydra configuration

    Returns:
        Success rate as metric for Hydra sweeps
    """
    import sys

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80)
    print("Contrastive Learning Pipeline for TAMP")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Get output directory from Hydra
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"\nOutput directory: {output_dir}")

    # Create system
    system_cls = SYSTEM_CLASSES[cfg.env_name]
    system_kwargs = {
        "seed": cfg.seed,
        "render_mode": cfg.render_mode if cfg.render_mode != "null" else None,
    }

    # Add system-specific parameters
    if hasattr(cfg, "num_cells"):
        system_kwargs["num_cells"] = cfg.num_cells
    if hasattr(cfg, "num_states_per_cell"):
        system_kwargs["num_states_per_cell"] = cfg.num_states_per_cell
    if hasattr(cfg, "num_teleporters"):
        system_kwargs["num_teleporters"] = cfg.num_teleporters

    system = system_cls.create_default(**system_kwargs)  # type: ignore
    print(f"System created: {cfg.env_name}")

    # Create pipeline configuration
    pipeline_config = ContrastivePipelineConfig(
        seed=cfg.seed,
        collect_episodes=cfg.collect_episodes,
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        gamma=cfg.gamma,
        repetition_factor=cfg.repetition_factor,
        policy_temperature=cfg.policy_temperature,
        eval_temperature=cfg.eval_temperature,
        iters_per_epoch=cfg.iters_per_epoch,
        learn_frequency=cfg.learn_frequency,
        num_epochs=cfg.num_epochs,
        trajectories_per_epoch=cfg.trajectories_per_epoch,
        max_episode_steps=cfg.max_episode_steps,
        eval_rollouts=cfg.eval_rollouts,
        eval_max_steps=cfg.eval_max_steps,
        fast_eval=cfg.get("fast_eval", False),
        debug=cfg.debug,
    )

    # Run pipeline
    results = train_and_evaluate(system, pipeline_config, output_dir)

    # Return success rate as metric
    return results["evaluation"]["success_rate"]


if __name__ == "__main__":
    main()
