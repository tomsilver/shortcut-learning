"""Training script for SLAP using pipeline V2 with unified heuristic interface.

This script uses the new simplified pipeline (V2) that treats all heuristic methods
uniformly through a common interface. The pipeline creates heuristics and policies
directly from the config.

Heuristic types:
- "rollouts": Random rollout-based heuristic
- "v4": Distance heuristic V4 with CRL (TODO)
- "random": Random pruning baseline (TODO)

Configuration:
- heuristic_type: Required. Determines which heuristic class to use
- policy_type: Defaults to "multiRL"
- Parameters prefixed with rl_ are used for MultiRL policy
- All other parameters are used for heuristic training
"""

import pickle
from pathlib import Path
from typing import Any, Type

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.pipeline_v2 import (
    PipelineResults,
    run_pipeline,
)
from tamp_improv.approaches.improvisational.policies.base import (  # noqa: F401
    GoalConditionedTrainingData,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedTAMPSystem
from tamp_improv.benchmarks.gridworld_continuous import GridworldContinuousTAMPSystem
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)

SYSTEM_CLASSES: dict[str, Type[ImprovisationalTAMPSystem[Any, Any]]] = {
    "GraphObstacle2DTAMPSystem": GraphObstacle2DTAMPSystem,
    "GraphObstacleTowerTAMPSystem": GraphObstacleTowerTAMPSystem,
    "ClutteredDrawerTAMPSystem": ClutteredDrawerTAMPSystem,
    "CleanupTableTAMPSystem": CleanupTableTAMPSystem,
    "GridworldTAMPSystem": GridworldTAMPSystem,
    "GridworldFixedTAMPSystem": GridworldFixedTAMPSystem,
    "GridworldContinuousTAMPSystem": GridworldContinuousTAMPSystem,
}


import inspect


def filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    valid_params = sig.parameters

    # If the function has **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()):
        return kwargs

    return {k: v for k, v in kwargs.items() if k in valid_params}


def save_results(results: PipelineResults, path: Path):
    """Save PipelineResults directly — it's already fully serializable."""
    with open(path / "results.pkl", "wb") as f:
        pickle.dump(results, f)


@hydra.main(version_base=None, config_path="configs", config_name="unit_test")
def main(cfg: DictConfig) -> float:
    """Main training function using pipeline V2."""
    if torch.cuda.is_available():
        _dummy = torch.zeros(1, device="cuda")

    print("=" * 80)
    print(f"Training SLAP with Pipeline V2 on {cfg.env.name}")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Create system
    system_cls = SYSTEM_CLASSES[cfg.env.name]

    system_kwargs = {
        "seed": cfg.seed,
        "render_mode": cfg.render_mode if cfg.render_mode != "null" else None,
        "n_blocks": cfg.env.n_blocks,
        "num_obstacle_blocks": cfg.env.num_obstacle_blocks,
        "num_cells": cfg.env.num_cells,
        "num_states_per_cell": cfg.env.num_states_per_cell,
        "grid_size": cfg.env.num_states_per_cell * cfg.env.num_cells,
        "num_teleporters": cfg.env.num_teleporters,
    }

    create_fn = system_cls.create_default
    filtered_kwargs = filter_kwargs(create_fn, system_kwargs)

    print(f"\nCreating system: {cfg.env.name} with kwargs: {filtered_kwargs}")
    system = create_fn(**filtered_kwargs)
    # Convert config to dict for pipeline
    # config_dict = OmegaConf.to_container(cfg, resolve=True)
    # assert isinstance(config_dict, dict), "Config should be a dictionary"

    print(f"\nHeuristic type: {cfg.heuristic.type}")
    print(f"Policy type: {cfg.policy.type}")
    print(f"Debug mode: {cfg.debug}")

    # Get Hydra output directory for results
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Run pipeline V2 (no caching)
    results = run_pipeline(
        system=system,
        cfg=cfg,
        output_dir=output_dir,
        system_cls=system_cls,
        system_kwargs=filtered_kwargs,
    )

    # Print final metrics
    if results.avg_success_rate is None:
        raise RuntimeError("Pipeline returned no evaluation metrics!")

    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(f"Success Rate: {results.avg_success_rate:.2%}")
    print(f"Average Steps: {results.avg_steps:.2f}")
    print(f"Average Reward: {results.avg_reward:.3f}")
    print("=" * 80)

    print("Times:", results.times)

    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"env_name: {cfg.env.name}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"heuristic_type: {cfg.heuristic.type}\n")
        f.write(f"policy_type: {cfg.policy.type}\n")
        f.write(f"success_rate: {results.avg_success_rate}\n")
        f.write(f"avg_steps: {results.avg_steps}\n")
        f.write(f"avg_reward: {results.avg_reward}\n")

    # Save detailed results if debug mode
    if cfg.debug:
        results_detail_file = output_dir / "results_detail.txt"
        with open(results_detail_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PIPELINE V2 DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")

            # Collection stats
            if results.unique_shortcuts is not None:
                f.write("COLLECTION:\n")
                f.write(
                    f"  Unique shortcuts: {len(results.unique_shortcuts)}\n"
                )
                f.write(
                    f"  Nodes: {len(results.node_atoms) if results.node_atoms else 0}\n"
                )
                f.write("\n")

            # Pruning stats
            if results.pruned_shortcuts is not None:
                f.write("PRUNING:\n")
                f.write(
                    f"  Shortcuts after pruning: {len(results.pruned_shortcuts)}\n"
                )
                f.write("\n")

            # Evaluation
            f.write("EVALUATION:\n")
            f.write(f"  Success rate: {results.avg_success_rate:.2%}\n")
            f.write(f"  Avg steps: {results.avg_steps:.2f}\n")
            f.write(f"  Avg reward: {results.avg_reward:.3f}\n")
    print(output_dir)

    save_results(results, output_dir)
    return results.avg_success_rate


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
