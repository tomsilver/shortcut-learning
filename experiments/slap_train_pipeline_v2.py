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

from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData )

from pathlib import Path
from typing import Any, Type
import pickle

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.pipeline_v2 import run_pipeline, PipelineResults
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedTAMPSystem
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
}


import inspect

def filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    valid_params = sig.parameters

    # If the function has **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD
           for p in valid_params.values()):
        return kwargs

    return {
        k: v for k, v in kwargs.items()
        if k in valid_params
    }

from dataclasses import dataclass, asdict
from typing import Any, Optional

@dataclass
class SerializableResults:
    graph_distances: Optional[dict[tuple[int, int], float]] = None
    heuristic_training_history: Optional[dict[str, Any]] = None
    heuristic_quality_results: Optional[dict[str, Any]] = None
    shortcut_quality_results: Optional[dict[str, Any]] = None
    times: Optional[dict[str, float]] = None
    evaluation_results: Optional[Any] = None  # Metrics is probably pickleable

    training_data: Optional[GoalConditionedTrainingData] = None
    pruned_training_data: Optional[GoalConditionedTrainingData] = None
    final_training_data: Optional[GoalConditionedTrainingData] = None
    teleporter_locations: Optional[list] = None
    successful_shortcut_paths: Optional[list[list[Any]]] = None

def extract_serializable_results(results: PipelineResults) -> SerializableResults:
    teleporter_locations = None
    if results.approach and results.approach.system:
        # Try to find portal_positions in the environment
        env = results.approach.system.env
        # Unwrap just in case
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        if hasattr(env, "portal_positions") and hasattr(env, "num_states_per_cell"):
            teleporter_locations = []
            for p1, p2 in env.portal_positions:
                c1 = (int(p1[0] // env.num_states_per_cell), int(p1[1] // env.num_states_per_cell))
                c2 = (int(p2[0] // env.num_states_per_cell), int(p2[1] // env.num_states_per_cell))
                teleporter_locations.append((c1, c2))
        elif hasattr(env, "portal_positions"):
            teleporter_locations = env.portal_positions

    return SerializableResults(
        graph_distances=results.graph_distances,
        heuristic_training_history=results.heuristic_training_history,
        heuristic_quality_results=results.heuristic_quality_results,
        shortcut_quality_results=results.shortcut_quality_results,
        times=results.times,
        evaluation_results=results.evaluation_results,
        training_data=results.training_data,
        pruned_training_data=results.pruned_training_data,
        final_training_data=results.pruned_training_data,
        teleporter_locations=teleporter_locations,
        successful_shortcut_paths=results.shortcut_quality_results.get('successful_shortcut_paths') if results.shortcut_quality_results else None,
    )



def save_serializable_results(results: PipelineResults, path: Path):
    serializable = extract_serializable_results(results)
    with open(path / "results.pkl", "wb") as f:
        pickle.dump(serializable, f)


@hydra.main(version_base=None, config_path="configs", config_name="unit_test")
def main(cfg: DictConfig) -> float:
    """Main training function using pipeline V2."""
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
    )

    

    # Extract metrics
    metrics = results.evaluation_results
    if metrics is None:
        raise RuntimeError("Pipeline returned no evaluation metrics!")

    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Steps: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.3f}")
    print("=" * 80)

    print("Times:", results.times)

    OmegaConf.save(cfg, output_dir / "config.yaml")

    heuristic = results.heuristic
    try:
        heuristic.save(output_dir / "heuristic")
    except:
        print("Heuristic has no save() method, skipping heuristic save.")
    
    policy = results.policy
    try:
        policy.save(output_dir / "policy")
    except:
        print("Policy has no save() method, skipping policy save.")

    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"env_name: {cfg.env.name}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"heuristic_type: {cfg.heuristic.type}\n")
        f.write(f"policy_type: {cfg.policy.type}\n")
        f.write(f"success_rate: {metrics.success_rate}\n")
        f.write(f"avg_steps: {metrics.avg_episode_length}\n")
        f.write(f"avg_reward: {metrics.avg_reward}\n")

    # Save detailed results if debug mode
    if cfg.debug:
        results_detail_file = output_dir / "results_detail.txt"
        with open(results_detail_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PIPELINE V2 DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")

            # Collection stats
            if results.training_data:
                f.write("COLLECTION:\n")
                f.write(f"  Unique shortcuts: {len(results.training_data.unique_shortcuts)}\n")
                f.write(
                    f"  State-node pairs: {len(results.training_data.valid_shortcuts)}\n"
                )
                f.write(
                    f"  Graph nodes: {len(results.training_data.graph.nodes) if results.training_data.graph else 0}\n"
                )
                f.write("\n")

            # Heuristic quality
            if results.heuristic_quality_results:
                f.write("HEURISTIC QUALITY:\n")
                f.write(
                    f"  Avg estimated distance: {results.heuristic_quality_results['avg_estimated_distance']:.2f}\n"
                )
                f.write(
                    f"  Avg graph distance: {results.heuristic_quality_results['avg_graph_distance']:.2f}\n"
                )
                f.write(
                    f"  Avg absolute error: {results.heuristic_quality_results['avg_absolute_error']:.2f}\n"
                )
                f.write("\n")

            # Pruning stats
            if results.pruned_training_data:
                f.write("PRUNING:\n")
                f.write(
                    f"  Shortcuts after pruning: {len(results.pruned_training_data.unique_shortcuts)}\n"
                )
                f.write("\n")



            # Shortcut quality
            if results.shortcut_quality_results:
                # print(results.shortcut_quality_resul)
                f.write("SHORTCUT QUALITY:\n")
                f.write(
                    f"  Avg success rate: {(results.shortcut_quality_results['avg_success_rate']):.2%}\n"
                )
                f.write(
                    f"  Avg steps: {results.shortcut_quality_results['avg_steps']:.1f}\n"
                )
                f.write("\n")

            # Evaluation
            f.write("EVALUATION:\n")
            f.write(f"  Success rate: {metrics.success_rate:.2%}\n")
            f.write(f"  Avg steps: {metrics.avg_episode_length:.2f}\n")
            f.write(f"  Avg reward: {metrics.avg_reward:.3f}s\n")
    print(output_dir)

    save_serializable_results(results, output_dir)
    return metrics.success_rate


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
