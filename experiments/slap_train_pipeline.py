"""Training script for SLAP using modular pipeline with Hydra.

This script uses the new modular pipeline with caching and cascade invalidation.
It supports multiple pruning methods:
- "none": No pruning (collect all shortcuts)
- "rollouts": Prune based on random rollout success rates
- "distance_heuristic": Prune using learned distance heuristic with f(s,s') < min(D(s,s'), K)

To switch between methods, just change the pruning_method in the config file.
"""

from pathlib import Path
from typing import Any, Type

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.pipeline import (
    train_and_evaluate_with_pipeline,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)
from tamp_improv.benchmarks.gridworld import (
    GridworldTAMPSystem,
)
from tamp_improv.benchmarks.gridworld_fixed import (
    GridworldFixedTAMPSystem,
)

SYSTEM_CLASSES: dict[str, Type[ImprovisationalTAMPSystem[Any, Any]]] = {
    "GraphObstacle2DTAMPSystem": GraphObstacle2DTAMPSystem,
    "GraphObstacleTowerTAMPSystem": GraphObstacleTowerTAMPSystem,
    "ClutteredDrawerTAMPSystem": ClutteredDrawerTAMPSystem,
    "CleanupTableTAMPSystem": CleanupTableTAMPSystem,
    "GridworldTAMPSystem": GridworldTAMPSystem,  # Reuse for gridworld
    "GridworldFixedTAMPSystem": GridworldFixedTAMPSystem
}


@hydra.main(version_base=None, config_path="configs", config_name="gridworld_fixed")
def main(cfg: DictConfig) -> float:
    """Main training function using modular pipeline."""
    print("=" * 80)
    print(f"Training SLAP with Modular Pipeline on {cfg.env_name}")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Create system
    system_cls = SYSTEM_CLASSES[cfg.env_name]
    system_kwargs = {
        "seed": cfg.seed,
        "render_mode": cfg.render_mode if cfg.render_mode != "null" else None,
    }
    if hasattr(cfg, "n_blocks"):
        system_kwargs["n_blocks"] = cfg.n_blocks
    if hasattr(cfg, "num_obstacle_blocks"):
        system_kwargs["num_obstacle_blocks"] = cfg.num_obstacle_blocks
    if hasattr(cfg, "num_cells"):
        system_kwargs["num_cells"] = cfg.num_cells
    if hasattr(cfg, "num_states_per_cell"):
        system_kwargs["num_states_per_cell"] = cfg.num_states_per_cell
    if hasattr(cfg, "num_teleporters"):
        system_kwargs["num_teleporters"] = cfg.num_teleporters
    print(f"\nCreating system: {cfg.env_name} with kwargs: {system_kwargs}")
    system = system_cls.create_default(**system_kwargs)  # type: ignore[attr-defined]

    # Setup device and RL config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rl_config = RLConfig(
        learning_rate=cfg.rl_learning_rate,
        batch_size=cfg.rl_batch_size,
        n_epochs=cfg.rl_n_epochs,
        gamma=cfg.rl_gamma,
        ent_coef=cfg.rl_ent_coef,
        deterministic=cfg.deterministic,
        device=device,
    )
    print(f"\nUsing device: {device}")
    print(f"Pruning method: {cfg.get('pruning_method', 'none')}")

    # Get Hydra output directory (always needed for results.txt)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Get pipeline cache directory from config or use default
    if hasattr(cfg, 'pipeline_cache_dir') and cfg.pipeline_cache_dir:
        save_dir = Path(cfg.pipeline_cache_dir)
        print(f"\nPipeline cache directory: {save_dir}")
        print("  (Shared across runs - artifacts will be cached and reused)")
    else:
        # Fallback: use Hydra output directory (no sharing between runs)
        save_dir = output_dir / "pipeline_cache"
        print(f"\nPipeline cache directory: {save_dir}")
        print("  (Run-specific - no caching across runs)")

    # Create cache directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to dict for pipeline
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config_dict, dict), "Config should be a dictionary"

    # Run modular pipeline
    metrics = train_and_evaluate_with_pipeline(
        system=system,
        policy_factory=lambda seed: MultiRLPolicy(seed=seed, config=rl_config),
        config=config_dict,
        save_dir=save_dir,
        policy_name="MultiRL",
        num_eval_episodes=cfg.eval_rollouts,
    )

    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.1f}s")
    print(f"Total Time: {metrics.total_time:.1f}s")
    print("=" * 80)

    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"env_name: {cfg.env_name}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"pruning_method: {cfg.get('pruning_method', 'none')}\n")
        f.write(f"success_rate: {metrics.success_rate}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length}\n")
        f.write(f"avg_reward: {metrics.avg_reward}\n")
        f.write(f"training_time: {metrics.training_time}\n")
        f.write(f"total_time: {metrics.total_time}\n")

    return metrics.success_rate


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
