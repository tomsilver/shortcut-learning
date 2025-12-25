"""Training and evaluation script for distance heuristic with Custom DQN.

This script is optimized for cluster execution and uses the custom DQN
implementation instead of stable_baselines3 for full transparency.

Usage:
    python heuristic_train_eval_custom_dqn.py env_name=GridworldTAMPSystem
"""

from pathlib import Path
from typing import Any, Type

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.collection import collect_total_shortcuts
from tamp_improv.approaches.improvisational.distance_heuristic import (
    DistanceHeuristicConfig,
    GoalConditionedDistanceHeuristic,
)
from tamp_improv.approaches.improvisational.graph_training import compute_graph_distances
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem
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
}


@hydra.main(version_base=None, config_path="configs", config_name="obstacle2d")
def main(cfg: DictConfig) -> float:
    """Main function for custom DQN heuristic training and evaluation."""
    import sys
    import time

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80, flush=True)
    print("Distance Heuristic Training with CUSTOM DQN", flush=True)
    print("=" * 80, flush=True)
    print("\nConfiguration:", flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    print("=" * 80, flush=True)

    # =========================================================================
    # Setup
    # =========================================================================
    print("\n[SETUP] Creating system...", flush=True)
    system_cls = SYSTEM_CLASSES[cfg.env_name]
    system_kwargs = {
        "seed": cfg.seed,
        "render_mode": cfg.render_mode if cfg.render_mode != "null" else None,
    }

    # Add system-specific parameters
    if hasattr(cfg, "num_obstacle_blocks"):
        system_kwargs["num_obstacle_blocks"] = cfg.num_obstacle_blocks
    if hasattr(cfg, "num_cells"):
        system_kwargs["num_cells"] = cfg.num_cells
    if hasattr(cfg, "num_states_per_cell"):
        system_kwargs["num_states_per_cell"] = cfg.num_states_per_cell
    if hasattr(cfg, "num_teleporters"):
        system_kwargs["num_teleporters"] = cfg.num_teleporters
    if hasattr(cfg, "n_blocks"):
        system_kwargs["n_blocks"] = cfg.n_blocks

    system = system_cls.create_default(**system_kwargs)  # type: ignore[attr-defined]
    print(f"[SETUP] System created: {system.name}", flush=True)

    # Create approach
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SETUP] Using device: {device}", flush=True)

    rl_config = RLConfig(device=device)
    policy = MultiRLPolicy(seed=cfg.seed, config=rl_config)
    approach = ImprovisationalTAMPApproach(system, policy, seed=cfg.seed)
    approach.training_mode = True

    # =========================================================================
    # Step 1: Collect Training Data
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 1: Collecting Training Data", flush=True)
    print("=" * 80, flush=True)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config_dict, dict)

    # Override collect_episodes if heuristic-specific value provided
    config_dict["collect_episodes"] = cfg.get("heuristic_collect_episodes", cfg.collect_episodes)

    print(f"Collecting from {config_dict['collect_episodes']} episode(s)...", flush=True)
    collection_start = time.time()

    rng = np.random.default_rng(cfg.seed)
    training_data = collect_total_shortcuts(system, approach, config_dict, rng=rng)

    collection_time = time.time() - collection_start
    print(f"\n✓ Collection complete in {collection_time:.1f}s", flush=True)
    print(f"  Nodes with states: {len(training_data.node_states)}", flush=True)
    print(f"  Total shortcuts: {len(training_data.valid_shortcuts)}", flush=True)
    print(f"  Planning graph: {len(training_data.graph.nodes)} nodes, {len(training_data.graph.edges)} edges", flush=True)

    # =========================================================================
    # Step 2: Prepare State Pairs
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: Preparing State Pairs", flush=True)
    print("=" * 80, flush=True)

    planning_graph = training_data.graph
    all_node_states = training_data.node_states

    # Collect state pairs
    state_pairs = []
    for source_node in planning_graph.nodes:
        for target_node in planning_graph.nodes:
            if source_node.id == target_node.id:
                continue

            if source_node.id not in all_node_states or target_node.id not in all_node_states:
                continue

            source_states = all_node_states[source_node.id]
            target_states = all_node_states[target_node.id]

            if not source_states or not target_states:
                continue

            # Take first state from each node
            state_pairs.append((source_states[0], target_states[0]))

    print(f"✓ Prepared {len(state_pairs)} state pairs", flush=True)

    # =========================================================================
    # Step 3: Train Distance Heuristic with Custom DQN
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 3: Training Distance Heuristic with Custom DQN", flush=True)
    print("=" * 80, flush=True)

    # Get custom DQN parameters from config
    training_steps = cfg.get("heuristic_training_steps", 50000)
    max_episode_steps = cfg.get("heuristic_max_steps", 100)
    learning_starts = cfg.get("heuristic_learning_starts", 1000)
    log_freq = cfg.get("heuristic_log_freq", 1000)
    eval_freq = cfg.get("heuristic_eval_freq", 5000)

    # Create custom DQN config
    heuristic_config = DistanceHeuristicConfig(
        use_custom_dqn=True,  # Enable custom DQN
        learning_rate=cfg.get("heuristic_learning_rate", 3e-4),
        batch_size=cfg.get("heuristic_batch_size", 256),
        buffer_size=cfg.get("heuristic_buffer_size", 100000),
        max_episode_steps=max_episode_steps,
        learning_starts=learning_starts,
        device=device,
        # Custom DQN specific
        custom_dqn_hidden_sizes=cfg.get("custom_dqn_hidden_sizes", [256, 256]),
        custom_dqn_target_update_freq=cfg.get("custom_dqn_target_update_freq", 1000),
        custom_dqn_epsilon_decay_steps=cfg.get("custom_dqn_epsilon_decay_steps", 10000),
        custom_dqn_her_k=cfg.get("custom_dqn_her_k", 4),
        custom_dqn_log_freq=log_freq,
        custom_dqn_eval_freq=eval_freq,
    )

    print(f"Training configuration:", flush=True)
    print(f"  Training steps: {training_steps}", flush=True)
    print(f"  Max episode steps: {max_episode_steps}", flush=True)
    print(f"  Learning starts: {learning_starts}", flush=True)
    print(f"  Log frequency: {log_freq}", flush=True)
    print(f"  Eval frequency: {eval_freq}", flush=True)
    print(f"  Hidden sizes: {heuristic_config.custom_dqn_hidden_sizes}", flush=True)
    print(f"  HER k: {heuristic_config.custom_dqn_her_k}", flush=True)

    training_start = time.time()

    # Create and train heuristic
    heuristic = GoalConditionedDistanceHeuristic(config=heuristic_config, seed=cfg.seed)
    heuristic.train(
        env=system.env,
        state_pairs=state_pairs,
        perceiver=system.perceiver,
        max_training_steps=training_steps,
    )

    training_time = time.time() - training_start
    print(f"\n✓ Training complete in {training_time:.1f}s ({training_time/60:.1f} minutes)", flush=True)

    # Save heuristic
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    heuristic_path = output_dir / "distance_heuristic"
    heuristic.save(str(heuristic_path))
    print(f"✓ Saved heuristic to {heuristic_path}", flush=True)

    # =========================================================================
    # Step 4: Evaluate Distance Heuristic
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 4: Evaluating Distance Heuristic", flush=True)
    print("=" * 80, flush=True)

    # Compute graph distances
    print("Computing graph distances...", flush=True)
    graph_distances = compute_graph_distances(planning_graph, exclude_shortcuts=True)
    print(f"✓ Computed {len(graph_distances)} pairwise distances", flush=True)

    # Evaluate on all state pairs
    from tamp_improv.approaches.improvisational.analyze import compute_true_distance

    print(f"\nEvaluating {len(state_pairs)} state pairs...", flush=True)
    eval_start = time.time()

    results = []
    for i, (source_state, target_state) in enumerate(state_pairs):
        # Get learned distance
        learned_dist = heuristic.estimate_distance(source_state, target_state)

        # Get true distance
        target_atoms = system.perceiver.step(target_state)
        true_dist = compute_true_distance(system, source_state, target_atoms)

        # Get graph distance
        source_atoms = system.perceiver.step(source_state)
        source_node_id = None
        target_node_id = None

        for node in planning_graph.nodes:
            if node.atoms == source_atoms:
                source_node_id = node.id
            if node.atoms == target_atoms:
                target_node_id = node.id

        if source_node_id is not None and target_node_id is not None:
            graph_dist = graph_distances.get((source_node_id, target_node_id), float('inf'))
        else:
            graph_dist = float('inf')

        results.append({
            'source_idx': i,
            'learned_distance': learned_dist,
            'true_distance': true_dist,
            'graph_distance': graph_dist,
        })

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(state_pairs)} pairs...", flush=True)

    eval_time = time.time() - eval_start
    print(f"✓ Evaluation complete in {eval_time:.1f}s", flush=True)

    # =========================================================================
    # Step 5: Analyze Results
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("STEP 5: Results Analysis", flush=True)
    print("=" * 80, flush=True)

    # Filter results
    finite_results = [r for r in results if r['graph_distance'] != float('inf')]
    infinite_results = [r for r in results if r['graph_distance'] == float('inf')]

    print(f"\nResults Summary:", flush=True)
    print(f"  Total pairs: {len(results)}", flush=True)
    print(f"  Finite distance pairs: {len(finite_results)}", flush=True)
    print(f"  Infinite distance pairs: {len(infinite_results)}", flush=True)

    correlation_true = 0.0

    if finite_results:
        true_dists = np.array([r['true_distance'] for r in finite_results])
        learned_dists = np.array([r['learned_distance'] for r in finite_results])
        graph_dists = np.array([r['graph_distance'] for r in finite_results])

        # Statistics vs True distance
        mae_true = np.mean(np.abs(true_dists - learned_dists))
        rmse_true = np.sqrt(np.mean((true_dists - learned_dists) ** 2))
        correlation_true = np.corrcoef(true_dists, learned_dists)[0, 1]

        # Statistics vs Graph distance
        mae_graph = np.mean(np.abs(graph_dists - learned_dists))
        correlation_graph = np.corrcoef(graph_dists, learned_dists)[0, 1]

        print(f"\nStatistics (vs True Distance):", flush=True)
        print(f"  MAE: {mae_true:.2f}", flush=True)
        print(f"  RMSE: {rmse_true:.2f}", flush=True)
        print(f"  Correlation: {correlation_true:.3f}", flush=True)

        print(f"\nStatistics (vs Graph Distance):", flush=True)
        print(f"  MAE: {mae_graph:.2f}", flush=True)
        print(f"  Correlation: {correlation_graph:.3f}", flush=True)

        print(f"\nDistance Ranges:", flush=True)
        print(f"  True:    [{true_dists.min():.1f}, {true_dists.max():.1f}]", flush=True)
        print(f"  Graph:   [{graph_dists.min():.1f}, {graph_dists.max():.1f}]", flush=True)
        print(f"  Learned: [{learned_dists.min():.1f}, {learned_dists.max():.1f}]", flush=True)

        # Show sample comparisons
        print("\nSample Comparisons (sorted by true distance):", flush=True)
        print(f"{'Idx':>4} | {'True':>6} | {'Graph':>6} | {'Learned':>8} | {'Error':>6}", flush=True)
        print("-" * 45, flush=True)

        sorted_results = sorted(finite_results, key=lambda r: r['true_distance'])
        for r in sorted_results[:20]:
            error = abs(r['true_distance'] - r['learned_distance'])
            print(
                f"{r['source_idx']:>4} | "
                f"{r['true_distance']:>6.1f} | "
                f"{r['graph_distance']:>6.1f} | "
                f"{r['learned_distance']:>8.1f} | "
                f"{error:>6.1f}",
                flush=True
            )
    else:
        print("\nNo finite distance pairs to analyze!", flush=True)

    # Save results
    results_file = output_dir / "custom_dqn_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Custom DQN Distance Heuristic Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Environment: {cfg.env_name}\n")
        f.write(f"Seed: {cfg.seed}\n")
        f.write(f"Training steps: {training_steps}\n")
        f.write(f"Training time: {training_time:.1f}s\n")
        f.write(f"Total pairs: {len(results)}\n")
        f.write(f"Finite pairs: {len(finite_results)}\n")
        f.write(f"Infinite pairs: {len(infinite_results)}\n\n")

        if finite_results:
            f.write(f"Correlation with true distance: {correlation_true:.3f}\n")
            f.write(f"MAE: {mae_true:.2f}\n")
            f.write(f"RMSE: {rmse_true:.2f}\n")

    print(f"\n✓ Results saved to {results_file}", flush=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Collection time: {collection_time:.1f}s", flush=True)
    print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)", flush=True)
    print(f"Evaluation time: {eval_time:.1f}s", flush=True)
    print(f"Total time: {collection_time + training_time + eval_time:.1f}s", flush=True)
    print(f"\nFinal Correlation (vs True Distance): {correlation_true:.3f}", flush=True)
    print("=" * 80, flush=True)

    return float(correlation_true)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
