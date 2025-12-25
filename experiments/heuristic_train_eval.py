"""Training and evaluation script for distance heuristic.

This script validates the distance heuristic by:
1. Collecting training data (building planning graphs)
2. Training the heuristic on a subset of state pairs
3. Evaluating the heuristic on all state pairs
4. Comparing f(s, s') (learned) vs D(s, s') (graph distances)
"""

from pathlib import Path
from typing import Any, Type

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.collection import (
    collect_total_shortcuts,
)
from tamp_improv.approaches.improvisational.graph_training import (
    compute_graph_distances,
)
from tamp_improv.approaches.improvisational.pruning import (
    train_distance_heuristic,
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
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem

SYSTEM_CLASSES: dict[str, Type[ImprovisationalTAMPSystem[Any, Any]]] = {
    "GraphObstacle2DTAMPSystem": GraphObstacle2DTAMPSystem,
    "GraphObstacleTowerTAMPSystem": GraphObstacleTowerTAMPSystem,
    "ClutteredDrawerTAMPSystem": ClutteredDrawerTAMPSystem,
    "CleanupTableTAMPSystem": CleanupTableTAMPSystem,
    "GridworldTAMPSystem": GridworldTAMPSystem,
}


@hydra.main(version_base=None, config_path="configs", config_name="obstacle2d")
def main(cfg: DictConfig) -> float:
    """Main function for heuristic training and evaluation."""
    import sys

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80, flush=True)
    print("Distance Heuristic Training and Evaluation", flush=True)
    print("=" * 80, flush=True)
    print("\nConfiguration:", flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    print("=" * 80, flush=True)

    print("[DEBUG] Creating system...", flush=True)
    # Create system
    system_cls = SYSTEM_CLASSES[cfg.env_name]
    system_kwargs = {
        "seed": cfg.seed,
        "render_mode": cfg.render_mode if cfg.render_mode != "null" else None,
    }

    # Add system-specific parameters from config
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
    print("[DEBUG] System created", flush=True)

    # Create approach for graph building
    print("[DEBUG] Creating approach...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rl_config = RLConfig(device=device)
    policy = MultiRLPolicy(seed=cfg.seed, config=rl_config)
    approach = ImprovisationalTAMPApproach(system, policy, seed=cfg.seed)
    approach.training_mode = True

    print(f"\nUsing device: {device}", flush=True)

    # Add memory tracking
    import psutil
    import os
    process = psutil.Process(os.getpid())

    def print_memory():
        mem_mb = process.memory_info().rss / 1024 / 1024
        msg = f"[MEMORY] Current usage: {mem_mb:.1f} MB"
        print(msg, flush=True)
        return mem_mb

    print_memory()

    # =========================================================================
    # Step 1: Collect training data (builds graphs and collects states)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Step 1: Collecting all shortcuts (no pruning)", flush=True)
    print("=" * 80, flush=True)

    # Use new collection function - much simpler and cleaner
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config_dict, dict)

    # Override collect_episodes with heuristic-specific value (use very small for testing)
    config_dict["collect_episodes"] = cfg.get("heuristic_collect_episodes", 1)

    print(f"[DEBUG] Collecting shortcuts from {config_dict['collect_episodes']} episode(s)", flush=True)
    print_memory()

    print("[DEBUG] About to call collect_total_shortcuts...", flush=True)
    # Collect ALL shortcuts without any pruning
    rng = np.random.default_rng(cfg.seed)
    training_data = collect_total_shortcuts(system, approach, config_dict, rng=rng)

    print(f"[DEBUG] Collection complete", flush=True)
    print_memory()

    # Extract data from training_data
    all_node_states = training_data.node_states  # node_id -> list of states
    planning_graph = training_data.graph
    assert planning_graph is not None

    print(f"\nCollected states for {len(all_node_states)} nodes")
    print(f"Planning graph has {len(planning_graph.nodes)} nodes, {len(planning_graph.edges)} edges")
    print(f"Total shortcuts: {len(training_data.valid_shortcuts)}")
    print_memory()

    # =========================================================================
    # Step 2: Compute graph distances D(s, s') for all pairs
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Computing graph distances D(s, s')")
    print("=" * 80)

    print("[DEBUG] Computing graph distances...")
    print_memory()
    graph_distances = compute_graph_distances(planning_graph, exclude_shortcuts=True)
    print(f"  Computed {len(graph_distances)} pairwise distances")
    print_memory()

    # =========================================================================
    # Step 3: Prepare state pairs for evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Preparing state pairs for evaluation")
    print("=" * 80)

    print("[DEBUG] Collecting state pairs...")
    print_memory()

    # Collect all possible state pairs (all combinations of source and target states)
    all_pairs = []
    node_pair_to_state_pairs = {}  # Track which state pairs belong to each node pair

    for source_node in planning_graph.nodes:
        for target_node in planning_graph.nodes:
            if source_node.id == target_node.id:
                continue

            # Check if we have states for both nodes
            if source_node.id not in all_node_states or target_node.id not in all_node_states:
                continue

            source_states = all_node_states[source_node.id]
            target_states = all_node_states[target_node.id]

            if not source_states or not target_states:
                continue

            # Get graph distance (same for all state pairs from this node pair)
            graph_dist = graph_distances.get(
                (source_node.id, target_node.id), float("inf")
            )

            # Create all combinations of source and target states
            state_pair_indices = []
            for source_state in source_states:
                for target_state in target_states:
                    state_pair_idx = len(all_pairs)
                    all_pairs.append(
                        {
                            "source_id": source_node.id,
                            "target_id": target_node.id,
                            "source_state": source_state,
                            "target_state": target_state,
                            "graph_distance": graph_dist,
                        }
                    )
                    state_pair_indices.append(state_pair_idx)

            node_pair_to_state_pairs[(source_node.id, target_node.id)] = state_pair_indices

    num_node_pairs = len(node_pair_to_state_pairs)
    avg_state_pairs = len(all_pairs) / max(1, num_node_pairs)
    print(f"\nCollected {len(all_pairs)} total state pairs from {num_node_pairs} node pairs")
    print(f"Average {avg_state_pairs:.1f} state pairs per node pair")
    print_memory()

    # =========================================================================
    # Step 4: Train distance heuristic
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Training distance heuristic")
    print("=" * 80)

    print("[DEBUG] Training heuristic using train_distance_heuristic()...")
    print_memory()

    # Track training time
    import time
    training_start_time = time.time()

    # Use the new train_distance_heuristic function from pruning.py
    heuristic = train_distance_heuristic(training_data, system, config_dict, rng)

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time

    print(f"[DEBUG] Training complete in {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
    print_memory()

    # Save heuristic
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    heuristic_path = output_dir / "distance_heuristic"
    heuristic.save(str(heuristic_path))
    print(f"\nSaved distance heuristic to {heuristic_path}")

    # =========================================================================
    # Step 5: Evaluate on all pairs and compute averaged distances per node pair
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Evaluating distance heuristic on all state pairs")
    print("=" * 80)

    print(f"Evaluating {len(all_pairs)} state pairs...")

    # Import true distance function
    from tamp_improv.approaches.improvisational.analyze import compute_true_distance

    # Compute learned and true distances for all state pairs
    all_learned_distances = []
    all_true_distances = []
    for pair in all_pairs:
        learned_dist = heuristic.estimate_distance(
            pair["source_state"], pair["target_state"]
        )
        all_learned_distances.append(learned_dist)

        # Compute true distance from source state to target node atoms
        target_node = planning_graph.nodes[pair["target_id"]]
        true_dist = compute_true_distance(
            system, pair["source_state"], target_node.atoms
        )
        all_true_distances.append(true_dist)

    # Compute average learned and true distances for each node pair
    node_pair_learned_distances = {}  # (source_id, target_id) -> avg learned distance
    node_pair_true_distances = {}  # (source_id, target_id) -> avg true distance

    for node_pair, state_pair_indices in node_pair_to_state_pairs.items():
        learned_distances = [all_learned_distances[idx] for idx in state_pair_indices]
        true_distances = [all_true_distances[idx] for idx in state_pair_indices]

        avg_learned_dist = sum(learned_distances) / len(learned_distances)
        avg_true_dist = sum(true_distances) / len(true_distances)

        node_pair_learned_distances[node_pair] = avg_learned_dist
        node_pair_true_distances[node_pair] = avg_true_dist

    print(f"Computed average distances for {len(node_pair_learned_distances)} node pairs")

    # Create results based on averaged node pair distances
    results = []
    for node_pair, avg_learned_dist in node_pair_learned_distances.items():
        source_id, target_id = node_pair
        graph_dist = graph_distances.get(node_pair, float("inf"))
        true_dist = node_pair_true_distances[node_pair]

        results.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "graph_distance": graph_dist,
                "true_distance": true_dist,
                "learned_distance": avg_learned_dist,
            }
        )

    # =========================================================================
    # Step 6: Analyze and report results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Analysis of Results")
    print("=" * 80)

    print(f"\nTotal node pairs: {len(results)}")
    print(f"Total state-node pairs evaluated: {len(all_pairs)}")

    # First, analyze RAW state-node pair distances (no averaging)
    print("\n" + "=" * 80)
    print("STATE-NODE PAIR ANALYSIS (No Averaging)")
    print("=" * 80)
    print("Comparing learned distances f(s,g) to true optimal distances D_true(s,g)")
    print("where s is a low-level state and g is a high-level node (set of atoms)")

    true_state_dists = np.array(all_true_distances)
    learned_state_dists = np.array(all_learned_distances)

    # Filter out any infinite distances
    finite_mask = np.isfinite(true_state_dists)
    true_state_finite = true_state_dists[finite_mask]
    learned_state_finite = learned_state_dists[finite_mask]

    if len(true_state_finite) > 0:
        mae_state = np.mean(np.abs(true_state_finite - learned_state_finite))
        rmse_state = np.sqrt(np.mean((true_state_finite - learned_state_finite) ** 2))
        correlation_state = np.corrcoef(true_state_finite, learned_state_finite)[0, 1]
        relative_errors_state = np.abs(true_state_finite - learned_state_finite) / (true_state_finite + 1e-6)
        mean_relative_error_state = np.mean(relative_errors_state)

        print(f"\nState-Node Pair Statistics:")
        print(f"  Valid pairs: {len(true_state_finite)}")
        print(f"  Mean Absolute Error (MAE): {mae_state:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse_state:.2f}")
        print(f"  Correlation: {correlation_state:.3f}")
        print(f"  Mean Relative Error: {mean_relative_error_state:.2%}")
        print(f"  True distance range: [{true_state_finite.min():.1f}, {true_state_finite.max():.1f}]")
        print(f"  Learned distance range: [{learned_state_finite.min():.1f}, {learned_state_finite.max():.1f}]")

    print("\n" + "=" * 80)
    print("NODE-NODE PAIR ANALYSIS (Averaged)")
    print("=" * 80)
    print("Comparing averaged learned distances to graph distances D(n,n')")

    # Compute statistics
    def analyze_results(result_list, name):
        if not result_list:
            return

        # Filter out infinite distances
        valid_results = [r for r in result_list if r["graph_distance"] != float("inf")]

        if not valid_results:
            print(f"\n{name}: No valid pairs (all unreachable)")
            return

        graph_dists = np.array([r["graph_distance"] for r in valid_results])
        true_dists = np.array([r["true_distance"] for r in valid_results])
        learned_dists = np.array([r["learned_distance"] for r in valid_results])

        # Compute metrics vs TRUE distance (ground truth)
        mae_true = np.mean(np.abs(true_dists - learned_dists))
        rmse_true = np.sqrt(np.mean((true_dists - learned_dists) ** 2))
        correlation_true = np.corrcoef(true_dists, learned_dists)[0, 1]
        relative_errors_true = np.abs(true_dists - learned_dists) / (true_dists + 1e-6)
        mean_relative_error_true = np.mean(relative_errors_true)

        # Compute metrics vs GRAPH distance (approximation)
        mae_graph = np.mean(np.abs(graph_dists - learned_dists))
        rmse_graph = np.sqrt(np.mean((graph_dists - learned_dists) ** 2))
        correlation_graph = np.corrcoef(graph_dists, learned_dists)[0, 1]

        print(f"\n{name}:")
        print(f"  Valid pairs: {len(valid_results)}")
        print(f"\n  vs TRUE distance (D_true):")
        print(f"    Mean Absolute Error (MAE): {mae_true:.2f}")
        print(f"    Root Mean Squared Error (RMSE): {rmse_true:.2f}")
        print(f"    Correlation: {correlation_true:.3f}")
        print(f"    Mean Relative Error: {mean_relative_error_true:.2%}")
        print(f"\n  vs GRAPH distance (D - approximation):")
        print(f"    Mean Absolute Error (MAE): {mae_graph:.2f}")
        print(f"    Correlation: {correlation_graph:.3f}")
        print(f"\n  Distance ranges:")
        print(f"    True distance:    [{true_dists.min():.1f}, {true_dists.max():.1f}]")
        print(f"    Graph distance:   [{graph_dists.min():.1f}, {graph_dists.max():.1f}]")
        print(f"    Learned distance: [{learned_dists.min():.1f}, {learned_dists.max():.1f}]")

    analyze_results(results, "All Node Pairs")

    # Print detailed comparison for a sample of pairs
    print("\n" + "=" * 80)
    print("Sample Comparisons (True vs Learned vs Graph Distance)")
    print("=" * 80)
    print("Note: All distances are averaged across all state pairs for each node pair")
    print("D_true = True optimal distance, D = Graph distance (approx), f = Learned distance")

    # Separate finite and infinite distance pairs
    finite_results = [r for r in results if r["graph_distance"] != float("inf")]
    infinite_results = [r for r in results if r["graph_distance"] == float("inf")]

    # Sort finite by true distance
    sorted_finite = sorted(finite_results, key=lambda r: r["true_distance"])
    sorted_infinite = sorted(infinite_results, key=lambda r: r["learned_distance"])

    # Show finite distance pairs
    if sorted_finite:
        print(f"\nPairs with existing non-shortcut path:")
        print(f"{'Source':>6} -> {'Target':>6} | {'D_true':>10} | {'D(graph)':>10} | {'f(learned)':>10} | {'Error':>8}")
        print("-" * 80)

        sample_size = min(20, len(sorted_finite))
        sample_step = max(1, len(sorted_finite) // sample_size)

        for i in range(0, len(sorted_finite), sample_step):
            r = sorted_finite[i]
            error = abs(r["true_distance"] - r["learned_distance"])

            print(
                f"{r['source_id']:>6} -> {r['target_id']:>6} | "
                f"{r['true_distance']:>10.1f} | "
                f"{r['graph_distance']:>10.1f} | "
                f"{r['learned_distance']:>10.1f} | "
                f"{error:>8.1f}"
            )

    # Show infinite distance pairs (no existing non-shortcut path)
    if sorted_infinite:
        print(f"\n{'=' * 80}")
        print(f"Pairs with NO existing non-shortcut path (D=∞):")
        print(f"These are the most valuable shortcuts to learn!")
        print(f"{'=' * 80}")
        f_label_inf = "f(s,s')"
        print(f"{'Source':>6} -> {'Target':>6} | {f_label_inf:>10}")
        print("-" * 30)

        for r in sorted_infinite:
            print(
                f"{r['source_id']:>6} -> {r['target_id']:>6} | "
                f"{r['learned_distance']:>10.1f}"
            )

    # Save results
    results_file = output_dir / "heuristic_eval_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Distance Heuristic Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Environment: {cfg.env_name}\n")
        f.write(f"Seed: {cfg.seed}\n")
        f.write(f"Total node pairs: {len(results)}\n")
        f.write(f"Total state pairs evaluated: {len(all_pairs)}\n")
        f.write(f"Average state pairs per node pair: {len(all_pairs) / max(1, len(results)):.1f}\n")
        f.write(f"Pairs with finite graph distance: {len(finite_results)}\n")
        f.write(f"Pairs with infinite graph distance (no non-shortcut path): {len(infinite_results)}\n")
        f.write(f"\nNote: f(s,s') values are averaged across all state pairs for each node pair\n\n")

        # Write finite distance comparisons
        d_label = "D(s,s')"
        f_label = "f(s,s')"
        f.write("Pairs with existing non-shortcut path:\n")
        f.write(
            f"{'Source':>6} -> {'Target':>6} | "
            f"{d_label:>10} | {f_label:>10} | {'Error':>8}\n"
        )
        f.write("-" * 60 + "\n")

        for r in sorted_finite:
            error = abs(r["graph_distance"] - r["learned_distance"])

            f.write(
                f"{r['source_id']:>6} -> {r['target_id']:>6} | "
                f"{r['graph_distance']:>10.1f} | "
                f"{r['learned_distance']:>10.1f} | "
                f"{error:>8.1f}\n"
            )

        # Write infinite distance comparisons
        if sorted_infinite:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Pairs with NO existing non-shortcut path (D=∞):\n")
            f.write("These are the most valuable shortcuts to learn!\n")
            f.write("=" * 80 + "\n")
            f_label_inf = "f(s,s')"
            f.write(
                f"{'Source':>6} -> {'Target':>6} | {f_label_inf:>10}\n"
            )
            f.write("-" * 30 + "\n")

            for r in sorted_infinite:
                f.write(
                    f"{r['source_id']:>6} -> {r['target_id']:>6} | "
                    f"{r['learned_distance']:>10.1f}\n"
                )

    print(f"\nDetailed results saved to {results_file}")

    # Return STATE-LEVEL correlation as the main metric (more accurate)
    print("\n" + "=" * 80)
    if len(true_state_finite) > 0:
        print(f"Final Correlation (State-Node pairs vs True Distance): {correlation_state:.3f}")
    else:
        print(f"Final Correlation (State-Node pairs vs True Distance): N/A (no finite pairs)")
        correlation_state = 0.0
    print("=" * 80)

    return float(correlation_state)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
