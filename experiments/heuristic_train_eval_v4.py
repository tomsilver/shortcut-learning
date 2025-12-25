"""Training and evaluation script for distance heuristic V4.

This script performs the complete pipeline mirroring heuristic_train_v4.ipynb:
1. Initialize environment (GridworldFixed)
2. Collect training data using collect_total_shortcuts
3. Extract state-node pairs from training data
4. Train the distance heuristic V4
5. Evaluate policy with rollouts
6. Compare estimated vs true node distances
7. Save the trained heuristic
"""

from pathlib import Path
from typing import Any, Type

import hydra
import numpy as np
import torch
from gymnasium.spaces import GraphInstance
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.analyze import (
    compute_true_node_distance, 
    compute_graph_node_distance,
    compute_true_distance
)
from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.collection import (
    collect_total_shortcuts,
)
from tamp_improv.approaches.improvisational.distance_heuristic_v4 import (
    DistanceHeuristicV4,
    DistanceHeuristicV4Config,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedTAMPSystem
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
    "GridworldFixedTAMPSystem": GridworldFixedTAMPSystem,
}


@hydra.main(version_base=None, config_path="configs", config_name="gridworld_fixed")
def main(cfg: DictConfig) -> float:
    """Main function for heuristic training and evaluation."""
    import sys

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80, flush=True)
    print("Distance Heuristic V4 Training and Evaluation", flush=True)
    print("=" * 80, flush=True)
    print("\nConfiguration:", flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    print("=" * 80, flush=True)

    print("[DEBUG] Creating system...", flush=True)
    # Create system
    print(cfg.env_name)
    system_cls = SYSTEM_CLASSES[cfg.env_name]
    print(system_cls)
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

    output_dir = Path(HydraConfig.get().runtime.output_dir)


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
    # Step 2: (Skipped - no need for graph distances in V4)
    # =========================================================================

    # =========================================================================
    # Step 3: Extract State-Node Pairs
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Extracting state-node pairs")
    print("=" * 80)

    print("[DEBUG] Creating atoms_to_node mapping...")
    # Create atoms_to_node mapping
    atoms_to_node = {}
    for node in planning_graph.nodes:
        atoms_to_node[node.atoms] = node.id

    print("[DEBUG] Extracting state-node pairs...")
    print_memory()

    print("Atoms to node:", atoms_to_node)

    # Extract state-node pairs (state from node A, target node B)
    state_node_pairs = []
    for node_id, states in all_node_states.items():
        # Find node by atoms

        for state in states:

            # Add pairs to all other nodes
            for target_node in planning_graph.nodes:
                if target_node.id != node_id:
                    state_node_pairs.append((state, target_node.id))

    print(f"\nCreated {len(state_node_pairs)} state-node training pairs")
    print_memory()

    # =========================================================================
    # Step 4: Train Distance Heuristic V4
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Training distance heuristic V4")
    print("=" * 80)

    print("[DEBUG] Creating heuristic config...")
    # Create heuristic config
    heuristic_config = DistanceHeuristicV4Config(
        hidden_dims=cfg['hidden_dims'],
        latent_dim=cfg['latent_dim'],
        normalize_embeddings=cfg.get('normalize_embeddings', True),
        learning_rate=cfg['learning_rate'],
        batch_size=cfg['batch_size'],
        buffer_size=cfg['buffer_size'],
        gamma=cfg['gamma'],
        repetition_factor=cfg['repetition_factor'],
        policy_temperature=cfg['policy_temperature'],
        eval_temperature=cfg['eval_temperature'],
        iters_per_epoch=cfg['iters_per_epoch'],
        learn_frequency=cfg['learn_frequency'],
        num_rounds=cfg.get('num_rounds', 1),
        keep_fraction=cfg.get('keep_fraction', 0.5),
        device=device,
    )

    print(f"[DEBUG] Initializing heuristic (k={heuristic_config.latent_dim})...")
    # Initialize heuristic
    heuristic = DistanceHeuristicV4(config=heuristic_config, 
                                    env=system.env,
                                    perceiver=system.perceiver,
                                    atoms_to_node=atoms_to_node,
                                    node_to_states=training_data.node_states,
                                    seed=cfg.seed)

    # Track training time
    import time
    training_start_time = time.time()

    print("[DEBUG] Training heuristic...")
    print_memory()

    # Compute graph distances if using multi-round training (num_rounds > 1)
    graph_distances = None
    if heuristic_config.num_rounds > 1:
        print(f"\n[DEBUG] Computing graph distances for multi-round training...")
        print(f"  num_rounds={heuristic_config.num_rounds}, keep_fraction={heuristic_config.keep_fraction}")

        # Get unique node pairs from state-node pairs
        unique_node_pairs = set()
        for state, target_node in state_node_pairs:
            source_node = heuristic.get_node(state)
            unique_node_pairs.add((source_node, target_node))

        print(f"  Computing distances for {len(unique_node_pairs)} unique node-node pairs...")

        # Compute graph distance for each unique node pair
        graph_distances = {}
        for i, (source_node, target_node) in enumerate(unique_node_pairs):
            if i % 100 == 0:
                print(f"    Progress: {i}/{len(unique_node_pairs)}")

            graph_dist = compute_graph_node_distance(
                planning_graph,
                start_node_atoms=training_data.node_atoms[source_node],
                goal_node_atoms=training_data.node_atoms[target_node]
            )
            graph_distances[(source_node, target_node)] = graph_dist if graph_dist is not None else float('inf')

        print(f"  Computed {len(graph_distances)} graph distances")

    # Train using multi_train (which calls train internally if num_rounds=1)
    training_history = heuristic.multi_train(
        state_node_pairs=state_node_pairs,
        graph_distances=graph_distances if graph_distances else {},
        num_rounds=heuristic_config.num_rounds,
        num_epochs_per_round=cfg['num_epochs'],
        trajectories_per_epoch=cfg['trajectories_per_epoch'],
        max_episode_steps=cfg['max_episode_steps'],
        keep_fraction=heuristic_config.keep_fraction,
    )

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time

    print(f"[DEBUG] Training complete in {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
    print_memory()

    # Save distance matrices from multi-round training (if any)
    if 'distance_matrices' in training_history and len(training_history['distance_matrices']) > 0:
        import pickle
        distance_matrices_path = output_dir / "distance_matrices.pkl"
        with open(distance_matrices_path, 'wb') as f:
            pickle.dump({
                'distance_matrices': training_history['distance_matrices'],
                'graph_distances': training_history.get('graph_distances', {}),
            }, f)
        print(f"\n[DEBUG] Saved {len(training_history['distance_matrices'])} distance matrices to {distance_matrices_path}")

    # =========================================================================
    # Step 5: Evaluate Policy with Rollouts
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Evaluating policy with rollouts")
    print("=" * 80)

    # Sample random state-node pairs for evaluation
    eval_rollouts = cfg.get("eval_rollouts", 20)
    eval_temperature = cfg.get("eval_temperature", 1e-9)
    eval_max_steps = cfg.get("eval_max_steps", 100)

    eval_pairs_indices = rng.choice(
        len(state_node_pairs),
        size=min(eval_rollouts, len(state_node_pairs)),
        replace=False
    )

    rollout_results = []
    successes = 0
    total_steps = 0
    successful_steps = 0
    total_error = 0
    successful_error = 0

    print(f"Running {len(eval_pairs_indices)} rollouts (temp={eval_temperature})...")
    for idx in eval_pairs_indices:
        state, goal_node_id = state_node_pairs[idx]

        states, nodes, success = heuristic.rollout(
            env=system.env,
            start_state=state,
            goal_node=goal_node_id,
            max_steps=eval_max_steps,
            temperature=eval_temperature,
        )

        start_node_id = nodes[0]
        goal_node_atoms = training_data.node_atoms[goal_node_id]
        true_length = compute_true_distance(system, state, goal_node_atoms)

        rollout_results.append({
            'start_state': state,
            'start_node': start_node_id,
            'goal_node': goal_node_id,
            'goal_atoms': goal_node_atoms,
            'trajectory_length': len(nodes) - 1,
            'success': success,
            'true_length': true_length,
            'all_states': states,
            'all_nodes': nodes
        })


        if success:
            successes += 1
            successful_steps += len(nodes)
            successful_error += (len(nodes) - true_length)
        total_steps += len(nodes)
        total_error += (len(nodes) - true_length)

    success_rate = successes / len(eval_pairs_indices) * 100
    avg_length = total_steps / len(eval_pairs_indices)
    avg_error = total_error / len(eval_pairs_indices)
    avg_success_length = successful_steps / successes if successes > 0 else 0
    avg_success_error = successful_error / successes if successes > 0 else 0

    print(f"\nRollout Evaluation Results:")
    print(f"  Evaluated: {len(eval_pairs_indices)} rollouts")
    print(f"  Success rate: {success_rate:.1f}% ({successes}/{len(eval_pairs_indices)})")
    print(f"  Avg trajectory length: {avg_length:.1f}")
    print(f"  Avg error: {avg_error:.1f}")
    print(f"  Avg successful length: {avg_success_length:.1f}")
    print(f"  Avg successful error: {avg_success_error:.1f}")

    # =========================================================================
    # Step 6: Compare Estimated vs True Node Distances
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Comparing estimated vs true node distances")
    print("=" * 80)

    # Compute all pairwise node distances
    node_ids = sorted([n.id for n in planning_graph.nodes])
    num_nodes = len(node_ids)

    estimated_distances = np.zeros((num_nodes, num_nodes))
    true_distances = np.zeros((num_nodes, num_nodes))
    graph_distances = np.zeros((num_nodes, num_nodes))

    print(f"Computing distances for {num_nodes} x {num_nodes} = {num_nodes**2} node pairs...")
    for i, source_id in enumerate(node_ids):
        for j, target_id in enumerate(node_ids):
            if source_id == target_id:
                continue

            # Estimated distance (from embeddings)
            estimated_distances[i, j] = heuristic.estimate_node_distance(source_id, target_id)

            graph_dist = compute_graph_node_distance(
                planning_graph,
                start_node_atoms=training_data.node_atoms[source_id],
                goal_node_atoms=training_data.node_atoms[target_id]
            )
            graph_distances[i, j] = graph_dist if graph_dist is not None else float('inf')

            # True distance (from graph search)
            true_dist = compute_true_node_distance(
                system,
                start_states=all_node_states[source_id],
                goal_node_atoms=training_data.node_atoms[target_id],
            )
            true_distances[i, j] = true_dist if true_dist is not None else float('inf')

            print("Distances of", source_id, "to", target_id, 
                  ": True=", true_dist,
                  "Graph=", graph_dist,
                  "Learned=", estimated_distances[i, j])

    # Compute statistics
    valid_mask = np.isfinite(true_distances) & (true_distances > 0)
    if valid_mask.any():
        mae = np.mean(np.abs(estimated_distances[valid_mask] - true_distances[valid_mask]))
        rmse = np.sqrt(np.mean((estimated_distances[valid_mask] - true_distances[valid_mask])**2))
        correlation = np.corrcoef(
            estimated_distances[valid_mask].flatten(),
            true_distances[valid_mask].flatten()
        )[0, 1]

        print(f"\nNode Distance Estimation Metrics:")
        print(f"  Valid pairs: {valid_mask.sum()}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Correlation: {correlation:.3f}")
    else:
        print("  Warning: No valid distances to compare")
        mae = rmse = correlation = float('nan')

    # =========================================================================
    # Step 7: Save Heuristic and Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Saving heuristic and results")
    print("=" * 80)

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Save heuristic using built-in save method
    heuristic_path = output_dir / "distance_heuristic_v4.pkl"
    heuristic.save(str(heuristic_path))
    print(f"  Saved heuristic to: {heuristic_path}")

    # Save results
    import pickle
    # Extract portal positions if available (for GridworldFixed)
    portal_positions = None
    if hasattr(system.env, 'portal_positions'):
        # Convert numpy arrays to lists for JSON serialization
        portal_positions = [
            (pos1.tolist(), pos2.tolist())
            for pos1, pos2 in system.env.portal_positions
        ]

    results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'training_data': {
            'num_state_node_pairs': len(state_node_pairs),
            'num_shortcuts': len(training_data.valid_shortcuts),
            'num_nodes': len(planning_graph.nodes),
            'num_edges': len(planning_graph.edges),
            'node_atoms': training_data.node_atoms,  # Dict[int, frozenset[GroundAtom]]
            'node_states': training_data.node_states,  # Dict[int, list[ObsType]]
        },
        'environment': {
            'portal_positions': portal_positions,  # List[Tuple[List[int], List[int]]]
            'num_cells': cfg.get('num_cells', None),
            'num_states_per_cell': cfg.get('num_states_per_cell', None),
        },
        'training_history': training_history,
        'rollout_evaluation': {
            'success_rate': success_rate,
            'avg_length': avg_length,
            'avg_success_length': avg_success_length,
            'avg_error': avg_error,
            'avg_success_error': avg_success_error,
            'rollout_results': rollout_results,
        },
        'distance_evaluation': {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'estimated_distances': estimated_distances,
            'true_distances': true_distances,
            'graph_distances': graph_distances,
            'node_ids': node_ids,
        },
    }

    results_path = output_dir / "heuristic_v4_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved results to: {results_path}")

    print("\n" + "=" * 80)
    print("Training and Evaluation Complete!")
    print("=" * 80)
    print(f"Summary:")
    print(f"  Policy success rate: {success_rate:.1f}%")
    print(f"  Distance MAE: {mae:.3f}")
    print(f"  Distance correlation: {correlation:.3f}")
    print("=" * 80)

    return float(correlation)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
