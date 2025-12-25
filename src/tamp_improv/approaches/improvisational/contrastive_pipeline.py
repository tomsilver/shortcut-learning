"""Contrastive learning pipeline for TAMP.

This module provides the core pipeline for training and evaluating a distance
heuristic V4 for creating shortcuts in TAMP problems.

Pipeline steps:
1. Collect training data (shortcuts)
2. Initialize and train distance heuristic V4
3. (Optional) Evaluate heuristic quality
4. Add shortcuts to the approach
5. Evaluate the approach with shortcuts
"""

from __future__ import annotations

import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tamp_improv.approaches.improvisational.analyze import (
    compute_graph_node_distance,
    compute_true_node_distance,
)
from tamp_improv.approaches.improvisational.base import (
    ImprovisationalTAMPApproach,
)
from tamp_improv.approaches.improvisational.collection import (
    collect_total_shortcuts,
)
from tamp_improv.approaches.improvisational.contrastive_approach import (
    ContrastiveApproach,
)
from tamp_improv.approaches.improvisational.distance_heuristic_v4 import (
    DistanceHeuristicV4,
    DistanceHeuristicV4Config,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    run_evaluation_episode,
    TrainingConfig,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


@dataclass
class ContrastivePipelineConfig:
    """Configuration for contrastive learning pipeline."""

    # Random seed
    seed: int = 42

    # Collection parameters
    collect_episodes: int = 100

    # Distance heuristic V4 parameters
    latent_dim: int = 16
    hidden_dims: list[int] | None = None
    learning_rate: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 100000
    gamma: float = 0.99
    repetition_factor: int = 2
    policy_temperature: float = 1.0
    eval_temperature: float = 0.0
    iters_per_epoch: int = 100
    learn_frequency: int = 1

    # Training parameters for V4
    num_epochs: int = 500
    trajectories_per_epoch: int = 10
    max_episode_steps: int = 100

    # Evaluation parameters
    eval_rollouts: int = 20
    eval_max_steps: int = 100
    fast_eval: bool = False

    # Debug mode (enables heuristic quality evaluation)
    debug: bool = False


def train_and_evaluate(
    system: ImprovisationalTAMPSystem[Any, Any],
    config: ContrastivePipelineConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Train distance heuristic and evaluate with shortcuts.

    Args:
        system: TAMP system
        config: Pipeline configuration
        output_dir: Directory to save results

    Returns:
        Dictionary of results
    """
    cfg = config  # Alias for compatibility with existing code

    # =========================================================================
    # Step 1: System already initialized
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: System Initialized")
    print("=" * 80)

    # =========================================================================
    # Step 2: Collect training data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Collecting Training Data")
    print("=" * 80)

    # Create temporary approach for graph building
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    rl_config = RLConfig(device=device)
    dummy_policy = MultiRLPolicy(seed=cfg.seed, config=rl_config)
    temp_approach = ImprovisationalTAMPApproach(system, dummy_policy, seed=cfg.seed)
    temp_approach.training_mode = True

    # Collect shortcuts
    # Convert dataclass to dict for collect_total_shortcuts
    config_dict = asdict(cfg)
    training_data = collect_total_shortcuts(
        system=system,
        approach=temp_approach,
        config=config_dict,
    )

    planning_graph = training_data.graph

    print(f"\n[Collection Summary]")
    print(f"  Planning graph nodes: {len(planning_graph.nodes)}")
    print(f"  Planning graph edges: {len(planning_graph.edges)}")
    print(f"  Valid shortcuts found: {len(training_data.valid_shortcuts)}")

    state_node_pairs = []
    for node_id, states in training_data.node_states.items():
        # Find node by atoms
        for state in states:
            # Add pairs to all other nodes
            for target_node in planning_graph.nodes:
                if target_node.id != node_id:
                    state_node_pairs.append((state, target_node.id))

    print(f"  State-node pairs: {len(state_node_pairs)}")

    # Create atoms_to_node mapping from planning graph
    atoms_to_node = {}
    for node in planning_graph.nodes:
        atoms_to_node[node.atoms] = node.id

    # =========================================================================
    # Step 3: Initialize distance heuristic V4
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Initializing Distance Heuristic V4")
    print("=" * 80)

    heuristic_config = DistanceHeuristicV4Config(
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims if cfg.hidden_dims else [64, 64],
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        gamma=cfg.gamma,
        repetition_factor=cfg.repetition_factor,
        policy_temperature=cfg.policy_temperature,
        eval_temperature=cfg.eval_temperature,
        iters_per_epoch=cfg.iters_per_epoch,
        learn_frequency=cfg.learn_frequency,
        device=device,
    )

    heuristic = DistanceHeuristicV4(
        env=system.env,
        perceiver=system.perceiver,
        atoms_to_node=atoms_to_node,
        node_to_states=training_data.node_states,
        config=heuristic_config,
        seed=cfg.seed,
    )

    print(f"  Latent dimension: {cfg.latent_dim}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Batch size: {cfg.batch_size}")

    # =========================================================================
    # Step 4: Train the heuristic
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Training Distance Heuristic")
    print("=" * 80)

    training_history = heuristic.train(
        state_node_pairs=state_node_pairs,
        num_epochs=cfg.num_epochs,
        trajectories_per_epoch=cfg.trajectories_per_epoch,
        max_episode_steps=cfg.max_episode_steps,
    )

    print(f"\n[Training Complete]")
    print(
        f"  Final success rate: {training_history['success_rate'][-1] if training_history['success_rate'] else 0:.1%}"
    )
    print(
        f"  Final loss: {training_history['total_loss'][-1] if training_history['total_loss'] else 0:.4f}"
    )

    # Save trained heuristic
    heuristic_path = output_dir / "distance_heuristic_v4"
    heuristic.save(heuristic_path)
    print(f"  Saved heuristic to: {heuristic_path}")

    # =========================================================================
    # Step 4.5: (Optional) Evaluate heuristic quality
    # =========================================================================
    if cfg.debug:
        print("\n" + "=" * 80)
        print("Step 4.5: Evaluating Heuristic Quality")
        print("=" * 80)

        # Compute distance estimates for node pairs
        node_ids = list(training_data.node_atoms.keys())
        num_nodes = len(node_ids)

        estimated_distances = np.zeros((num_nodes, num_nodes))
        true_distances = np.zeros((num_nodes, num_nodes))
        graph_distances = np.zeros((num_nodes, num_nodes))

        for i, source_id in enumerate(node_ids):
            source_states = training_data.node_states.get(source_id, [])
            if not source_states:
                continue
            source_state = source_states[0]
            source_atoms = training_data.node_atoms[source_id]

            for j, target_id in enumerate(node_ids):
                target_atoms = training_data.node_atoms[target_id]

                # Estimated distance
                estimated_distances[i, j] = heuristic.estimate_node_distance(source_id, target_id)

                # True distance
                true_distances[i, j] = compute_true_node_distance(
                    system=system,
                    start_states=source_states,
                    goal_node_atoms=target_atoms,
                )

                # Graph distance
                graph_distances[i, j] = compute_graph_node_distance(
                    planning_graph=planning_graph,
                    start_node_atoms=source_atoms,
                    goal_node_atoms=target_atoms,
                )

        # Compute metrics
        valid_mask = np.isfinite(true_distances) & (true_distances > 0)
        if np.any(valid_mask):
            mae = np.mean(np.abs(estimated_distances[valid_mask] - true_distances[valid_mask]))
            rmse = np.sqrt(np.mean((estimated_distances[valid_mask] - true_distances[valid_mask]) ** 2))
            correlation = np.corrcoef(
                estimated_distances[valid_mask].flatten(),
                true_distances[valid_mask].flatten()
            )[0, 1]

            print(f"\n[Distance Estimation Metrics]")
            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  Correlation: {correlation:.3f}")

        # Rollout-based evaluation
        print(f"\n[Rollout-Based Evaluation]")
        num_rollout_pairs = min(20, num_nodes * (num_nodes - 1))  # Sample up to 20 pairs
        rollout_successes = 0
        rollout_step_errors = []
        max_rollout_steps = cfg.max_episode_steps

        # Sample random source-target pairs
        sampled_pairs = []
        for _ in range(num_rollout_pairs):
            source_idx = random.randint(0, num_nodes - 1)
            target_idx = random.randint(0, num_nodes - 1)
            if source_idx != target_idx:
                sampled_pairs.append((source_idx, target_idx))

        for source_idx, target_idx in sampled_pairs:
            source_id = node_ids[source_idx]
            target_id = node_ids[target_idx]

            source_states = training_data.node_states.get(source_id, [])
            if not source_states:
                continue

            source_state = source_states[0]

            # Perform rollout
            states, nodes, success = heuristic.rollout(
                env=system.env,
                start_state=source_state,
                goal_node=target_id,
                max_steps=max_rollout_steps,
                temperature=cfg.eval_temperature,
            )

            if success:
                rollout_successes += 1
                # Compare rollout steps to graph distance
                rollout_steps = len(states) - 1  # Number of steps taken
                graph_dist = graph_distances[source_idx, target_idx]

                if np.isfinite(graph_dist) and graph_dist > 0:
                    error = abs(rollout_steps - graph_dist)
                    rollout_step_errors.append(error)

        success_rate = rollout_successes / len(sampled_pairs) if sampled_pairs else 0
        avg_step_error = np.mean(rollout_step_errors) if rollout_step_errors else float("inf")

        print(f"  Rollout pairs tested: {len(sampled_pairs)}")
        print(f"  Success rate: {success_rate:.1%} ({rollout_successes}/{len(sampled_pairs)})")
        print(f"  Avg step error vs graph distance: {avg_step_error:.3f}")

    # =========================================================================
    # Step 5: Add shortcuts to approach
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Creating Contrastive Approach with Shortcuts")
    print("=" * 80)

    # Create the contrastive approach
    approach = ContrastiveApproach(
        system=system,
        heuristic=heuristic,
        seed=cfg.seed,
    )
    approach.training_mode = False  # Set to eval mode

    # Add shortcuts for all node pairs
    # Dijkstra will naturally select the best paths during planning
    node_ids = list(training_data.node_atoms.keys())
    num_added = 0

    for source_id in node_ids:
        source_atoms = training_data.node_atoms[source_id]
        for target_id in node_ids:
            if source_id == target_id:
                continue  # Skip self-loops

            target_atoms = training_data.node_atoms[target_id]
            approach.add_shortcut(
                source_atoms=source_atoms,
                target_atoms=target_atoms,
                target_node_id=target_id,
                temperature=0.0,
            )
            num_added += 1

    print(f"[INFO] Added {num_added} shortcuts for all node pairs")

    # Debug: Print edge count per node to see if we have multiple edges
    print(f"\n[DEBUG] Edge counts per node in planning graph after adding shortcuts:")
    for node in planning_graph.nodes[:5]:  # Just first 5 nodes
        outgoing = len(planning_graph.node_to_outgoing_edges.get(node, []))
        print(f"  Node {node.id}: {outgoing} outgoing edges")

    # =========================================================================
    # Step 6: Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating Approach with Shortcuts")
    print("=" * 80)

    training_config = TrainingConfig(
        eval_max_steps=cfg.eval_max_steps,
        render=False,
        fast_eval=cfg.fast_eval,
    )

    eval_episodes = cfg.eval_rollouts
    eval_results = []

    for episode_num in range(eval_episodes):
        system.reset()
        total_reward, step_count, success = run_evaluation_episode(
            system=system,
            approach=approach,
            policy_name="contrastive",
            config=training_config,
            episode_num=episode_num,
        )
        print("Episode", episode_num, "succeded:", success, f"(step count {step_count})")

        eval_results.append({
            "episode": episode_num,
            "success": success,
            "steps": step_count,
            "reward": total_reward,
        })

        if (episode_num + 1) % 10 == 0:
            successes = sum(1 for r in eval_results if r["success"])
            print(
                f"  Episode {episode_num + 1}/{eval_episodes}: "
                f"{successes}/{episode_num + 1} successes "
                f"({100 * successes / (episode_num + 1):.1f}%)"
            )

    # Compute summary statistics
    successes = sum(1 for r in eval_results if r["success"])
    success_rate = 100 * successes / len(eval_results)
    avg_steps = np.mean([r["steps"] for r in eval_results])
    success_steps = [r["steps"] for r in eval_results if r["success"]]
    avg_success_steps = np.mean(success_steps) if success_steps else 0

    print(f"\n[Evaluation Summary]")
    print(f"  Success rate: {success_rate:.1f}% ({successes}/{len(eval_results)})")
    print(f"  Average steps: {avg_steps:.1f}")
    print(f"  Average successful steps: {avg_success_steps:.1f}")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    results = {
        "config": {
            "seed": cfg.seed,
            "collect_episodes": cfg.collect_episodes,
            "latent_dim": cfg.latent_dim,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "trajectories_per_epoch": cfg.trajectories_per_epoch,
            "max_episode_steps": cfg.max_episode_steps,
            "eval_rollouts": cfg.eval_rollouts,
            "eval_max_steps": cfg.eval_max_steps,
        },
        "training_data": {
            "num_state_node_pairs": len(state_node_pairs),
            "num_shortcuts": len(training_data.valid_shortcuts),
            "num_nodes": len(planning_graph.nodes),
            "num_edges": len(planning_graph.edges),
            "node_atoms": training_data.node_atoms,
            "node_states": training_data.node_states,
        },
        "training_history": training_history,
        "evaluation": {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_success_steps": avg_success_steps,
            "eval_results": eval_results,
        },
        "shortcuts": {
            "num_added": num_added,
            "total_node_pairs": len(node_ids) * (len(node_ids) - 1),
        },
    }

    results_path = output_dir / "contrastive_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved results to: {results_path}")

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)

    return results
