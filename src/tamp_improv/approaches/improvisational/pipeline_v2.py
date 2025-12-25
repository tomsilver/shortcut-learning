"""Simplified pipeline for SLAP training - V2 with unified heuristic interface.

This pipeline treats all heuristic methods uniformly through a common interface,
eliminating special cases and complexity. No caching - just clean execution.

Pipeline stages:
1. Collect data - collect_total_shortcuts → training_data + planning_graph + graph_distances
2. Train heuristic - heuristic.multi_train() → trained heuristic + training_history
3. Test heuristic quality (if debug) → heuristic quality results
4. Prune with heuristic - heuristic.prune() → pruned_training_data
5. Random selection - randomly select max_shortcuts_per_graph shortcuts
6. Train policy - policy.train() → trained policy
7. Test shortcut quality (if debug) → shortcut quality results
8. Evaluate - run_evaluation_episode() → evaluation results

The pipeline returns all results; the experiment handles saving.
"""
import time
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Union
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic

from tamp_improv.approaches.improvisational.analyze import compute_true_node_distance
from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.collection import collect_total_shortcuts
from tamp_improv.approaches.improvisational.graph_training import compute_graph_distances

from tamp_improv.approaches.improvisational.heuristics.heuristic_none import NoneHeuristic
from tamp_improv.approaches.improvisational.heuristics.heuristic_rollouts import RolloutsHeuristic
from tamp_improv.approaches.improvisational.heuristics.heuristic_smart_rollouts import SmartRolloutsHeuristic
from tamp_improv.approaches.improvisational.heuristics.heuristic_crl import CRLHeuristic, CRLHeuristicConfig
from tamp_improv.approaches.improvisational.heuristics.heuristic_dqn import DQNHeuristic, DQNHeuristicConfig
from tamp_improv.approaches.improvisational.heuristics.heuristic_cmd import CMDHeuristic, CMDHeuristicConfig


from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
    Policy,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    Metrics,
    TrainingConfig,
    run_evaluation_episode,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.utils.gpu_utils import set_torch_seed

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


# =============================================================================
# Helper Functions: Create Heuristics and Policies
# =============================================================================

from dataclasses import fields

def dataclass_from_cfg(dataclass_type, cfg_section):
    field_names = {f.name for f in fields(dataclass_type)}

    kwargs = {
        k: v
        for k, v in cfg_section.items()
        if k in field_names
    }

    return dataclass_type(**kwargs)

def create_heuristic(
    training_data: GoalConditionedTrainingData,
    graph_distances: dict[tuple[int, int], float],
    system: ImprovisationalTAMPSystem,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> "BaseHeuristic":
    """Create a heuristic instance based on type.

    Args:
        heuristic_type: Type of heuristic ("rollouts", "v4", etc.)
        training_data: Training data from collection
        graph_distances: Graph distances
        system: TAMP system
        config: Configuration dictionary

    Returns:
        Heuristic instance
    """
    if cfg.heuristic.type == "none":
        return NoneHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            rng=rng
        )
    elif cfg.heuristic.type == "rollouts":
        return RolloutsHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            num_rollouts=cfg.heuristic.num_rollouts_per_node,
            max_steps_per_rollout=cfg.heuristic.max_steps_per_rollout,
            threshold=cfg.heuristic.shortcut_success_threshold,
            action_scale=cfg.heuristic.action_scale,
            seed=cfg.seed,
            rng=rng
        )
    elif cfg.heuristic.type == "smart_rollouts":
        return SmartRolloutsHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            num_rollouts=cfg.heuristic.num_rollouts_per_node,
            max_steps_per_rollout=cfg.heuristic.max_steps_per_rollout,
            threshold=cfg.heuristic.shortcut_success_threshold,
            action_scale=cfg.heuristic.action_scale,
            seed=cfg.seed,
        )
    elif cfg.heuristic.type == "crl":
        # crl_config is now created outside this function for broader scope
        crl_config = dataclass_from_cfg(CRLHeuristicConfig, cfg.heuristic)
        crl_config.wandb_enabled = cfg.wandb_enabled
        print("CRL Config:", crl_config)

        return CRLHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=crl_config,
            seed=cfg.seed,
        )
    elif cfg.heuristic.type == "dqn":
        # TODO: Implement V4 heuristic
        dqn_config = dataclass_from_cfg(DQNHeuristicConfig, cfg.heuristic.dqn)
        print("DQN Config:", dqn_config)

        return DQNHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=dqn_config,
            seed=cfg.seed,
        )
    elif cfg.heuristic.type == "cmd":
        # TODO: Implement V4 heuristic
        cmd_config = dataclass_from_cfg(CMDHeuristicConfig, cfg.heuristic)
        cmd_config.wandb_enabled = cfg.wandb_enabled
        print("CMD Config:", cmd_config)

        return CMDHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=system,
            config=cmd_config,
            seed=cfg.seed,
        )
    else:   
        raise ValueError(f"Unknown heuristic type: {cfg.heuristic.type}")


def create_policy(
    cfg: DictConfig,
) -> Policy:
    """Create a policy instance based on type.

    Args:
        policy_type: Type of policy ("multiRL")
        system: TAMP system
        config: Configuration dictionary with rl_* parameters

    Returns:
        Policy instance
    """
    if cfg.policy.type == "multiRL":
        # Extract RL config from parameters prefixed with rl_
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rl_config = RLConfig(
            learning_rate=cfg.policy.learning_rate,
            batch_size=cfg.policy.batch_size,
            n_epochs=cfg.policy.n_epochs,
            gamma=cfg.policy.gamma,
            ent_coef=cfg.policy.ent_coef,
            deterministic=cfg.policy.deterministic,
            device=device,
        )
        return MultiRLPolicy(seed=cfg.seed, config=rl_config)
    else:
        raise ValueError(f"Unknown policy type: {cfg.policy.type}")


# =============================================================================
# Pipeline Results
# =============================================================================


class PipelineResults:
    """Container for all pipeline outputs."""

    def __init__(self):
        # Core components
        self.approach: ImprovisationalTAMPApproach | None = None
        self.policy: Policy | None = None

        # Stage outputs
        self.training_data: GoalConditionedTrainingData | None = None
        self.graph_distances: dict[tuple[int, int], float] | None = None
        self.heuristic: "BaseHeuristic | None" = None
        self.heuristic_training_history: dict[str, Any] | None = None
        self.heuristic_quality_results: dict[str, Any] | None = None
        self.pruned_training_data: GoalConditionedTrainingData | None = None
        self.shortcut_quality_results: dict[str, Any] | None = None
        self.successful_shortcut_paths: list[list[Any]] | None = None # New field for storing successful shortcut paths
        self.evaluation_results: Metrics | None = None
        self.times: dict[str, float] | None = None


# =============================================================================
# Stage 1: Collection
# =============================================================================


def collect_training_data(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[GoalConditionedTrainingData, dict[tuple[int, int], float]]:
    """Stage 1: Collect all shortcuts and compute graph distances.

    Args:
        system: TAMP system
        approach: Improvisational TAMP approach
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        Tuple of (training_data, graph_distances)
    """
    print("\n" + "=" * 80)
    print("STAGE 1: COLLECT TRAINING DATA")
    print("=" * 80)

    # Collect shortcuts
    training_data = collect_total_shortcuts(
        system=system,
        approach=approach,
        cfg=cfg,
        rng=rng,
    )

    print(f"\nCollected {len(training_data.unique_shortcuts)} unique shortcuts")
    print(f"  ({len(training_data.valid_shortcuts)} state-node pairs total)")
    print(f"Planning graph has {len(training_data.graph.nodes) if training_data.graph else 0} nodes")

    # Compute graph distances
    print("\nComputing graph distances...")
    graph_distances = compute_graph_distances(training_data.graph, exclude_shortcuts=True)
    print(f"Computed {len(graph_distances)} pairwise distances")

    return training_data, graph_distances


# =============================================================================
# Stage 2: Train Heuristic
# =============================================================================


def train_heuristic(
    heuristic: "BaseHeuristic",
) -> dict[str, Any]:
    """Stage 2: Train the heuristic.

    Args:
        heuristic: Initialized heuristic instance
        config: Configuration dictionary

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 80)
    print("STAGE 2: TRAIN HEURISTIC")
    print("=" * 80)

    # Call multi_train - interface is the same for all heuristics
    training_history = heuristic.multi_train()

    print("\nHeuristic training complete")

    return training_history


# =============================================================================
# Stage 2.5: Test Heuristic Quality (Optional)
# =============================================================================


def test_heuristic_quality(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    heuristic: "BaseHeuristic",
    training_data: GoalConditionedTrainingData,
    graph_distances: dict[tuple[int, int], float],
    cfg: DictConfig,
    times: dict[str, float],
) -> dict[str, Any]:
    """Stage 2.5: Test heuristic quality on sample node pairs.

    Args:
        heuristic: Trained heuristic
        training_data: Full training data
        graph_distances: Graph distances
        config: Configuration dictionary

    Returns:
        Dictionary with heuristic quality results
    """
    print("\n" + "=" * 80)
    print("STAGE 2.5: TEST HEURISTIC QUALITY")
    print("=" * 80)

    # If there are no nodes in the graph, skip plotting
    if not training_data.graph or not training_data.graph.nodes:
        print("[WARN] No nodes in the planning graph. Skipping heuristic quality plotting.")
        return {"samples": [], "avg_estimated_distance": 0.0, "avg_graph_distance": 0.0, "avg_absolute_error": 0.0}

    # Sample some node pairs to evaluate (use unique_shortcuts for node-node pairs)
    shortcuts = training_data.unique_shortcuts
    sample_indices = [i for i in range(len(shortcuts))]

    results = {
        "samples": [],
        "avg_estimated_distance": 0.0,
        "avg_graph_distance": 0.0,
        "avg_absolute_error": 0.0,
    }

    total_est = 0.0
    total_graph = 0.0
    total_abs_error = 0.0
    all_estimated_distances = []
    all_graph_distances = []
    all_true_distances = []

    print(f"\nEvaluating heuristic on {len(sample_indices)} sample shortcuts:")
    print(f"{'Source':>8} {'Target':>8} {'Estimated':>12} {'Graph':>12} {'Error':>12}")
    print("-" * 60)
    

    for idx in sample_indices:
        source_id, target_id = shortcuts[idx]
        estimated_dist = heuristic.estimate_node_distance(source_id, target_id)
        graph_dist = graph_distances.get((source_id, target_id), float("inf"))
        true_dist = compute_true_node_distance(system,
                                               training_data.node_states[source_id], 
                                               training_data.node_atoms[target_id])
        abs_error = abs(estimated_dist - graph_dist) if graph_dist != float("inf") else float("inf")

        results["samples"].append({
            "source_id": source_id,
            "target_id": target_id,
            "estimated_distance": estimated_dist,
            "graph_distance": graph_dist,
            "true_distance": true_dist,
            "absolute_error": abs_error,
        })

        print(f"{source_id:8d} {target_id:8d} {estimated_dist:12.2f} {graph_dist:12.2f} {abs_error:12.2f}")
        print(f"TRUE DISTANCE: {true_dist}")
        
        if graph_dist != float("inf"):
            total_est += estimated_dist
            total_graph += graph_dist
            total_abs_error += abs_error
            all_estimated_distances.append(estimated_dist)
            all_graph_distances.append(graph_dist)
            all_true_distances.append(true_dist)

    # Compute averages
    num_finite = sum(1 for s in results["samples"] if s["graph_distance"] != float("inf"))
    if num_finite > 0:
        results["avg_estimated_distance"] = total_est / num_finite
        results["avg_graph_distance"] = total_graph / num_finite
        results["avg_absolute_error"] = total_abs_error / num_finite
        
        # Compute min/max/correlation
        min_estimated_distance = np.min(all_estimated_distances)
        max_estimated_distance = np.max(all_estimated_distances)
        min_graph_distance = np.min(all_graph_distances)
        max_graph_distance = np.max(all_graph_distances)
        correlation_distance = np.corrcoef(all_estimated_distances, all_graph_distances)[0, 1]

        print("-" * 60)
        print(f"{'Average':>8} {'':<8} {results['avg_estimated_distance']:12.2f} {results['avg_graph_distance']:12.2f} {results['avg_absolute_error']:12.2f}")
        print(f"{'Min Estimated':>28} {min_estimated_distance:12.2f}")
        print(f"{'Max Estimated':>28} {max_estimated_distance:12.2f}")
        print(f"{'Min Graph':>28} {min_graph_distance:12.2f}")
        print(f"{'Max Graph':>28} {max_graph_distance:12.2f}")
        print(f"{'Correlation':>28} {correlation_distance:12.2f}")

        # Get all unique node IDs from the graph
        all_graph_node_ids = sorted([node.id for node in training_data.graph.nodes])
        num_nodes = len(all_graph_node_ids)

        # Initialize dense matrices with NaN
        true_distances_matrix = np.full((num_nodes, num_nodes), np.nan)
        graph_distances_matrix = np.full((num_nodes, num_nodes), np.nan)
        estimated_distances_matrix = np.full((num_nodes, num_nodes), np.nan)

        node_id_to_idx = {node_id: i for i, node_id in enumerate(all_graph_node_ids)}

        # Populate matrices
        for sample in results["samples"]:
            src_idx = node_id_to_idx[sample["source_id"]]
            tgt_idx = node_id_to_idx[sample["target_id"]]
            true_distances_matrix[src_idx, tgt_idx] = sample["true_distance"]
            graph_distances_matrix[src_idx, tgt_idx] = sample["graph_distance"]
            estimated_distances_matrix[src_idx, tgt_idx] = sample["estimated_distance"]

        if cfg.wandb_enabled:
            _log_distance_plots_to_wandb(
                true_distances=true_distances_matrix,
                graph_distances=graph_distances_matrix,
                estimated_distances=estimated_distances_matrix,
                node_ids=all_graph_node_ids,
            )

        if cfg.wandb_enabled:
            wandb.log({
                "test/avg_estimated_distance": results["avg_estimated_distance"],
                "test/avg_graph_distance": results["avg_graph_distance"],
                "test/avg_absolute_error": results["avg_absolute_error"],
                "test/min_estimated_distance": min_estimated_distance,
                "test/max_estimated_distance": max_estimated_distance,
                "test/min_graph_distance": min_graph_distance,
                "test/max_graph_distance": max_graph_distance,
                "test/correlation_distance": correlation_distance,
                "times/heuristic_training_time": times['heuristic_training_time'],
            })

    return results


def _log_distance_plots_to_wandb(
    true_distances: np.ndarray,
    graph_distances: np.ndarray,
    estimated_distances: np.ndarray,
    node_ids: list[int],
) -> None:
    """Helper to plot distance matrices and scatterplots and log to WandB."""
    # Create figure for heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Common settings for heatmaps
    vmin = 0
    vmax = max(np.nanmax(true_distances[np.isfinite(true_distances)]),
               np.nanmax(estimated_distances))

    # Determine whether to show annotations (hide if grid is larger than 5x5)
    num_nodes = len(node_ids)
    show_annot = num_nodes <= 25  # 5x5 or smaller

    # Heatmap 1: True Distances
    sns.heatmap(true_distances, annot=show_annot, fmt='.1f', cmap='YlOrRd',
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Distance'},
                ax=axes[0])
    axes[0].set_title('True Distances (Graph Search)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Target Node', fontsize=12)
    axes[0].set_ylabel('Source Node', fontsize=12)

    # Heatmap 2: Graph Distances
    sns.heatmap(graph_distances, annot=show_annot, fmt='.1f', cmap='YlOrRd',
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Distance'},
                ax=axes[1])
    axes[1].set_title('Graph Distances (Planning Graph)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Target Node', fontsize=12)
    axes[1].set_ylabel('Source Node', fontsize=12)

    # Heatmap 3: Estimated Distances
    sns.heatmap(estimated_distances, annot=show_annot, fmt='.1f', cmap='YlOrRd',
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Distance'},
                ax=axes[2])
    axes[2].set_title('Learned Distances (V4 Heuristic)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Target Node', fontsize=12)
    axes[2].set_ylabel('Source Node', fontsize=12)

    plt.tight_layout()
    wandb.log({"test/distance_heatmaps": wandb.Image(fig)})
    plt.close(fig) # Close the figure to free up memory

    # Create scatterplots comparing distances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out invalid distances
    valid_mask = np.isfinite(true_distances) & (true_distances > 0)
    valid_indices = np.where(valid_mask)

    true_flat = true_distances[valid_indices]
    graph_flat = graph_distances[valid_indices]
    estimated_flat = estimated_distances[valid_indices]

    # Scatterplot 1: Learned vs True
    axes[0].scatter(true_flat, estimated_flat, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

    # Add perfect prediction line
    max_val = max(np.max(true_flat), np.max(estimated_flat))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Compute and display correlation
    corr = np.corrcoef(true_flat, estimated_flat)[0, 1]
    mae = np.mean(np.abs(estimated_flat - true_flat))
    rmse = np.sqrt(np.mean((estimated_flat - true_flat)**2))

    axes[0].set_xlabel('True Distance', fontsize=12)
    axes[0].set_ylabel('Learned Distance', fontsize=12)
    axes[0].set_title(f'Learned vs True Distances\nCorr={corr:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Scatterplot 2: Graph vs True
    axes[1].scatter(true_flat, graph_flat, alpha=0.6, s=80, color='green',
                    edgecolors='black', linewidth=0.5)

    # Add perfect prediction line
    max_val = max(np.max(true_flat), np.max(graph_flat))
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')

    # Compute correlation
    corr_graph = np.corrcoef(true_flat, graph_flat)[0, 1]
    mae_graph = np.mean(np.abs(graph_flat - true_flat))

    axes[1].set_xlabel('True Distance (Search)', fontsize=12)
    axes[1].set_ylabel('Graph Distance (Planning Graph)', fontsize=12)
    axes[1].set_title(f'Graph vs True Distances\nCorr={corr_graph:.3f}, MAE={mae_graph:.2f}',
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"test/distance_scatterplots": wandb.Image(fig)})
    plt.close(fig) # Close the figure to free up memory

    # Error heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    error_matrix = estimated_distances - true_distances
    error_matrix[~valid_mask] = np.nan

    max_error = np.nanmax(np.abs(error_matrix))
    sns.heatmap(error_matrix, annot=show_annot, fmt='.1f', cmap='RdBu_r', center=0,
                xticklabels=node_ids, yticklabels=node_ids,
                vmin=-max_error, vmax=max_error,
                cbar_kws={'label': 'Error (Learned - True)'},
                ax=ax)
    ax.set_title('Distance Estimation Error', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Node', fontsize=12)
    ax.set_ylabel('Source Node', fontsize=12)

    plt.tight_layout()
    wandb.log({"test/distance_error_heatmap": wandb.Image(fig)})
    plt.close(fig) # Close the figure to free up memory


# =============================================================================
# Stage 3: Prune with Heuristic
# =============================================================================


def prune_with_heuristic(
    heuristic: "BaseHeuristic",
    max_shortcuts: int | None
) -> GoalConditionedTrainingData:
    """Stage 3: Prune shortcuts using the heuristic.

    Args:
        heuristic: Trained heuristic
        config: Configuration dictionary

    Returns:
        Pruned training data
    """
    print("\n" + "=" * 80)
    print("STAGE 3: PRUNE WITH HEURISTIC")
    print("=" * 80)

    # Call prune - interface is the same for all heuristics
    pruned_data = heuristic.prune(max_shortcuts=max_shortcuts)

    print(f"\nPruning complete: {len(pruned_data.unique_shortcuts)} unique shortcuts remaining")
    print(f"  ({len(pruned_data.valid_shortcuts)} state-node pairs)")

    return pruned_data


# =============================================================================
# Stage 3.5: Random Selection
# =============================================================================


def random_selection(
    training_data: GoalConditionedTrainingData,
    max_shortcuts: int,
    rng: np.random.Generator,
) -> GoalConditionedTrainingData:
    """Stage 3.5: Randomly select up to max_shortcuts from pruned data.

    If max_shortcuts is 0, returns empty training data (pure planning mode).
    If max_shortcuts >= num shortcuts, returns all shortcuts unchanged.

    Note: We select from unique_shortcuts (node-node pairs) but keep all
    corresponding state-node pairs in valid_shortcuts for MultiRL training.

    Args:
        training_data: Pruned training data
        max_shortcuts: Maximum number of shortcuts to keep (0 = pure planning)
        rng: Random number generator

    Returns:
        Training data with randomly selected shortcuts
    """
    print("\n" + "=" * 80)
    print("STAGE 3.5: RANDOM SELECTION")
    print("=" * 80)

    num_shortcuts = len(training_data.unique_shortcuts)

    # Handle edge cases
    if max_shortcuts == 0:
        print("max_shortcuts_per_graph = 0: Using pure planning (no shortcuts)")
        # Return empty training data
        return GoalConditionedTrainingData(
            states=[],
            current_atoms=[],
            goal_atoms=[],
            valid_shortcuts=[],
            unique_shortcuts=[],
            node_states=training_data.node_states,
            node_atoms=training_data.node_atoms,
            graph=training_data.graph,
            config={
                **training_data.config,
                "random_selection": True,
                "max_shortcuts_per_graph": max_shortcuts,
            },
        )

    if max_shortcuts >= num_shortcuts:
        print(f"max_shortcuts_per_graph ({max_shortcuts}) >= num shortcuts ({num_shortcuts}): Keeping all shortcuts")
        return training_data

    # Random selection
    print(f"Randomly selecting {max_shortcuts} from {num_shortcuts} unique shortcuts")

    # Randomly select unique shortcuts (node-node pairs)
    print(training_data.unique_shortcuts)
    sample = (rng.choice(training_data.unique_shortcuts, size=max_shortcuts, replace=False).tolist())

    selected_unique_shortcuts = set(
        (x, y) for x, y in sample
    )

    # Filter valid_shortcuts to keep only those matching selected unique shortcuts
    # valid_shortcuts has one entry per state-node pair, so we filter by node pairs
    selected_indices = []
    for i, (source_id, target_id) in enumerate(training_data.valid_shortcuts):
        if (source_id, target_id) in selected_unique_shortcuts:
            selected_indices.append(i)

    # Filter training data
    original_shortcut_info = training_data.config.get("shortcut_info", [])
    selected_shortcut_info = (
        [original_shortcut_info[i] for i in selected_indices]
        if original_shortcut_info
        else []
    )

    selected_data = GoalConditionedTrainingData(
        states=[training_data.states[i] for i in selected_indices],
        current_atoms=[training_data.current_atoms[i] for i in selected_indices],
        goal_atoms=[training_data.goal_atoms[i] for i in selected_indices],
        valid_shortcuts=[training_data.valid_shortcuts[i] for i in selected_indices],
        unique_shortcuts=list(selected_unique_shortcuts),
        node_states=training_data.node_states,
        node_atoms=training_data.node_atoms,
        graph=training_data.graph,
        config={
            **training_data.config,
            "shortcut_info": selected_shortcut_info,
            "random_selection": True,
            "max_shortcuts_per_graph": max_shortcuts,
        },
    )

    print(f"Selected {len(selected_data.unique_shortcuts)} unique shortcuts")
    print(f"  ({len(selected_data.valid_shortcuts)} state-node pairs)")

    return selected_data


# =============================================================================
# Stage 4: Train Policy
# =============================================================================


def train_policy(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy: Policy[ObsType, ActType],
    training_data: GoalConditionedTrainingData,
    cfg: DictConfig,
) -> None:
    """Stage 4: Train policy on final training data.

    Args:
        policy: Policy instance to train
        training_data: Final training data (after pruning and selection)
        system: TAMP system
        config: Configuration dictionary
        rng: Random number generator
    """
    print("\n" + "=" * 80)
    print("STAGE 4: TRAIN POLICY")
    print("=" * 80)

    # Skip training if no shortcuts
    if len(training_data.valid_shortcuts) == 0:
        print("No shortcuts to train on - skipping policy training")
        return

    # Extract policy training parameters
    num_epochs = cfg.policy.n_epochs
    train_steps_per_shortcut = cfg.policy.max_episode_steps

    print(f"Training policy for {num_epochs} epochs")
    print(f"Max steps per shortcut: {train_steps_per_shortcut}")
    if hasattr(system.wrapped_env, "configure_training"):
            system.wrapped_env.configure_training(training_data)

    # Train policy
    policy.train(
        env=system.wrapped_env,
        train_data=training_data,
    )

    print("\nPolicy training complete")


# =============================================================================
# Stage 4.5: Test Shortcut Quality (Optional)
# =============================================================================

def test_shortcut_quality(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy: Policy,
    training_data: GoalConditionedTrainingData,
    cfg: DictConfig,
) -> None:
    """Test quality of trained shortcuts by running rollouts.

    Args:
        system: The TAMP system
        policy: Trained policy
        training_data: Training data with shortcuts
        config: Config with max_steps
    """
    from tamp_improv.approaches.improvisational.analyze import execute_shortcut_once

    print(f"\nTesting {len(training_data.unique_shortcuts)} shortcuts...")
    print("=" * 80)
    results = []
    successful_shortcut_paths = []

    max_steps = cfg.policy.max_episode_steps
    num_test_rollouts = cfg.policy.eval_rollouts  # Test each shortcut multiple times

    success_counts = {}
    length_stats = {}

    for idx, (source_node, target_node) in enumerate(training_data.unique_shortcuts):
        source_states = training_data.node_states[source_node]
        target_atoms = training_data.node_atoms[target_node]

        if source_states is None or target_atoms is None:
            continue

        # Ensure source_states is a list
        if not isinstance(source_states, list):
            source_states = [source_states]

        successes = 0
        lengths = []

        for _ in range(num_test_rollouts):
            # Randomly sample a source state from the available states
            source_state = random.choice(source_states)

            # Execute shortcut once using helper function
            success, num_steps, path = execute_shortcut_once(
                policy=policy,
                system=system,
                start_state=source_state,
                goal_atoms=target_atoms,
                max_steps=max_steps,
                source_node_id=source_node,
                target_node_id=target_node,
            )

            if success:
                successes += 1
                lengths.append(num_steps)
                successful_shortcut_paths.append(path)

        success_rate = successes / num_test_rollouts
        avg_length = np.mean(lengths) if lengths else max_steps

        results.append({'source_node': source_node,
                        'target_node': target_node,
                        'success_rate': success_rate,
                        'avg_length': avg_length})

        success_counts[(source_node, target_node)] = success_rate
        length_stats[(source_node, target_node)] = avg_length

        # Print progress every 5 shortcuts
        if (idx + 1) % 5 == 0 or (idx + 1) == len(training_data.unique_shortcuts):
            print(f"  Tested {idx + 1}/{len(training_data.unique_shortcuts)} shortcuts...")

    # Print detailed results
    print("Mapping:", training_data.node_atoms)
    print("\nDetailed Results:")
    print("=" * 80)
    for (source_node, target_node), success_rate in success_counts.items():
        avg_length = length_stats[(source_node, target_node)]
        print(f"  Shortcut {source_node}->{target_node}: "
              f"success={success_rate:.1%}, avg_length={avg_length:.1f}")

    # Summary statistics
    if success_counts:
        overall_success = np.mean(list(success_counts.values()))
        overall_length = np.mean(list(length_stats.values()))
        num_successful = sum(1 for sr in success_counts.values() if sr > 0.5)
        print("\n" + "=" * 80)
        print(f"Overall Statistics:")
        print(f"  Average success rate: {overall_success:.1%}")
        print(f"  Average length (when successful): {overall_length:.1f} steps")
        print(f"  Shortcuts with >50% success: {num_successful}/{len(success_counts)}")
        print("=" * 80)

    full_results = {'pairs': results, 'avg_success_rate': overall_success, 'avg_steps': overall_length, 'successful_shortcut_paths': successful_shortcut_paths}
    
    return full_results




# =============================================================================
# Stage 5: Evaluate
# =============================================================================


def evaluate_approach(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach,
    cfg: DictConfig,
) -> Metrics:
    """Stage 5: Evaluate the trained approach.

    Args:
        system: TAMP system
        approach: Trained approach
        config: Configuration dictionary
        num_eval_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("STAGE 5: EVALUATE")
    print("=" * 80)

    num_eval_episodes = cfg.evaluation.num_episodes

    print(f"Running {num_eval_episodes} evaluation episodes...")

    # Create TrainingConfig for evaluation
    eval_config = TrainingConfig(
        render=cfg.evaluation.render,
        eval_max_steps=cfg.evaluation.max_episode_steps,
    )

    # Run evaluations
    rewards = []
    lengths = []
    successes = []
    all_episode_data = []

    for ep in range(num_eval_episodes):
        if (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{num_eval_episodes} episodes")

        reward, length, success, episode_data = run_evaluation_episode(
            system=system,
            approach=approach,
            policy_name="MultiRL",
            config=eval_config,
            episode_num=ep,
        )
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)
        all_episode_data.append(episode_data)

    # Create Metrics object
    avg_metrics = Metrics(
        success_rate=sum(successes) / len(successes) if successes else 0.0,
        avg_episode_length=sum(lengths) / len(lengths) if lengths else 0.0,
        avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        episode_data=all_episode_data,
    )

    print("\nEvaluation complete:")
    print(f"  Success rate: {avg_metrics.success_rate:.2%}")
    print(f"  Avg steps: {avg_metrics.avg_episode_length:.1f}")
    print(f"  Avg reward: {avg_metrics.avg_reward:.3f}s")

    return avg_metrics


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    cfg: DictConfig,
) -> PipelineResults:
    """Run the complete SLAP pipeline.

    Args:
        system: TAMP system
        config: Configuration dictionary with all parameters including:
            - heuristic_type: Type of heuristic to use ("rollouts", "v4", etc.)
            - policy_type: Type of policy to use ("multiRL")
            - heuristic: Nested config for heuristic-specific parameters
            - rl_*: Parameters for RL policy
        num_eval_episodes: Number of evaluation episodes

    Returns:
        PipelineResults with all outputs from the pipeline
    """
    results = PipelineResults()

    # Setup RNG
    seed = cfg.seed
    rng = np.random.Generator(np.random.PCG64(seed))
    set_torch_seed(seed)
    random.seed(seed)

    if cfg.wandb_enabled:
        wandb_run_name = os.getenv("WANDB_RUN_NAME", None)
        wandb.init(project="slap_gridworld_fixed", config=OmegaConf.to_container(cfg, resolve=True), name=wandb_run_name)

    # Create policy instance based on config
    policy = create_policy(
        cfg=cfg
    )

    # Create approach (wraps system + policy)
    approach = ImprovisationalTAMPApproach(system, policy, seed=seed,
                                           planner_id=cfg.collection.planner_id,
                                           max_skill_steps=cfg.collection.max_steps_per_edge)

    times = {}
    # Stage 1: Collect training data
    start = time.time()
    results.training_data, results.graph_distances = collect_training_data(
        system=system,
        approach=approach,
        cfg=cfg,
        rng=rng,
    )
    times['collection_time'] = time.time() - start

    # Create heuristic instance based on config
    start = time.time()
    # Retrieve heuristic_config here to ensure it's available for conditional WandB logging
    if cfg.heuristic.type == "crl":
        heuristic_config = dataclass_from_cfg(CRLHeuristicConfig, cfg.heuristic)
    elif cfg.heuristic.type == "dqn":
        heuristic_config = dataclass_from_cfg(DQNHeuristicConfig, cfg.heuristic.dqn)
    elif cfg.heuristic.type == "cmd":
        heuristic_config = dataclass_from_cfg(CMDHeuristicConfig, cfg.heuristic)
    else:
        heuristic_config = None

    heuristic = create_heuristic(
        training_data=results.training_data,
        graph_distances=results.graph_distances,
        system=system,
        cfg=cfg,
        rng=rng
    )


    # Stage 2: Train heuristic
    results.heuristic_training_history = train_heuristic(
        heuristic=heuristic,
    )

    results.heuristic = heuristic

    times['heuristic_training_time'] = time.time() - start

    # Stage 2.5: Test heuristic quality (optional)
    if cfg.debug:
        print(f"[DEBUG] training_data.graph.nodes (ids): {[node.id for node in results.training_data.graph.nodes]}")
        print(f"[DEBUG] training_data.unique_shortcuts: {results.training_data.unique_shortcuts}")
        results.heuristic_quality_results = test_heuristic_quality(
            system=system,
            heuristic=heuristic,
            training_data=results.training_data,
            graph_distances=results.graph_distances,
            cfg=cfg,
            times=times,
        )

    if cfg.eval_heuristic_only:
        results.approach = approach
        results.times = times
        return results


    # Stage 3: Prune with heuristic
    start = time.time()
    results.pruned_training_data = prune_with_heuristic(
        heuristic=heuristic,
        max_shortcuts=cfg.heuristic.max_shortcuts_per_graph,
    )
    times['heuristic_pruning_time'] = time.time() - start

    # # Stage 3.5: Random selection
    # max_shortcuts = cfg.heuristic.max_shortcuts_per_graph
    # if max_shortcuts != float("inf"):
    #     results.final_training_data = random_selection(
    #         training_data=results.pruned_training_data,
    #         max_shortcuts=int(max_shortcuts),
    #         rng=rng,
    #     )
    # else:
    #     results.final_training_data = results.pruned_training_data


    # Stage 4: Train policy
    start = time.time()
    train_policy(
        system=system,
        policy=policy,
        training_data=results.pruned_training_data,
        cfg=cfg,
    )

    results.policy = policy
    times["policy_training_time"] = time.time() - start

    # Stage 4.5: Test shortcut quality (optional)
    if cfg.debug and len(results.pruned_training_data.valid_shortcuts) > 0:
        results.shortcut_quality_results = test_shortcut_quality(
            system=system,
            policy=policy,
            training_data=results.pruned_training_data,
            cfg=cfg,
        )

    # Stage 5: Evaluate
    start = time.time()
    results.evaluation_results = evaluate_approach(
        system=system,
        approach=approach,
        cfg=cfg,
    )
    times["evaluation_time"] = time.time() - start

    results.approach = approach
    results.times = times

    if cfg.wandb_enabled:
        wandb.finish()

    return results
