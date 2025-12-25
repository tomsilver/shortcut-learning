"""Modular pipeline for SLAP training with caching and cascade invalidation."""

import hashlib
import json
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.distance_heuristic_v4 import (
        DistanceHeuristicV4,
    )

from tamp_improv.approaches.improvisational.base import (
    ImprovisationalTAMPApproach,
    ShortcutSignature,
)
from tamp_improv.approaches.improvisational.collection import collect_all_shortcuts, collect_total_shortcuts
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
    Policy,
    TrainingData,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.pruning import (
    prune_training_data,
    train_distance_heuristic,
)
from tamp_improv.approaches.improvisational.training import (
    Metrics,
    TrainingConfig,
    run_evaluation_episode_with_caching,
    run_evaluation_episode
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.utils.gpu_utils import set_torch_seed

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


# =============================================================================
# Helper Functions
# =============================================================================


def _hash_config(config: dict[str, Any], keys: list[str]) -> str:
    """Hash specific config keys for fingerprinting."""
    filtered = {k: config.get(k) for k in keys if k in config}
    config_str = json.dumps(filtered, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def _configs_match(
    current_config: dict[str, Any],
    saved_config: dict[str, Any],
    keys: list[str],
) -> bool:
    """Check if relevant config keys match between current and saved."""
    for key in keys:
        if current_config.get(key) != saved_config.get(key):
            return False
    return True


def _invalidate_downstream_stages(save_dir: Path, stage: str) -> None:
    """Delete cached artifacts for stages after 'stage'."""
    stage_order = ["collection", "heuristic", "pruning", "policy"]
    try:
        invalidate_from = stage_order.index(stage) + 1
    except ValueError:
        return

    stage_dirs = {
        "heuristic": save_dir / "distance_heuristic",
        "pruning": save_dir / "pruned_training_data",
        "policy": save_dir / "trained_policy",
    }

    for downstream_stage in stage_order[invalidate_from:]:
        cache_path = stage_dirs.get(downstream_stage)
        if cache_path and cache_path.exists():
            print(f"  Invalidating {downstream_stage} cache due to {stage} change")
            shutil.rmtree(cache_path)


def _test_shortcut_quality(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy: Policy,
    training_data: GoalConditionedTrainingData,
    config: dict[str, Any],
) -> None:
    """Test quality of trained shortcuts by running rollouts.

    Args:
        system: The TAMP system
        policy: Trained policy
        training_data: Training data with shortcuts
        config: Config with max_steps
    """
    from tamp_improv.approaches.improvisational.analyze import execute_shortcut_once

    print(f"\nTesting {len(training_data.valid_shortcuts)} shortcuts...")
    print("=" * 80)

    max_steps = config.get("max_steps_per_rollout", 100)
    num_test_rollouts = 10  # Test each shortcut multiple times

    success_counts = {}
    length_stats = {}

    for idx, (source_node, target_node) in enumerate(training_data.valid_shortcuts):
        source_states = training_data.node_states.get(source_node)
        target_atoms = training_data.node_atoms.get(target_node)

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
            success, num_steps = execute_shortcut_once(
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

        success_rate = successes / num_test_rollouts
        avg_length = np.mean(lengths) if lengths else max_steps

        success_counts[(source_node, target_node)] = success_rate
        length_stats[(source_node, target_node)] = avg_length

        # Print progress every 5 shortcuts
        if (idx + 1) % 5 == 0 or (idx + 1) == len(training_data.valid_shortcuts):
            print(f"  Tested {idx + 1}/{len(training_data.valid_shortcuts)} shortcuts...")

    # Print detailed results
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


def _test_heuristic_quality(
    heuristic,  # DistanceHeuristicV4
    training_data: GoalConditionedTrainingData,
    planning_graph: PlanningGraph,
) -> None:
    """Test quality of trained heuristic by comparing with graph distances.

    Args:
        heuristic: Trained distance heuristic (V4)
        training_data: Training data with node states and atoms
        planning_graph: Planning graph for computing true graph distances
    """
    from tamp_improv.approaches.improvisational.graph_training import (
        compute_graph_distances,
    )

    print("\n" + "=" * 80)
    print("STAGE 2.5: Evaluating Distance Heuristic Quality")
    print("=" * 80)

    # Compute graph distances (ground truth)
    print("Computing graph distances (ground truth)...")
    graph_distances = compute_graph_distances(planning_graph, exclude_shortcuts=True)
    print(f"  Computed {len(graph_distances)} pairwise distances")

    # Use V4's estimate_node_distance for node-node pairs
    print("Estimating node-node distances with learned heuristic...")
    results = []
    for source_node in planning_graph.nodes:
        for target_node in planning_graph.nodes:
            if source_node.id == target_node.id:
                continue

            # Check if we have states for both nodes
            if (
                source_node.id not in training_data.node_states
                or target_node.id not in training_data.node_states
            ):
                continue

            source_states = training_data.node_states[source_node.id]
            target_states = training_data.node_states[target_node.id]

            if not source_states or not target_states:
                continue

            # Get graph distance
            graph_dist = graph_distances.get(
                (source_node.id, target_node.id), float("inf")
            )

            # Use V4's estimate_node_distance for node-node estimation
            learned_dist = heuristic.estimate_node_distance(source_node.id, target_node.id)

            results.append(
                {
                    "source_id": source_node.id,
                    "target_id": target_node.id,
                    "graph_distance": graph_dist,
                    "learned_distance": learned_dist,
                }
            )

    print(f"  Evaluated {len(results)} node-node pairs")

    # Separate finite and infinite distance pairs
    finite_results = [r for r in results if r["graph_distance"] != float("inf")]
    infinite_results = [r for r in results if r["graph_distance"] == float("inf")]

    # Compute statistics
    print("\nHeuristic Quality Analysis:")
    print("=" * 80)
    print(f"Total node pairs: {len(results)}")
    print(f"  Pairs with finite graph distance: {len(finite_results)}")
    print(f"  Pairs with infinite graph distance: {len(infinite_results)}")

    if finite_results:
        graph_dists = np.array([r["graph_distance"] for r in finite_results])
        learned_dists = np.array([r["learned_distance"] for r in finite_results])

        mae = np.mean(np.abs(graph_dists - learned_dists))
        rmse = np.sqrt(np.mean((graph_dists - learned_dists) ** 2))
        correlation = np.corrcoef(graph_dists, learned_dists)[0, 1]
        relative_errors = np.abs(graph_dists - learned_dists) / (graph_dists + 1e-6)
        mean_relative_error = np.mean(relative_errors)

        print(f"\nFinite Distance Pairs ({len(finite_results)} pairs):")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  Correlation: {correlation:.3f}")
        print(f"  Mean Relative Error: {mean_relative_error:.2%}")
        print(f"  Graph distance range: [{graph_dists.min():.1f}, {graph_dists.max():.1f}]")
        print(f"  Learned distance range: [{learned_dists.min():.1f}, {learned_dists.max():.1f}]")

        # Show sample comparisons
        sorted_finite = sorted(finite_results, key=lambda r: r["graph_distance"])
        print("\nSample Comparisons (first 10 pairs):")
        d_label = "D(s,s')"
        f_label = "f(s,s')"
        print(f"{'Source':>6} -> {'Target':>6} | {d_label:>10} | {f_label:>10} | {'Error':>8}")
        print("-" * 60)
        for r in sorted_finite[:10]:
            error = abs(r["graph_distance"] - r["learned_distance"])
            print(
                f"{r['source_id']:>6} -> {r['target_id']:>6} | "
                f"{r['graph_distance']:>10.1f} | "
                f"{r['learned_distance']:>10.1f} | "
                f"{error:>8.1f}"
            )

    if infinite_results:
        learned_dists_inf = [r["learned_distance"] for r in infinite_results]
        print(f"\nInfinite Distance Pairs ({len(infinite_results)} pairs):")
        print(f"  These pairs have no non-shortcut path (are true shortcuts)")
        print(f"  Learned distance range: [{min(learned_dists_inf):.1f}, {max(learned_dists_inf):.1f}]")

    print("=" * 80)


# =============================================================================
# Pipeline Stages
# =============================================================================


def get_or_collect_full_data(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    config: dict[str, Any],
    save_dir: Path,
    rng: np.random.Generator,
    force_collect: bool = False,
) -> GoalConditionedTrainingData:
    """
    Stage 1: Get or collect full training data (all shortcuts, no pruning).

    Args:
        system: The TAMP system
        approach: The improvisational approach
        config: Full config dictionary
        save_dir: Directory to save/load data
        rng: Random number generator
        force_collect: Force re-collection even if cached data exists

    Returns:
        GoalConditionedTrainingData with all collected shortcuts
    """
    collection_dir = save_dir / "full_training_data"
    data_path = collection_dir / "data.pkl"
    config_path = collection_dir / "config.json"

    # Config keys that affect collection
    collection_config_keys = [
        "seed",
        "collect_episodes",
        "use_random_rollouts",
        "num_rollouts_per_node",
        "max_steps_per_rollout",
        "shortcut_success_threshold",
    ]

    # Try to load existing data
    if not force_collect and data_path.exists() and config_path.exists():
        print(f"\nChecking cached training data at {collection_dir}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            if _configs_match(config, saved_config, collection_config_keys):
                print("  Config matches! Loading cached data...")
                with open(data_path, "rb") as f:
                    full_data = pickle.load(f)
                print(f"  Loaded {len(full_data.valid_shortcuts)} shortcuts")
                return full_data
            else:
                print("  Config mismatch - will re-collect")
                # Invalidate all downstream stages
                _invalidate_downstream_stages(save_dir, "collection")
        except Exception as e:
            print(f"  Error loading cached data: {e}")
            print("  Will re-collect")

    # Collect new data using collect_all_shortcuts (no pruning)
    print(f"\nCollecting full training data...")
    full_data = collect_total_shortcuts(system, approach, config, rng)

    # Save data and config
    collection_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving training data to {collection_dir}")
    with open(data_path, "wb") as f:
        pickle.dump(full_data, f)

    # Save relevant config
    saved_config = {k: config.get(k) for k in collection_config_keys}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2)

    print(f"  Saved {len(full_data.valid_shortcuts)} shortcuts")
    return full_data


def get_or_train_heuristic(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    training_data: GoalConditionedTrainingData,
    config: dict[str, Any],
    save_dir: Path,
    rng: np.random.Generator,
    force_train: bool = False,
):
    """
    Stage 2: Get or train distance heuristic using V4 with multi-train.

    Args:
        system: The TAMP system
        training_data: Full collected training data
        config: Full config dictionary
        save_dir: Directory to save/load heuristic
        rng: Random number generator
        force_train: Force re-training even if cached heuristic exists

    Returns:
        Trained DistanceHeuristicV4
    """
    from tamp_improv.approaches.improvisational.distance_heuristic_v4 import (
        DistanceHeuristicV4,
        DistanceHeuristicV4Config,
    )
    from tamp_improv.approaches.improvisational.graph_training import (
        compute_graph_distances,
    )

    heuristic_dir = save_dir / "distance_heuristic_v4"
    config_path = heuristic_dir / "config.json"

    # Config keys that affect heuristic training
    heuristic_config_keys = [
        "seed",
        "latent_dim",
        "hidden_dims",
        "learning_rate",
        "batch_size",
        "buffer_size",
        "gamma",
        "num_rounds",
        "keep_fraction",
        "num_epochs",
        "trajectories_per_epoch",
        "max_episode_steps",
    ]

    # Collection fingerprint (to detect if upstream data changed)
    collection_config_keys = [
        "seed",
        "collect_episodes",
        "use_random_rollouts",
        "num_rollouts_per_node",
    ]
    collection_fingerprint = _hash_config(config, collection_config_keys)

    # Try to load existing heuristic
    if not force_train and heuristic_dir.exists() and config_path.exists():
        print(f"\nChecking cached heuristic at {heuristic_dir}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            # Check if both heuristic config and collection fingerprint match
            config_matches = _configs_match(config, saved_config, heuristic_config_keys)
            fingerprint_matches = (
                saved_config.get("collection_fingerprint") == collection_fingerprint
            )

            if config_matches and fingerprint_matches:
                print("  Config and upstream data match! Loading cached heuristic...")
                heuristic_config = DistanceHeuristicV4Config(
                    latent_dim=config.get("latent_dim", 32),
                    hidden_dims=config.get("hidden_dims"),
                    learning_rate=config.get("learning_rate", 0.001),
                    batch_size=config.get("batch_size", 256),
                    buffer_size=config.get("buffer_size", 10000),
                    gamma=config.get("gamma", 0.99),
                    num_rounds=config.get("num_rounds", 1),
                    keep_fraction=config.get("keep_fraction", 0.5),
                    device=config.get("device", "cuda"),
                )
                heuristic = DistanceHeuristicV4(
                    config=heuristic_config,
                    env=system.env,
                    get_node=system.perceiver.step,
                    node_atoms=training_data.node_atoms,
                )
                heuristic.load(str(heuristic_dir))
                print("  Heuristic loaded successfully")
                return heuristic
            else:
                if not config_matches:
                    print("  Heuristic config mismatch - will retrain")
                if not fingerprint_matches:
                    print("  Collection data changed - will retrain")
                # Invalidate downstream stages
                _invalidate_downstream_stages(save_dir, "heuristic")
        except Exception as e:
            print(f"  Error loading cached heuristic: {e}")
            print("  Will retrain")

    # Train new heuristic using V4 with multi_train
    print("Training distance heuristic V4 with multi-round training...")

    # Prepare state-node pairs from training data
    state_node_pairs = []
    for source_id, target_id in training_data.valid_shortcuts:
        if source_id not in training_data.node_states:
            continue

        source_states = training_data.node_states[source_id]
        if not source_states:
            continue

        # Use all source states with target node
        for source_state in source_states:
            state_node_pairs.append((source_state, target_id))

    print(f"  Total state-node pairs: {len(state_node_pairs)}")

    # Compute graph distances for multi-round training
    planning_graph = training_data.graph
    if planning_graph is None:
        raise ValueError("Planning graph required for heuristic training")

    graph_distances = {}
    if config.get("num_rounds", 1) > 1:
        print("  Computing graph distances for pruning...")
        graph_distances = compute_graph_distances(planning_graph, exclude_shortcuts=True)
        print(f"  Computed {len(graph_distances)} pairwise graph distances")

    # Create heuristic config
    heuristic_config = DistanceHeuristicV4Config(
        latent_dim=config.get("latent_dim", 32),
        hidden_dims=config.get("hidden_dims"),
        learning_rate=config.get("learning_rate", 0.001),
        batch_size=config.get("batch_size", 256),
        buffer_size=config.get("buffer_size", 10000),
        gamma=config.get("gamma", 0.99),
        num_rounds=config.get("num_rounds", 1),
        keep_fraction=config.get("keep_fraction", 0.5),
        device=config.get("device", "cuda"),
    )

    # Initialize heuristic
    heuristic = DistanceHeuristicV4(
        config=heuristic_config,
        env=system.env,
        get_node=system.perceiver.step,
        node_atoms=training_data.node_atoms,
    )

    # Train using multi_train
    training_history = heuristic.multi_train(
        state_node_pairs=state_node_pairs,
        graph_distances=graph_distances,
        num_rounds=heuristic_config.num_rounds,
        num_epochs_per_round=config.get("num_epochs", 200),
        trajectories_per_epoch=config.get("trajectories_per_epoch", 10),
        max_episode_steps=config.get("max_episode_steps", 100),
        keep_fraction=heuristic_config.keep_fraction,
    )

    print("  Heuristic training complete")

    # Save heuristic and config
    heuristic_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving heuristic to {heuristic_dir}")
    heuristic.save(str(heuristic_dir))

    # Save distance matrices from multi-round training (if any)
    if 'distance_matrices' in training_history and len(training_history['distance_matrices']) > 0:
        import pickle
        distance_matrices_path = heuristic_dir / "distance_matrices.pkl"
        with open(distance_matrices_path, 'wb') as f:
            pickle.dump({
                'distance_matrices': training_history['distance_matrices'],
                'graph_distances': training_history.get('graph_distances', {}),
            }, f)
        print(f"  Saved {len(training_history['distance_matrices'])} distance matrices")

    # Save config with collection fingerprint
    saved_config = {k: config.get(k) for k in heuristic_config_keys}
    saved_config["collection_fingerprint"] = collection_fingerprint
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2)

    print("  Heuristic saved successfully")
    return heuristic


def prune_full_data(
    training_data: GoalConditionedTrainingData,
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    planning_graph: PlanningGraph,
    config: dict[str, Any],
    save_dir: Path,
    rng: np.random.Generator,
    heuristic: "DistanceHeuristicV4 | None" = None,
    force_prune: bool = False,
) -> GoalConditionedTrainingData:
    """
    Stage 3: Apply pruning method to full training data.

    Args:
        training_data: Full collected training data
        system: The TAMP system
        planning_graph: The planning graph
        config: Full config dictionary
        save_dir: Directory to save/load pruned data
        rng: Random number generator
        heuristic: Trained distance heuristic V4 (required if using distance_heuristic pruning)
        force_prune: Force re-pruning even if cached data exists

    Returns:
        Pruned GoalConditionedTrainingData
    """
    pruning_method = config.get("pruning_method", "none")
    pruning_dir = save_dir / f"pruned_data_{pruning_method}"
    data_path = pruning_dir / "data.pkl"
    config_path = pruning_dir / "config.json"

    # Config keys that affect pruning
    pruning_config_keys = ["seed", "pruning_method"]

    # Add method-specific config keys
    if pruning_method == "random":
        pruning_config_keys.extend(["num_shortcuts_to_keep"])
    elif pruning_method == "rollouts":
        pruning_config_keys.extend(
            [
                "num_rollouts_per_node",
                "max_steps_per_rollout",
                "shortcut_success_threshold",
            ]
        )
    elif pruning_method == "distance_heuristic":
        pruning_config_keys.extend(["heuristic_practical_horizon"])

    # Heuristic fingerprint (to detect if upstream heuristic changed)
    heuristic_fingerprint = None
    if pruning_method == "distance_heuristic":
        heuristic_config_keys = [
            "seed",
            "heuristic_training_pairs",
            "heuristic_training_steps",
            "heuristic_learning_rate",
        ]
        heuristic_fingerprint = _hash_config(config, heuristic_config_keys)

    # Try to load existing pruned data
    if not force_prune and data_path.exists() and config_path.exists():
        print(f"\nChecking cached pruned data at {pruning_dir}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            # Check if pruning config matches
            config_matches = _configs_match(config, saved_config, pruning_config_keys)

            # Check if heuristic fingerprint matches (if applicable)
            fingerprint_matches = True
            if heuristic_fingerprint is not None:
                fingerprint_matches = (
                    saved_config.get("heuristic_fingerprint") == heuristic_fingerprint
                )

            if config_matches and fingerprint_matches:
                print("  Config and upstream data match! Loading cached pruned data...")
                with open(data_path, "rb") as f:
                    pruned_data = pickle.load(f)
                print(f"  Loaded {len(pruned_data.valid_shortcuts)} pruned shortcuts")
                return pruned_data
            else:
                if not config_matches:
                    print("  Pruning config mismatch - will re-prune")
                if not fingerprint_matches:
                    print("  Heuristic changed - will re-prune")
                # Invalidate downstream stages
                _invalidate_downstream_stages(save_dir, "pruning")
        except Exception as e:
            print(f"  Error loading cached pruned data: {e}")
            print("  Will re-prune")

    # Apply pruning using the existing prune_training_data function
    print(f"\nApplying pruning method: {pruning_method}")

    if pruning_method == "distance_heuristic" and heuristic is None:
        raise ValueError(
            "Distance heuristic pruning requires a trained heuristic. "
            "Call get_or_train_heuristic() first."
        )

    pruned_data = prune_training_data(
        training_data, system, planning_graph, config, rng, heuristic=heuristic
    )

    # Save pruned data and config
    pruning_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving pruned data to {pruning_dir}")
    with open(data_path, "wb") as f:
        pickle.dump(pruned_data, f)

    # Save config with heuristic fingerprint (if applicable)
    saved_config = {k: config.get(k) for k in pruning_config_keys}
    if heuristic_fingerprint is not None:
        saved_config["heuristic_fingerprint"] = heuristic_fingerprint
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2)

    print(f"  Saved {len(pruned_data.valid_shortcuts)} pruned shortcuts")
    return pruned_data


def get_or_train_policy(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy: Policy[ObsType, ActType],
    training_data: TrainingData,
    config: dict[str, Any],
    save_dir: Path,
    policy_name: str = "MultiRL",
    force_train: bool = False,
) -> Policy[ObsType, ActType]:
    """
    Stage 4: Get or train policy.

    Args:
        system: The TAMP system
        policy: The policy instance to train
        training_data: Pruned training data
        config: Full config dictionary
        save_dir: Directory to save/load policy
        policy_name: Name of the policy for saving
        force_train: Force re-training even if cached policy exists

    Returns:
        Trained policy
    """
    policy_dir = save_dir / "trained_policy"
    policy_path = policy_dir / f"{system.name}_{policy_name}"
    config_path = policy_dir / "config.json"

    # Config keys that affect policy training
    policy_config_keys = [
        "seed",
        "max_training_steps_per_shortcut",
        "learning_rate",
        "rl_batch_size",
        "n_epochs",
        "gamma",
        "ent_coef",
    ]

    # Pruning fingerprint (to detect if upstream pruning changed)
    pruning_method = config.get("pruning_method", "none")
    pruning_config_keys = ["seed", "pruning_method"]
    if pruning_method == "distance_heuristic":
        pruning_config_keys.extend(["heuristic_practical_horizon"])
    pruning_fingerprint = _hash_config(config, pruning_config_keys)

    # Try to load existing policy
    if not force_train and policy_path.exists() and config_path.exists():
        print(f"\nChecking cached policy at {policy_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            # Check if policy config and pruning fingerprint match
            config_matches = _configs_match(config, saved_config, policy_config_keys)
            fingerprint_matches = (
                saved_config.get("pruning_fingerprint") == pruning_fingerprint
            )

            if config_matches and fingerprint_matches:
                print("  Config and upstream data match! Loading cached policy...")
                policy.load(str(policy_path))
                print("  Policy loaded successfully")
                return policy
            else:
                if not config_matches:
                    print("  Policy config mismatch - will retrain")
                if not fingerprint_matches:
                    print("  Pruning changed - will retrain")
        except Exception as e:
            print(f"  Error loading cached policy: {e}")
            print("  Will retrain")

    # Train new policy
    print(f"\nTraining policy...")

    # Configure environment for training
    if hasattr(system.wrapped_env, "configure_training"):
        system.wrapped_env.configure_training(training_data)

    # Train policy
    if isinstance(policy, MultiRLPolicy):
        policy.train(system.wrapped_env, training_data, save_dir=str(policy_path))
    else:
        policy.train(system.wrapped_env, training_data)

    # Save policy and config
    policy_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving policy to {policy_path}")
    policy.save(str(policy_path))

    # Save config with pruning fingerprint
    saved_config = {k: config.get(k) for k in policy_config_keys}
    saved_config["pruning_fingerprint"] = pruning_fingerprint
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2)

    print("  Policy saved successfully")
    return policy


# =============================================================================
# Pipeline Orchestrator
# =============================================================================


def train_and_evaluate_with_pipeline(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: dict[str, Any],
    save_dir: Path,
    policy_name: str = "MultiRL",
    num_eval_episodes: int = 50,
) -> Metrics:
    """Train and evaluate using the modular pipeline with caching.

    This function uses the new modular pipeline:
    1. get_or_collect_full_data - Collect all shortcuts
    2. get_or_train_heuristic - Train distance heuristic (if using distance_heuristic pruning)
    3. prune_full_data - Apply pruning method
    4. get_or_train_policy - Train policy on pruned data
    5. Evaluate trained policy

    Each stage caches its results and tracks upstream dependencies, automatically
    invalidating downstream caches when needed.

    Args:
        system: The TAMP system
        policy_factory: Factory function to create policy instances
        config: Full configuration dictionary
        save_dir: Root directory for saving all pipeline artifacts
        policy_name: Name of the policy for saving
        num_eval_episodes: Number of evaluation episodes

    Returns:
        Metrics with training and evaluation results
    """
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)
    set_torch_seed(seed)

    start_time = time.time()
    stage_times = {}  # Track time for each stage

    # Create policy and approach
    policy = policy_factory(seed)
    approach = ImprovisationalTAMPApproach(system, policy, seed=seed)

    print("\n" + "=" * 80)
    print("MODULAR PIPELINE: Training with caching and cascade invalidation")
    print("=" * 80)

    # Stage 1: Get or collect full training data
    print("\n" + "=" * 80)
    print("STAGE 1: Collection")
    print("=" * 80)
    stage_start = time.time()
    full_data = get_or_collect_full_data(
        system,
        approach,
        config,
        save_dir,
        rng,
        force_collect=config.get("force_collect", False),
    )
    stage_times["collection"] = time.time() - stage_start
    print(f"  Stage 1 completed in {stage_times['collection']:.1f}s")

    # Get planning graph (needed for pruning)
    planning_graph = full_data.graph
    if planning_graph is None:
        raise ValueError("Planning graph is required for pruning but not found in training data")

    # Stage 2: Get or train heuristic (if using distance_heuristic pruning)
    heuristic = None
    pruning_method = config.get("pruning_method", "none")
    if pruning_method == "distance_heuristic":
        print("\n" + "=" * 80)
        print("STAGE 2: Distance Heuristic Training")
        print("=" * 80)
        stage_start = time.time()
        heuristic = get_or_train_heuristic(
            system,
            full_data,
            config,
            save_dir,
            rng,
            force_train=config.get("force_train_heuristic", False),
        )
        stage_times["heuristic"] = time.time() - stage_start
        print(f"  Stage 2 completed in {stage_times['heuristic']:.1f}s")

        # Stage 2.5: Test heuristic quality (debug mode only)
        if config.get("debug", False):
            print("\n" + "=" * 80)
            print("STAGE 2.5: Testing Heuristic Quality")
            print("=" * 80)
            _test_heuristic_quality(heuristic, full_data, planning_graph)
    else:
        stage_times["heuristic"] = 0.0

    # Stage 3: Prune data
    print("\n" + "=" * 80)
    print("STAGE 3: Pruning")
    print("=" * 80)
    stage_start = time.time()
    pruned_data = prune_full_data(
        full_data,
        system,
        planning_graph,
        config,
        save_dir,
        rng,
        heuristic=heuristic,
        force_prune=config.get("force_prune", False),
    )
    stage_times["pruning"] = time.time() - stage_start
    print(f"  Stage 3 completed in {stage_times['pruning']:.1f}s")

    # Stage 4: Get or train policy
    print("\n" + "=" * 80)
    print("STAGE 4: Policy Training")
    print("=" * 80)
    stage_start = time.time()

    print("Force train:", config['force_train_policy'])

    # Update training_data config with current training parameters
    # (these may have been overridden since collection/pruning stages)
    training_params = [
        "max_training_steps_per_shortcut",
        "episodes_per_scenario",
        "learning_rate",
        "rl_batch_size",
        "n_epochs",
        "gamma",
        "ent_coef",
        "early_stopping",
    ]
    for param in training_params:
        if param in config:
            pruned_data.config[param] = config[param]

    policy = get_or_train_policy(
        system,
        policy,
        pruned_data,
        config,
        save_dir,
        policy_name,
        force_train=config.get("force_train_policy", False),
    )
    stage_times["policy_training"] = time.time() - stage_start
    print(f"  Stage 4 completed in {stage_times['policy_training']:.1f}s")

    # Note: Signatures are now registered during collection (in collect_all_shortcuts)
    # so we don't need to register them again here
    print(f"Using {len(approach.trained_signatures)} trained shortcut signatures from collection")

    training_time = time.time() - start_time

    # Stage 4.5: Test shortcut quality (optional, controlled by debug flag)
    if config.get("debug", False):
        print("\n" + "=" * 80)
        print("STAGE 4.5: Testing Shortcut Quality")
        print("=" * 80)
        _test_shortcut_quality(system, policy, pruned_data, config)

    # Stage 5: Evaluate
    print("\n" + "=" * 80)
    print("STAGE 5: Evaluation")
    print("=" * 80)
    print(f"Evaluating policy on {num_eval_episodes} episodes...")

    # Create TrainingConfig for evaluation (filter to only valid fields)
    from dataclasses import fields
    training_config_fields = {f.name for f in fields(TrainingConfig)}
    filtered_config = {k: v for k, v in config.items() if k in training_config_fields}
    eval_config = TrainingConfig(**filtered_config)

    stage_start = time.time()
    rewards = []
    lengths = []
    successes = []

    for episode in range(num_eval_episodes):
        reward, length, success = run_evaluation_episode(
            system, approach, policy_name, eval_config, episode
        )
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)
        print(f"  Episode {episode + 1}: reward={reward:.2f}, length={length}, success={success}")

        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_eval_episodes} episodes")

    stage_times["evaluation"] = time.time() - stage_start
    print(f"  Stage 5 completed in {stage_times['evaluation']:.1f}s")

    success_rate = sum(successes) / len(successes)
    avg_length = sum(lengths) / len(lengths)
    avg_reward = sum(rewards) / len(rewards)
    total_time = time.time() - start_time

    print(f"\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Avg episode length: {avg_length:.1f}")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"\nStage Timing Breakdown:")
    print(f"  Stage 1 (Collection):     {stage_times['collection']:>8.1f}s")
    print(f"  Stage 2 (Heuristic):      {stage_times['heuristic']:>8.1f}s")
    print(f"  Stage 3 (Pruning):        {stage_times['pruning']:>8.1f}s")
    print(f"  Stage 4 (Policy Train):   {stage_times['policy_training']:>8.1f}s")
    print(f"  Stage 5 (Evaluation):     {stage_times['evaluation']:>8.1f}s")
    print(f"  Total Training Time:      {training_time:>8.1f}s")
    print(f"  Total Time (with eval):   {total_time:>8.1f}s")
    print("=" * 80)

    return Metrics(
        success_rate=success_rate,
        avg_episode_length=avg_length,
        avg_reward=avg_reward,
        training_time=training_time,
        total_time=total_time,
    )
