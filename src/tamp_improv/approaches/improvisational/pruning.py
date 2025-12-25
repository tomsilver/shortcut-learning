"""Pruning methods for shortcut selection in SLAP.

This module separates the pruning phase from data collection, allowing different
pruning strategies to be applied to the same collected training data.
"""

from typing import Any
from collections import defaultdict
import gymnasium as gym

import numpy as np

from tamp_improv.approaches.improvisational.distance_heuristic import (
    DistanceHeuristicConfig,
    GoalConditionedDistanceHeuristic,
)
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.approaches.improvisational.graph_training import (
    ShortcutCandidate,
    compute_graph_distances,
    identify_promising_shortcuts_with_rollouts,
)
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


def train_distance_heuristic(
    training_data: GoalConditionedTrainingData,
    system: ImprovisationalTAMPSystem,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> "GoalConditionedDistanceHeuristic":
    """Train a distance heuristic on the collected training data.

    Args:
        training_data: Full collected training data
        system: The TAMP system
        config: Configuration with heuristic training parameters
        rng: Random number generator

    Returns:
        Trained GoalConditionedDistanceHeuristic
    """
    from tamp_improv.approaches.improvisational.distance_heuristic import (
        DistanceHeuristicConfig,
        GoalConditionedDistanceHeuristic,
    )

    print("Training distance heuristic...")

    # Prepare all state pairs from training data
    all_state_pairs = []
    for source_id, target_id in training_data.valid_shortcuts:
        if source_id not in training_data.node_states:
            continue
        if target_id not in training_data.node_states:
            continue

        source_states = training_data.node_states[source_id]
        target_states = training_data.node_states[target_id]

        if not source_states or not target_states:
            continue

        # Use all combinations of states
        for source_state in source_states:
            for target_state in target_states:
                all_state_pairs.append((source_state, target_state))

    print(f"  Total available state pairs: {len(all_state_pairs)}")

    # Sample training pairs
    heuristic_training_pairs = config.get("heuristic_training_pairs", 100)
    num_train = min(heuristic_training_pairs, len(all_state_pairs))
    train_indices = rng.choice(len(all_state_pairs), size=num_train, replace=False)
    training_pairs = [all_state_pairs[i] for i in train_indices]
    print(f"  Selected {len(training_pairs)} pairs for training")

    # Create and train heuristic
    heuristic_config = DistanceHeuristicConfig(
        learning_rate=config.get("heuristic_learning_rate", 3e-4),
        batch_size=config.get("heuristic_batch_size", 256),
        buffer_size=config.get("heuristic_buffer_size", 100000),
        max_episode_steps=config.get("heuristic_max_steps", 200),
        device=config.get("device", "cuda"),
    )

    heuristic = GoalConditionedDistanceHeuristic(
        config=heuristic_config, seed=config.get("seed", 42)
    )

    heuristic_training_steps = config.get("heuristic_training_steps", 50000)
    heuristic.train(system.env, training_pairs, system.perceiver, heuristic_training_steps)

    print("  Heuristic training complete")
    return heuristic


def prune_training_data(
    training_data: GoalConditionedTrainingData,
    system: ImprovisationalTAMPSystem,
    planning_graph: PlanningGraph,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
    heuristic: "GoalConditionedDistanceHeuristic | None" = None,
) -> GoalConditionedTrainingData:
    """Prune training data based on configured pruning method.

    Args:
        training_data: Unpruned training data with all possible shortcuts
        system: The TAMP system (needed for rollout-based pruning)
        planning_graph: The planning graph (needed for distance computation)
        config: Configuration dictionary with pruning settings
        rng: Random number generator
        heuristic: Pre-trained heuristic (for distance_heuristic pruning)

    Returns:
        Pruned training data containing only selected shortcuts
    """
    if rng is None:
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

    pruning_method = config.get("pruning_method", "random")
    max_shortcuts = config.get("max_shortcuts_per_graph", 150)

    if pruning_method == "none":
        return prune_none(training_data)
    elif pruning_method == "random":
        return prune_random(training_data, max_shortcuts, rng)
    elif pruning_method == "rollouts":
        pruned_data = prune_with_rollouts(training_data, system, planning_graph, config, rng)
        return prune_random(pruned_data, max_shortcuts, rng)
    elif pruning_method == "distance_heuristic":
        pruned_data = prune_with_distance_heuristic(
            training_data, system, planning_graph, config, rng, heuristic=heuristic
        )
        return prune_random(pruned_data, max_shortcuts, rng)
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")


def prune_none(
    training_data: GoalConditionedTrainingData,
) -> GoalConditionedTrainingData:
    """No pruning - return all shortcuts.

    Args:
        training_data: Unpruned training data

    Returns:
        The same training data (no pruning applied)
    """
    print("Pruning method: none (keeping all shortcuts)")
    print(f"Total shortcuts: {len(training_data.states)}")
    return training_data


def prune_random(
    training_data: GoalConditionedTrainingData,
    max_shortcuts: int,
    rng: np.random.Generator,
) -> GoalConditionedTrainingData:
    """Random pruning - select a random subset of shortcuts.

    Operates at the NODE PAIR level: if a node pair (A, B) is selected,
    ALL training examples (state pairs) for that node pair are kept.

    Args:
        training_data: Unpruned training data
        max_shortcuts: Maximum number of NODE PAIRS to keep
        rng: Random number generator

    Returns:
        Pruned training data with randomly selected shortcuts
    """
    print(f"Pruning method: random (max {max_shortcuts} node pairs)")

    # Get unique node pairs
    unique_node_pairs = list(set(training_data.valid_shortcuts))
    num_node_pairs = len(unique_node_pairs)
    print(f"Total node pairs before pruning: {num_node_pairs}")
    print(f"Total state pairs: {len(training_data.states)}")

    if num_node_pairs <= max_shortcuts:
        print(f"Keeping all {num_node_pairs} node pairs (under limit)")
        return training_data

    # Randomly select node pairs
    selected_node_pairs = set(
        tuple(unique_node_pairs[i])
        for i in rng.choice(num_node_pairs, size=max_shortcuts, replace=False)
    )

    # Find all state pair indices that belong to selected node pairs
    selected_indices = []
    for i, (source_id, target_id) in enumerate(training_data.valid_shortcuts):
        if (source_id, target_id) in selected_node_pairs:
            selected_indices.append(i)

    # Filter shortcut_info to match selected indices
    original_shortcut_info = training_data.config.get("shortcut_info", [])
    pruned_shortcut_info = [original_shortcut_info[i] for i in selected_indices] if original_shortcut_info else []

    # Filter valid_shortcuts to only include selected node pairs (deduplicated)
    selected_shortcuts_list = [training_data.valid_shortcuts[i] for i in selected_indices]

    # Create pruned training data
    pruned_data = GoalConditionedTrainingData(
        states=[training_data.states[i] for i in selected_indices],
        current_atoms=[training_data.current_atoms[i] for i in selected_indices],
        goal_atoms=[training_data.goal_atoms[i] for i in selected_indices],
        config={
            **training_data.config,
            "pruning_method": "random",
            "max_shortcuts": max_shortcuts,
            "original_node_pair_count": num_node_pairs,
            "original_state_pair_count": len(training_data.states),
            "pruned_node_pair_count": len(selected_node_pairs),
            "pruned_state_pair_count": len(selected_indices),
            "shortcut_info": pruned_shortcut_info,  # Use pruned shortcut_info
        },
        node_states=training_data.node_states,
        valid_shortcuts=selected_shortcuts_list,
        node_atoms=training_data.node_atoms,
    )

    print(f"Selected {len(selected_node_pairs)} node pairs ({len(selected_indices)} state pairs)")
    return pruned_data


def prune_with_rollouts(
    training_data: GoalConditionedTrainingData,
    system: ImprovisationalTAMPSystem,
    planning_graph: PlanningGraph,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> GoalConditionedTrainingData:
    """Rollout-based pruning - keep shortcuts that succeed in random rollouts.

    Performs random rollouts from each source node and keeps shortcuts that
    successfully reach the target node at least threshold times.

    Args:
        training_data: Unpruned training data
        system: The TAMP system for executing rollouts
        planning_graph: The planning graph
        config: Configuration with rollout parameters
        rng: Random number generator

    Returns:
        Pruned training data with shortcuts that passed rollout tests
    """


    print("Pruning method: rollouts")
    num_rollouts_per_node = config.get("num_rollouts_per_node", 1000)
    max_steps_per_rollout = config.get("max_steps_per_rollout", 100)
    threshold = config.get("shortcut_success_threshold", 1)
    action_scale = config.get("action_scale", 1.0)
    seed = config.get("seed", 42)

    print(f"Rollout parameters:")
    print(f"  Rollouts per node: {num_rollouts_per_node}")
    print(f"  Max steps per rollout: {max_steps_per_rollout}")
    print(f"  Success threshold: {threshold}")
    print(f"Total shortcuts before pruning: {len(training_data.valid_shortcuts)}")

    # Track how many times each shortcut succeeds
    shortcut_success_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    # Get the base environment for running rollouts
    raw_env = system.env

    if isinstance(raw_env.action_space, gym.spaces.Box):
        sampling_space = gym.spaces.Box(
            low=raw_env.action_space.low * action_scale,
            high=raw_env.action_space.high * action_scale,
            dtype=np.float32,
        )
    else:
        print("Warning: Action space is not Box, using original action space.")
        sampling_space = raw_env.action_space
    
    sampling_space.seed(seed)

    # Build node lookup for efficiency
    node_by_id = {node.id: node for node in planning_graph.nodes}

    # Pre-compute target node atom sets for efficiency
    target_atoms_by_id = {node.id: set(node.atoms) for node in planning_graph.nodes}

    # Perform rollouts from each source node
    for source_id, source_states in training_data.node_states.items():
        source_node = node_by_id.get(source_id)
        if source_node is None:
            continue

        source_atoms = set(source_node.atoms)
        rollouts_per_state = max(1, num_rollouts_per_node // len(source_states))

        print(
            f"\nPerforming {rollouts_per_state} rollouts for each of "
            f"{len(source_states)} state(s) from node {source_id}",
            flush=True
        )

        for state_idx, source_state in enumerate(source_states):
            for rollout_idx in range(rollouts_per_state):
                if rollout_idx > 0 and rollout_idx % 100 == 0:
                    print(f"  Completed {rollout_idx}/{rollouts_per_state} rollouts", flush=True)

                # Reset to source state
                raw_env.reset_from_state(source_state)
                curr_atoms = source_atoms.copy()

                # Track which nodes we've reached in this rollout
                reached_in_this_rollout: set[int] = set()

                # Execute random rollout
                for step_idx in range(max_steps_per_rollout):
                    action = sampling_space.sample()
                    obs, _, terminated, truncated, _ = raw_env.step(action)
                    curr_atoms = system.perceiver.step(obs)

                    # Check if we've reached any target nodes
                    for target_id in training_data.node_states.keys():
                        # print(target_id)
                        # if target_id <= source_id:
                        #     continue

                        # Skip if not a valid shortcut
                        if (source_id, target_id) not in training_data.valid_shortcuts:
                            continue

                        # Skip if already reached in this rollout
                        if target_id in reached_in_this_rollout:
                            continue

                        # Check if atoms match target node
                        target_atoms = target_atoms_by_id.get(target_id)
                        if target_atoms and target_atoms == curr_atoms:
                            shortcut_success_counts[(source_id, target_id)] += 1
                            reached_in_this_rollout.add(target_id)

                    if terminated or truncated:
                        break

            # Print progress after each state
            print(f"  Completed all rollouts for state {state_idx + 1}/{len(source_states)} from node {source_id}", flush=True)

    print("\nRollout results:")
    for (source_id, target_id), count in shortcut_success_counts.items():
        print(f"  Shortcut ({source_id} -> {target_id}): {count} successes")

    # Select shortcuts that meet the threshold
    selected_shortcuts = []
    for source_id, target_id in training_data.valid_shortcuts:
        success_count = shortcut_success_counts.get((source_id, target_id), 0)
        if success_count >= threshold:
            selected_shortcuts.append((source_id, target_id))

    pruned_count = len(training_data.valid_shortcuts) - len(selected_shortcuts)
    print(f"\nPruned {pruned_count} shortcuts")
    print(f"Kept {len(selected_shortcuts)} shortcuts")

    # Filter training data to match selected shortcuts
    # Note: selected_shortcuts already contains duplicates (one per state pair)
    # because it was built by iterating over training_data.valid_shortcuts
    selected_set = set(selected_shortcuts)
    selected_indices = []

    for i, (source_id, target_id) in enumerate(training_data.valid_shortcuts):
        if (source_id, target_id) in selected_set:
            selected_indices.append(i)

    # Filter shortcut_info to match the pruned data
    original_shortcut_info = training_data.config.get("shortcut_info", [])
    pruned_shortcut_info = [original_shortcut_info[i] for i in selected_indices] if original_shortcut_info else []

    pruned_data = GoalConditionedTrainingData(
        states=[training_data.states[i] for i in selected_indices],
        current_atoms=[training_data.current_atoms[i] for i in selected_indices],
        goal_atoms=[training_data.goal_atoms[i] for i in selected_indices],
        config={
            **training_data.config,
            "pruning_method": "rollouts",
            "num_rollouts_per_node": num_rollouts_per_node,
            "max_steps_per_rollout": max_steps_per_rollout,
            "shortcut_success_threshold": threshold,
            "original_node_pair_count": len(set(training_data.valid_shortcuts)),
            "original_state_pair_count": len(training_data.states),
            "pruned_node_pair_count": len(set(selected_shortcuts)),
            "pruned_state_pair_count": len(selected_shortcuts),
            "shortcut_info": pruned_shortcut_info,  # Use pruned shortcut_info
        },
        node_states=training_data.node_states,
        valid_shortcuts=selected_shortcuts,
        node_atoms=training_data.node_atoms,
    )

    return pruned_data


def prune_with_distance_heuristic(
    training_data: GoalConditionedTrainingData,
    system: ImprovisationalTAMPSystem,
    planning_graph: PlanningGraph,
    config: dict[str, Any],
    rng: np.random.Generator,
    heuristic: "GoalConditionedDistanceHeuristic | None" = None,
) -> GoalConditionedTrainingData:
    """Distance heuristic pruning - keep shortcuts where f(s,s') < min(D(s,s'), K).

    Uses a goal-conditioned distance heuristic to identify shortcuts that are
    learnable within the practical training horizon K.

    The criterion f(s,s') < min(D(s,s'), K) ensures that:
    - For shortcuts with finite graph distance, we keep those where f < min(D, K)
    - For shortcuts with infinite graph distance (no existing path), we keep if f < K
    - K represents the practical training horizon (max_training_steps_per_shortcut)
    - This avoids trying to learn shortcuts that are too distant to reach in training

    Args:
        training_data: Unpruned training data
        system: The TAMP system
        planning_graph: The planning graph for computing graph distances
        config: Configuration with heuristic parameters
        rng: Random number generator
        heuristic: Pre-trained heuristic (if None, will train a new one)

    Returns:
        Pruned training data with shortcuts that pass the distance heuristic test
    """
    print("Pruning method: distance_heuristic")

    # Extract parameters from config
    # Use heuristic_practical_horizon if specified, otherwise 2x max_training_steps_per_shortcut
    practical_horizon = config.get(
        "heuristic_practical_horizon",
        2 * config.get("max_training_steps_per_shortcut", 50)
    )

    # Extract heuristic training parameters for saving in config later
    heuristic_training_pairs = config.get("heuristic_training_pairs", 100)
    heuristic_training_steps = config.get("heuristic_training_steps", 50000)

    print(f"Heuristic parameters:")
    print(f"  Practical horizon K: {practical_horizon}")
    print(f"Total shortcuts before pruning: {len(training_data.states)}")

    # Step 1: Compute graph distances D(s, s')
    print("\nComputing graph distances...")
    graph_distances = compute_graph_distances(planning_graph, exclude_shortcuts=True)
    print(f"Computed {len(graph_distances)} pairwise distances")

    # Step 2: Prepare state pairs for all shortcuts
    print("\nPreparing state pairs for shortcuts...")

    # Track all state pairs for training and evaluation
    all_state_pairs = []  # List of (source_state, target_state, source_id, target_id, graph_dist)
    node_pair_to_state_pairs = {}  # Map (source_id, target_id) -> list of state pair indices

    for source_id, target_id in training_data.valid_shortcuts:
        if source_id not in training_data.node_states:
            continue
        if target_id not in training_data.node_states:
            continue

        source_states = training_data.node_states[source_id]
        target_states = training_data.node_states[target_id]

        if not source_states or not target_states:
            continue

        graph_dist = graph_distances.get((source_id, target_id), float("inf"))

        # Create all combinations of source and target states
        state_pair_indices = []
        for source_state in source_states:
            for target_state in target_states:
                state_pair_idx = len(all_state_pairs)
                all_state_pairs.append({
                    "source_state": source_state,
                    "target_state": target_state,
                    "source_id": source_id,
                    "target_id": target_id,
                    "graph_distance": graph_dist,
                })
                state_pair_indices.append(state_pair_idx)

        node_pair_to_state_pairs[(source_id, target_id)] = state_pair_indices

    print(f"Prepared {len(all_state_pairs)} state pairs from {len(training_data.valid_shortcuts)} node pairs")
    print(f"Average {len(all_state_pairs) / max(1, len(training_data.valid_shortcuts)):.1f} state pairs per node pair")

    # Step 3: Train distance heuristic (if not provided)
    if heuristic is None:
        print()  # Blank line before training output
        heuristic = train_distance_heuristic(training_data, system, config, rng)
    else:
        print("\nUsing pre-trained distance heuristic")

    # Step 4: Evaluate heuristic on all shortcuts and prune
    print(f"\nEvaluating heuristic on all {len(all_state_pairs)} state pairs...")
    print(f"Pruning criterion: f(s,s') < min(D(s,s'), {practical_horizon})")

    # Use V4's prune() method to select promising node pairs
    # This returns the union of:
    # 1. Shortcuts where estimated < graph distance
    # 2. Top keep_fraction of pairs by (estimated - graph) score
    # Also filters out pairs with estimated distance > practical_horizon
    keep_fraction = config.get("heuristic_keep_fraction", 0.5)
    selected_node_pairs = heuristic.prune(
        graph_distances,
        keep_fraction=keep_fraction,
        max_episode_steps=practical_horizon,
    )

    print(f"Pruned {len(training_data.valid_shortcuts) - len(selected_node_pairs)} shortcuts")
    print(f"Kept {len(selected_node_pairs)} shortcuts")

    # Convert to list for consistency with training data structure
    selected_shortcuts = list(selected_node_pairs)

    # Step 5: Create pruned training data
    # Filter training data to match selected shortcuts
    # Note: selected_shortcuts already contains duplicates (one per state pair)
    # because it was built by iterating over training_data.valid_shortcuts
    selected_set = set(selected_shortcuts)
    selected_indices = []

    for i, (source_id, target_id) in enumerate(training_data.valid_shortcuts):
        if (source_id, target_id) in selected_set:
            selected_indices.append(i)

    # Filter shortcut_info to match selected indices
    original_shortcut_info = training_data.config.get("shortcut_info", [])
    pruned_shortcut_info = [original_shortcut_info[i] for i in selected_indices] if original_shortcut_info else []

    pruned_data = GoalConditionedTrainingData(
        states=[training_data.states[i] for i in selected_indices],
        current_atoms=[training_data.current_atoms[i] for i in selected_indices],
        goal_atoms=[training_data.goal_atoms[i] for i in selected_indices],
        config={
            **training_data.config,
            "pruning_method": "distance_heuristic",
            "practical_horizon": practical_horizon,
            "heuristic_training_pairs": heuristic_training_pairs,
            "heuristic_training_steps": heuristic_training_steps,
            "original_node_pair_count": len(set(training_data.valid_shortcuts)),
            "original_state_pair_count": len(training_data.states),
            "pruned_node_pair_count": len(set(selected_shortcuts)),
            "pruned_state_pair_count": len(selected_shortcuts),
            "shortcut_info": pruned_shortcut_info,  # Use pruned shortcut_info
        },
        node_states=training_data.node_states,
        valid_shortcuts=selected_shortcuts,
        node_atoms=training_data.node_atoms,
    )

    return pruned_data
