"""Simplified training data collection for shortcut learning.

This module implements a cleaner architecture:
1. Collect m diverse states per node
2. Select k shortcut pairs (node pairs to train on)
3. Generate m training examples per shortcut (one for each source state)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from relational_structs import GroundAtom

from shortcut_learning.configs import CollectionConfig
from shortcut_learning.methods.graph_utils import PlanningGraph, PlanningGraphNode
from shortcut_learning.methods.training_data import ShortcutTrainingData

if TYPE_CHECKING:
    from shortcut_learning.methods.slap_approach import SLAPApproach

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def collect_diverse_states_per_node(
    approach: SLAPApproach,
    states_per_node: int = 10,
    perturbation_steps: int = 5,
    use_multi_start: bool = True,
) -> None:
    """Collect diverse low-level states for each node in the planning graph.

    Two collection strategies:
    1. Single-start (use_multi_start=False): Always start from node 0, execute paths to reach each node.
       - Pro: Predictable, tests specific paths
       - Con: Deep nodes are very hard to reach (e.g., node 5 gets only 2/50 states)

    2. Multi-start (use_multi_start=True, DEFAULT): Use each node as a starting point.
       For each node N, try to reach all other nodes from N.
       - Pro: Even coverage - every node serves as source equally often
       - Con: Need to know how to reach each starting node initially

    Args:
        approach: The SLAP approach (contains system and planning_graph)
        states_per_node: Target number of diverse states to collect per node
        perturbation_steps: Number of random action steps for perturbations
        use_multi_start: If True, use multi-start strategy for even coverage (default)
    """
    if use_multi_start:
        _collect_diverse_states_multi_start(approach, states_per_node, perturbation_steps)
    else:
        _collect_diverse_states_single_start(approach, states_per_node, perturbation_steps)


def _collect_diverse_states_multi_start(
    approach: SLAPApproach,
    states_per_node: int = 10,
    perturbation_steps: int = 5,
) -> None:
    """Collect states using multi-start strategy: each node serves as a starting point.

    Strategy:
    - For each node N in the graph:
      1. Reset environment and navigate to node N
      2. Try to reach all other nodes from N
      3. Collect states at each reachable node

    This ensures even coverage: every node is used as a source equally often,
    avoiding the bias where deep nodes are hard to reach from node 0.
    """
    print(f"\n=== Collecting states per node (MULTI-START) ===")
    print(f"Target: {states_per_node} states/node")

    system = approach.system
    planning_graph = approach.planning_graph
    assert planning_graph is not None, "Planning graph must be created first"

    if not planning_graph.nodes:
        return

    # For each node, use it as a starting point
    for start_node in planning_graph.nodes:
        print(f"\n--- Using node {start_node.id} as starting point ---")

        # Find paths from initial node (0) to this start_node
        initial_node = planning_graph.nodes[0]
        path_to_start = _find_regular_path(planning_graph, initial_node, start_node)

        if path_to_start is None and start_node != initial_node:
            print(f"  No path to reach start node {start_node.id}, skipping")
            continue

        # Now try to reach all other nodes from start_node
        for target_node in planning_graph.nodes:
            # Skip if we already have enough states for this target
            if len(target_node.states) >= states_per_node:
                continue

            # Find path from start_node to target_node
            if target_node == start_node:
                # Collecting state at the start node itself
                path_to_target = []
            else:
                path_to_target = _find_regular_path(planning_graph, start_node, target_node)
                if not path_to_target:
                    continue  # Can't reach this target from this start

            # Try to collect states for this target node
            attempts_needed = states_per_node - len(target_node.states)
            max_attempts = attempts_needed * 3

            for attempt in range(max_attempts):
                if len(target_node.states) >= states_per_node:
                    break

                # Reset environment
                obs, info = system.reset()

                # Navigate to start_node
                if path_to_start:
                    obs, success = _execute_path(approach, obs, path_to_start, 0)
                    if not success:
                        continue

                # Navigate from start_node to target_node
                if path_to_target:
                    final_obs, success = _execute_path(
                        approach, obs, path_to_target,
                        perturbation_steps if attempt > 0 else 0
                    )
                    if not success:
                        continue
                else:
                    # Already at target (target == start)
                    final_obs = obs

                # Verify we're at the right node
                atoms = system.perceiver.step(final_obs)
                if set(target_node.atoms) == atoms:
                    if not _is_duplicate_state(final_obs, target_node.states):
                        target_node.states.append(final_obs)

    # Print summary
    total_states = sum(len(node.states) for node in planning_graph.nodes)
    print(f"\nTotal states collected: {total_states} across {len(planning_graph.nodes)} nodes")
    for node in planning_graph.nodes:
        print(f"  Node {node.id}: {len(node.states)} states")


def _collect_diverse_states_single_start(
    approach: SLAPApproach,
    states_per_node: int = 10,
    perturbation_steps: int = 5,
) -> None:
    """Original single-start collection strategy (always start from node 0)."""
    print(f"\n=== Collecting {states_per_node} states per node (SINGLE-START) ===")

    system = approach.system
    planning_graph = approach.planning_graph
    assert planning_graph is not None, "Planning graph must be created first"

    # Get initial node
    if not planning_graph.nodes:
        return
    initial_node = planning_graph.nodes[0]

    # Collect multiple diverse states for the initial node
    # Note: Each system.reset() produces a different random initial state,
    # providing natural diversity in continuous states for the same abstract state
    print(f"\nCollecting states for node {initial_node.id} (initial node)...")
    for attempt in range(states_per_node):
        obs, info = system.reset()

        # Apply perturbations for additional diversity (skip first attempt)
        if attempt > 0 and perturbation_steps > 0:
            for _ in range(perturbation_steps):
                random_action = system.env.action_space.sample()
                obs, _, term, trunc, _ = system.env.step(random_action)
                if term or trunc:
                    break

            # Verify we're still at the initial node
            atoms = system.perceiver.step(obs)
            if set(initial_node.atoms) != atoms:
                continue

        # Check for duplicates and add state
        if not _is_duplicate_state(obs, initial_node.states):
            initial_node.states.append(obs)

    print(f"  Collected {len(initial_node.states)} states for node {initial_node.id}")

    # For each other node, reach it and collect states
    for target_node in planning_graph.nodes:
        if target_node == initial_node:
            continue

        print(f"\nCollecting states for node {target_node.id}...")

        # Find a path to this node (using only regular edges)
        path = _find_regular_path(planning_graph, initial_node, target_node)
        if not path:
            print(f"  No path found to node {target_node.id}")
            continue

        # Collect multiple diverse states for this node
        # Note: Each system.reset() produces a different initial state, and skills
        # may fail on some configurations or perturbations may change the abstract state.
        # We try multiple times to collect the requested number of states.
        states_collected = 0
        max_attempts = states_per_node * 3  # Try more times if some fail

        for attempt in range(max_attempts):
            if states_collected >= states_per_node:
                break

            # Reset to a new random initial state and execute path
            obs, info = system.reset()
            final_obs, success = _execute_path(approach, obs, path, perturbation_steps if attempt > 0 else 0)

            if success:
                # Get current atoms
                atoms = system.perceiver.step(final_obs)

                # Verify we're at the right node
                if set(target_node.atoms) == atoms:
                    # Check if this state is different from existing ones
                    if not _is_duplicate_state(final_obs, target_node.states):
                        target_node.states.append(final_obs)
                        states_collected += 1

        print(f"  Collected {len(target_node.states)} states for node {target_node.id}")

    # Print summary
    total_states = sum(len(node.states) for node in planning_graph.nodes)
    print(f"\nTotal states collected: {total_states} across {len(planning_graph.nodes)} nodes")


def _find_regular_path(
    graph: PlanningGraph,
    start_node: PlanningGraphNode,
    target_node: PlanningGraphNode,
) -> list[Any]:
    """Find a path using only regular (non-shortcut) edges."""
    from collections import deque

    queue = deque([(start_node, [])])
    visited = {start_node}

    while queue:
        current, path = queue.popleft()

        if current == target_node:
            return path

        for edge in graph.node_to_outgoing_edges.get(current, []):
            if edge.is_shortcut:
                continue
            if edge.target not in visited:
                visited.add(edge.target)
                queue.append((edge.target, path + [edge]))

    return []


def _execute_path(
    approach: SLAPApproach,
    obs: Any,
    path: list[Any],
    perturbation_steps: int = 0,
) -> tuple[Any, bool]:
    """Execute a path through the graph.

    Args:
        approach: SLAP approach (contains system)
        obs: Current observation (from system.reset())
        path: List of edges to traverse
        perturbation_steps: If > 0, apply random perturbations after each edge

    Returns:
        Tuple of (final_obs, success)
    """
    system = approach.system

    for edge in path:
        if not edge.operator:
            return obs, False

        # Get skill for this operator
        skill = None
        for s in system.skills:
            if s.can_execute(edge.operator):
                skill = s
                break

        if not skill:
            return obs, False

        skill.reset(edge.operator)

        # Execute skill
        max_skill_steps = 50
        skill_success = False
        for step in range(max_skill_steps):
            action = skill.get_action(obs)
            if action is None:
                skill_success = True
                break
            obs, _, term, trunc, _ = system.env.step(action)
            atoms = system.perceiver.step(obs)

            if set(edge.target.atoms) == atoms:
                skill_success = True
                break

            if term or trunc:
                return obs, False

        if not skill_success:
            return obs, False

        # Apply perturbations if requested
        if perturbation_steps > 0:
            for _ in range(perturbation_steps):
                random_action = system.env.action_space.sample()
                obs, _, term, trunc, _ = system.env.step(random_action)
                if term or trunc:
                    return obs, False

    return obs, True


def _is_duplicate_state(state: Any, existing_states: list[Any]) -> bool:
    """Check if a state is a duplicate of any existing state."""
    if not existing_states:
        return False

    for existing_state in existing_states:
        if hasattr(state, "nodes") and hasattr(existing_state, "nodes"):
            if np.array_equal(state.nodes, existing_state.nodes):
                return True
        elif isinstance(state, np.ndarray) and isinstance(existing_state, np.ndarray):
            if np.array_equal(state, existing_state):
                return True

    return False


def _select_promising_shortcuts_with_rollouts(
    system,
    planning_graph: PlanningGraph,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 30,
    shortcut_success_threshold: int = 1,
    action_scale: float = 1.0,
    random_seed: int = 42,
) -> list[tuple[PlanningGraphNode, PlanningGraphNode]]:
    """Identify promising shortcuts by performing random rollouts from each node.

    This function runs random rollouts from each source node in the planning graph
    and tracks which nodes' atoms are reached during exploration. Shortcuts that
    are reached at least `shortcut_success_threshold` times through random exploration
    are considered promising candidates for training.

    Args:
        system: TAMPSystem with env and perceiver
        planning_graph: The planning graph
        num_rollouts_per_node: Number of random rollouts per source node
        max_steps_per_rollout: Maximum steps per rollout
        shortcut_success_threshold: Minimum successes to consider promising
        action_scale: Scale factor for random actions
        random_seed: Random seed for rollouts

    Returns:
        List of (source_node, target_node) tuples for promising shortcuts
    """
    import gymnasium as gym
    from collections import defaultdict

    print("\n=== Identifying Promising Shortcuts with Random Rollouts ===")
    shortcut_success_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    # Get the base environment for running rollouts
    raw_env = system.env
    sampling_space = gym.spaces.Box(
        low=raw_env.action_space.low * action_scale,
        high=raw_env.action_space.high * action_scale,
        dtype=raw_env.action_space.dtype,
    )
    sampling_space.seed(random_seed)

    # For each node with states, perform random rollouts
    for source_node in planning_graph.nodes:
        if not source_node.states:
            continue

        source_atoms = set(source_node.atoms)
        rollouts_per_state = max(1, num_rollouts_per_node // len(source_node.states))
        print(
            f"\nPerforming {rollouts_per_state} rollouts for each of {len(source_node.states)} "
            f"state(s) from node {source_node.id}"
        )

        # Track other nodes reached from this source node
        reached_nodes: defaultdict[int, int] = defaultdict(int)

        # Perform random rollouts
        for source_state in source_node.states:
            for rollout_idx in range(rollouts_per_state):
                # Reset the environment to source state
                raw_env.reset_from_state(source_state)
                curr_atoms = source_atoms.copy()

                # Execute random actions
                reached_in_this_rollout: set[int] = set()
                for _ in range(max_steps_per_rollout):
                    action = sampling_space.sample()
                    obs, _, terminated, truncated, _ = raw_env.step(action)
                    curr_atoms = system.perceiver.step(obs)

                    # Check if any node is reached
                    for target_node in planning_graph.nodes:
                        if target_node.id <= source_node.id:
                            continue
                        if not target_node.states:
                            continue

                        # Check if there's already a direct regular edge
                        has_direct_edge = any(
                            edge.target == target_node and not edge.is_shortcut
                            for edge in planning_graph.node_to_outgoing_edges.get(source_node, [])
                        )
                        if has_direct_edge:
                            continue

                        # Note: no need to stop this rollout when we reach a node
                        # since we want to explore all reachable nodes
                        if (
                            set(target_node.atoms) == curr_atoms
                            and target_node.id not in reached_in_this_rollout
                        ):
                            reached_nodes[target_node.id] += 1
                            shortcut_success_counts[(source_node.id, target_node.id)] += 1
                            reached_in_this_rollout.add(target_node.id)

                    if terminated or truncated:
                        break

        if reached_nodes:
            print(f"  Nodes reached from node {source_node.id}:")
            for target_id, count in sorted(reached_nodes.items(), key=lambda x: -x[1]):
                total_rollouts = rollouts_per_state * len(source_node.states)
                print(f"    → Node {target_id}: {count}/{total_rollouts} times")
        else:
            print(f"  No nodes reached from node {source_node.id}")

    # Collect promising shortcut pairs
    promising_pairs = []
    print("\nShortcuts reaching success threshold:")
    for (source_id, target_id), count in sorted(
        shortcut_success_counts.items(), key=lambda x: -x[1]
    ):
        if count >= shortcut_success_threshold:
            source_node = next(
                (n for n in planning_graph.nodes if n.id == source_id), None
            )
            target_node = next(
                (n for n in planning_graph.nodes if n.id == target_id), None
            )

            assert source_node is not None and target_node is not None
            print(f"  Node {source_id} → Node {target_id}: {count} successes")
            promising_pairs.append((source_node, target_node))

    print(f"\nFound {len(promising_pairs)} promising shortcut pairs")
    return promising_pairs


def select_shortcut_pairs(
    planning_graph: PlanningGraph,
    max_shortcuts: int,
    rng: np.random.Generator,
    use_random_rollouts: bool = False,
    system=None,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 30,
    shortcut_success_threshold: int = 1,
    action_scale: float = 1.0,
) -> list[tuple[PlanningGraphNode, PlanningGraphNode]]:
    """Select k pairs of nodes to train shortcuts on.

    Can use either random selection or random rollout pruning.
    A "valid pair" is (source, target) where:
    - source.id < target.id (prevent duplicates and self-loops)
    - No direct regular edge exists from source to target
    - Both nodes have collected states

    Args:
        planning_graph: The planning graph
        max_shortcuts: Maximum number of shortcuts to select
        rng: Random number generator
        use_random_rollouts: If True, use random rollout pruning to find promising shortcuts
        system: TAMPSystem (required if use_random_rollouts=True)
        num_rollouts_per_node: Number of random rollouts per node (for pruning)
        max_steps_per_rollout: Max steps per rollout (for pruning)
        shortcut_success_threshold: Minimum successes to consider promising (for pruning)
        action_scale: Scale for random actions (for pruning)

    Returns:
        List of (source_node, target_node) tuples
    """
    print(f"\n=== Selecting up to {max_shortcuts} shortcut pairs ===")

    if use_random_rollouts:
        if system is None:
            raise ValueError("system is required when use_random_rollouts=True")

        # Use random rollout pruning
        selected_pairs = _select_promising_shortcuts_with_rollouts(
            system=system,
            planning_graph=planning_graph,
            num_rollouts_per_node=num_rollouts_per_node,
            max_steps_per_rollout=max_steps_per_rollout,
            shortcut_success_threshold=shortcut_success_threshold,
            action_scale=action_scale,
            random_seed=int(rng.integers(0, 2**31)),
        )

        # Further limit to max_shortcuts if needed
        if len(selected_pairs) > max_shortcuts:
            indices = rng.choice(len(selected_pairs), size=max_shortcuts, replace=False)
            selected_pairs = [selected_pairs[i] for i in indices]
            print(f"Randomly selected {len(selected_pairs)} from promising shortcuts")

        return selected_pairs
    else:
        # Random selection from all valid pairs (original behavior)
        valid_pairs = []
        for source_node in planning_graph.nodes:
            if not source_node.states:
                continue

            for target_node in planning_graph.nodes:
                if target_node.id <= source_node.id:
                    continue
                if not target_node.states:
                    continue

                # Check if there's already a direct regular edge
                has_direct_edge = any(
                    edge.target == target_node and not edge.is_shortcut
                    for edge in planning_graph.node_to_outgoing_edges.get(source_node, [])
                )
                if has_direct_edge:
                    continue

                valid_pairs.append((source_node, target_node))

        print(f"Found {len(valid_pairs)} valid shortcut pairs")

        # Randomly select up to max_shortcuts
        if len(valid_pairs) <= max_shortcuts:
            selected_pairs = valid_pairs
        else:
            indices = rng.choice(len(valid_pairs), size=max_shortcuts, replace=False)
            selected_pairs = [valid_pairs[i] for i in indices]

        print(f"Selected {len(selected_pairs)} shortcut pairs for training")

        return selected_pairs


def generate_training_data(
    shortcut_pairs: list[tuple[PlanningGraphNode, PlanningGraphNode]],
) -> ShortcutTrainingData:
    """Generate training data from shortcut pairs.

    With the new structure, we just return the shortcut pairs directly.
    The nodes themselves contain the states, so the training process can
    access source_node.states and target_node.atoms as needed.

    Args:
        shortcut_pairs: List of (source, target) node pairs

    Returns:
        ShortcutTrainingData object with shortcut pairs
    """
    print(f"\n=== Generating training data from {len(shortcut_pairs)} shortcuts ===")

    total_examples = sum(len(source.states) for source, _ in shortcut_pairs)
    avg_per_shortcut = total_examples / len(shortcut_pairs) if shortcut_pairs else 0

    print(f"Total training examples: {total_examples}")
    print(f"  Average examples per shortcut: {avg_per_shortcut:.1f}")

    # Store shortcut metadata
    shortcut_info = []
    for shortcut_idx, (source_node, target_node) in enumerate(shortcut_pairs):
        shortcut_info.append({
            "shortcut_id": shortcut_idx,
            "source_node_id": source_node.id,
            "target_node_id": target_node.id,
            "num_source_states": len(source_node.states),
            "num_target_states": len(target_node.states),
        })

    return ShortcutTrainingData(
        shortcuts=shortcut_pairs,
        config={
            "num_shortcuts": len(shortcut_pairs),
            "shortcut_info": shortcut_info,
        },
    )


def collect_training_data_v2(
    approach: SLAPApproach,
    config: CollectionConfig,
) -> ShortcutTrainingData:
    """Main entry point for v2 collection pipeline.

    Pipeline:
    1. Collect m diverse states per node
    2. Select k shortcut pairs
    3. Generate training data (shortcut pairs with states in nodes)

    Args:
        approach: SLAP approach (must have planning_graph already created)
        config: Collection configuration

    Returns:
        ShortcutTrainingData containing the k shortcut pairs
    """
    print("\n" + "="*60)
    print("V2 COLLECTION PIPELINE")
    print("="*60)

    assert approach.planning_graph is not None, "Must call approach.reset() first to build planning graph"

    rng = np.random.default_rng(config.seed)

    # Step 1: Collect diverse states per node
    collect_diverse_states_per_node(
        approach,
        states_per_node=config.states_per_node,
        perturbation_steps=config.perturbation_steps,
    )

    # Step 2: Select shortcut pairs
    shortcut_pairs = select_shortcut_pairs(
        approach.planning_graph,
        max_shortcuts=config.max_shortcuts_per_graph,
        rng=rng,
        use_random_rollouts=config.use_random_rollouts,
        system=approach.system,
        num_rollouts_per_node=config.num_rollouts_per_node,
        max_steps_per_rollout=config.max_steps_per_rollout,
        shortcut_success_threshold=config.shortcut_success_threshold,
        action_scale=config.action_scale,
    )

    # Step 3: Generate training data
    training_data = generate_training_data(shortcut_pairs)

    print("\n" + "="*60)
    print(f"COLLECTION COMPLETE: {training_data.num_training_examples()} examples for {len(training_data)} shortcuts")
    print("="*60 + "\n")

    return training_data
