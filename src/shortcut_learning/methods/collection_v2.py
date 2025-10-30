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
    _collect_diverse_states_multi_start(approach, states_per_node)

def _collect_diverse_states_multi_start(
    approach: SLAPApproach,
    states_per_node: int = 10,
) -> None:
    """Collect states using simple BFS exploration from reset.

    Strategy (for num_episodes episodes):
      1. Reset environment to a random initial state
      2. BFS: explore all EDGES reachable from initial state (not just nodes)
      3. Save states reached via each edge (one node can get multiple states per episode)
      4. Track which edge was used to reach each state

    This is simple, natural, and gets diverse incoming edges automatically.
    With ~100 episodes, nodes get many states from all their incoming edges.

    Note: states_per_node parameter is repurposed as number of collection episodes.
    """
    print(f"\n=== Collecting states via BFS edge exploration ===")
    num_episodes = states_per_node
    print(f"Running {num_episodes} collection episodes")

    system = approach.system
    planning_graph = approach.planning_graph
    assert planning_graph is not None, "Planning graph must be created first"

    if not planning_graph.nodes:
        return

    # Run collection episodes
    import time

    # Timing statistics
    total_reset_time = 0.0
    total_bfs_time = 0.0
    total_skill_time = 0.0
    total_episodes_with_timing = 0

    for episode in range(num_episodes):
        episode_start = time.time()
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")

        # Reset to random initial state
        reset_start = time.time()
        obs, info = system.reset()

        # Get initial atoms to find starting node
        atoms = system.perceiver.step(obs)

        # Find which node this initial state corresponds to
        start_node = None
        for node in planning_graph.nodes:
            if set(node.atoms) == atoms:
                start_node = node
                break
        reset_time = time.time() - reset_start

        if start_node is None:
            # Initial state not in graph - skip this episode
            continue

        # BFS exploration: traverse all reachable EDGES (not just nodes)
        # Track: (node, state, incoming_edge_id) tuples to save after BFS
        states_to_save = []

        # Queue: (current_node, current_obs, incoming_edge_id)
        from collections import deque
        queue = deque([(start_node, obs, None)])  # Start has no incoming edge
        visited_edges = set()  # Track (source_id, target_id, edge_id) tuples

        bfs_start = time.time()
        edges_explored = 0

        while queue:
            current_node, current_obs, incoming_edge = queue.popleft()

            # print(current_node)
            # Save this state (add after BFS to avoid modifying during iteration)
            states_to_save.append((current_node, current_obs, incoming_edge))

            # Explore all outgoing edges from current node
            for edge in planning_graph.edges:
                if edge.source != current_node:
                    continue

                # Skip shortcuts during collection
                if edge.is_shortcut:
                    continue

                # Create edge signature
                edge_sig = (edge.source.id, edge.target.id, edge.edge_id)

                # Skip if this edge already traversed
                if edge_sig in visited_edges:
                    continue

                visited_edges.add(edge_sig)

                # Try to execute this edge
                skill_start = time.time()
                edges_explored += 1

                env = system.env
                env.reset_from_state(current_obs)

                # Get skill and execute
                skill = approach._get_skill(edge.operator)
                skill.reset(edge.operator)

                next_obs = current_obs
                succeeded = False

                for step in range(approach._max_skill_steps):
                    action = skill.get_action(next_obs)
                    if action is None:
                        break

                    next_obs, _, term, trunc, _ = env.step(action)
                    next_atoms = system.perceiver.step(next_obs)

                    # Check if reached target
                    if set(next_atoms) == set(edge.target.atoms):
                        succeeded = True
                        break

                    if term or trunc:
                        break

                total_skill_time += time.time() - skill_start

                if succeeded:
                    # Add to queue for further exploration
                    queue.append((edge.target, next_obs, edge.edge_id))

        bfs_time = time.time() - bfs_start

        # Now save all collected states (avoiding duplicates)
        episode_collected = 0
        episode_duplicates = 0
        for node, state, incoming_edge_id in states_to_save:
            if not _is_duplicate_state(state, node.states):
                node.states.append(state)
                node.state_incoming_edges.append(incoming_edge_id)
                episode_collected += 1
            else:
                episode_duplicates += 1

        # Update timing stats
        total_reset_time += reset_time
        total_bfs_time += bfs_time
        total_episodes_with_timing += 1

        # Detailed timing for first few episodes or episodes with interesting activity
        episode_time = time.time() - episode_start
        if episode < 5 or episode_collected > 0 or (episode + 1) % 10 == 0:
            print(f"    Episode {episode}: node {start_node.id}, explored {edges_explored} edges, "
                  f"saved {episode_collected} states, rejected {episode_duplicates} dups "
                  f"[reset: {reset_time:.2f}s, BFS: {bfs_time:.2f}s, total: {episode_time:.2f}s]")

    # Print summary
    total_states = sum(len(node.states) for node in planning_graph.nodes)
    print(f"\nTotal states collected: {total_states} across {len(planning_graph.nodes)} nodes")
    for node in planning_graph.nodes:
        # Count unique incoming edges
        unique_edges = set(node.state_incoming_edges)
        edge_counts = {edge_id: node.state_incoming_edges.count(edge_id) for edge_id in unique_edges}
        edge_info = ", ".join(f"edge_{k}: {v}" if k is not None else f"initial: {v}"
                              for k, v in sorted(edge_counts.items(), key=lambda x: (x[0] is None, x[0])))
        print(f"  Node {node.id}: {len(node.states)} states ({len(unique_edges)} incoming edges: {edge_info})")

    # Print timing statistics
    if total_episodes_with_timing > 0:
        avg_reset = total_reset_time / total_episodes_with_timing
        avg_bfs = total_bfs_time / total_episodes_with_timing
        avg_skill = total_skill_time / total_episodes_with_timing
        print(f"\nTiming Statistics (avg per episode):")
        print(f"  Reset: {avg_reset:.3f}s")
        print(f"  BFS+Skills: {avg_bfs:.3f}s (skill execution: {avg_skill:.3f}s)")
        print(f"  Total collection time: {total_reset_time + total_bfs_time:.1f}s")


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


# def _execute_path(
#     approach: SLAPApproach,
#     obs: Any,
#     path: list[Any],
#     perturbation_steps: int = 0,
# ) -> tuple[Any, bool]:
#     """Execute a path through the graph.

#     Args:
#         approach: SLAP approach (contains system)
#         obs: Current observation (from system.reset())
#         path: List of edges to traverse
#         perturbation_steps: If > 0, apply random perturbations after each edge

#     Returns:
#         Tuple of (final_obs, success)
#     """
#     system = approach.system

#     for edge in path:
#         if not edge.operator:
#             return obs, False

#         # Get skill for this operator
#         skill = None
#         for s in system.skills:
#             if s.can_execute(edge.operator):
#                 skill = s
#                 break

#         if not skill:
#             return obs, False

#         skill.reset(edge.operator)

#         # Execute skill
#         max_skill_steps = 50
#         skill_success = False
#         for step in range(max_skill_steps):
#             action = skill.get_action(obs)
#             if action is None:
#                 skill_success = True
#                 break
#             obs, _, term, trunc, _ = system.env.step(action)
#             atoms = system.perceiver.step(obs)

#             if set(edge.target.atoms) == atoms:
#                 skill_success = True
#                 break

#             if term or trunc:
#                 return obs, False

#         if not skill_success:
#             return obs, False

#         # Apply perturbations if requested
#         if perturbation_steps > 0:
#             for _ in range(perturbation_steps):
#                 random_action = system.env.action_space.sample()
#                 obs, _, term, trunc, _ = system.env.step(random_action)
#                 if term or trunc:
#                     return obs, False

#     return obs, True


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
    save_path: str | None = None,
    load_path: str | None = None,
) -> ShortcutTrainingData:
    """Main entry point for v2 collection pipeline.

    Pipeline:
    1. Collect m diverse states per node
    2. Select k shortcut pairs
    3. Generate training data (shortcut pairs with states in nodes)

    Args:
        approach: SLAP approach (must have planning_graph already created)
        config: Collection configuration
        save_path: Path to save collected data (required if load_path not provided)
        load_path: Path to load previously collected data (skips collection if provided)

    Returns:
        ShortcutTrainingData containing the k shortcut pairs

    Raises:
        ValueError: If neither save_path nor load_path is provided
    """

    # If load_path is provided, try to load from cache
    if load_path is not None:
        import pickle
        from pathlib import Path

        load_file = Path(load_path)
        if not load_file.exists():
            raise FileNotFoundError(f"Cache file not found: {load_path}")

        print("\n" + "="*60)
        print("LOADING CACHED COLLECTION DATA")
        print("="*60)
        print(f"Loading from: {load_path}")

        with open(load_file, "rb") as f:
            training_data = pickle.load(f)

        print(f"Loaded {training_data.num_training_examples()} examples for {len(training_data)} shortcuts")
        print("="*60 + "\n")
        return training_data

    # Otherwise, perform collection
    print("\n" + "="*60)
    print("V2 COLLECTION PIPELINE")
    print("="*60)

    assert approach.planning_graph is not None, "Must call approach.reset() first to build planning graph"

    rng = np.random.default_rng(config.seed)

    # Step 1: Collect diverse states per node
    collect_diverse_states_per_node(
        approach,
        states_per_node=config.states_per_node,
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

    # Save to cache if save_path provided
    if save_path is not None:
        import pickle
        from pathlib import Path

        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, "wb") as f:
            pickle.dump(training_data, f)

        print(f"[CACHE] Saved collection data to: {save_path}\n")

    return training_data


def save_collection_cache(
    training_data: ShortcutTrainingData,
    system_name: str,
    cache_dir: str = "/scratch/gpfs/TSILVER/de7281/collection_cache",
) -> str:
    """Save collected training data to cache in /scratch/gpfs for reuse.

    Args:
        training_data: The collected training data
        system_name: Name of the system (e.g., 'obstacle2d', 'obstacle_tower')
        cache_dir: Directory to save cache files

    Returns:
        Path to the saved cache file
    """
    import pickle
    from pathlib import Path
    from datetime import datetime

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{system_name}_collection_{timestamp}.pkl"
    filepath = cache_path / filename

    # Save training data
    with open(filepath, "wb") as f:
        pickle.dump(training_data, f)

    # Also save a 'latest' symlink for easy access
    latest_link = cache_path / f"{system_name}_collection_latest.pkl"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(filename)

    print(f"\n[CACHE] Saved collection data to: {filepath}")
    print(f"[CACHE] Symlinked to: {latest_link}")

    return str(filepath)


def load_collection_cache(
    system_name: str,
    cache_dir: str = "/scratch/gpfs/TSILVER/de7281/collection_cache",
    use_latest: bool = True,
) -> ShortcutTrainingData | None:
    """Load cached training data from /scratch/gpfs.

    Args:
        system_name: Name of the system (e.g., 'obstacle2d', 'obstacle_tower')
        cache_dir: Directory containing cache files
        use_latest: If True, load the latest cached file; otherwise, list available files

    Returns:
        ShortcutTrainingData if found, None otherwise
    """
    import pickle
    from pathlib import Path

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"[CACHE] No cache directory found at {cache_path}")
        return None

    if use_latest:
        latest_link = cache_path / f"{system_name}_collection_latest.pkl"
        if not latest_link.exists():
            print(f"[CACHE] No cached collection found for {system_name}")
            return None

        print(f"[CACHE] Loading collection data from: {latest_link}")
        with open(latest_link, "rb") as f:
            training_data = pickle.load(f)

        print(f"[CACHE] Loaded {training_data.num_training_examples()} examples for {len(training_data)} shortcuts")
        return training_data
    else:
        # List available cache files
        cache_files = sorted(cache_path.glob(f"{system_name}_collection_*.pkl"))
        if not cache_files:
            print(f"[CACHE] No cached collections found for {system_name}")
            return None

        print(f"[CACHE] Available cache files for {system_name}:")
        for i, f in enumerate(cache_files):
            print(f"  {i}: {f.name}")
        return None
