"""Collection functions for SLAP training data.

This module provides clean data collection without any pruning logic.
Pruning is handled separately in pruning.py.
"""

from typing import Any

from omegaconf import DictConfig

import numpy as np

from tamp_improv.approaches.improvisational.base import (
    ImprovisationalTAMPApproach,
    ShortcutSignature,
)
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.graph_training import (
    ShortcutCandidate,
    identify_shortcut_candidates,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


def collect_states_for_all_nodes(
    system: ImprovisationalTAMPSystem,
    planning_graph: PlanningGraph,
    max_steps_per_skill: int = 50,
) -> dict[frozenset, list[Any]]:
    """Collect states for all reachable atoms using BFS on planning graph.

    Performs BFS on the planning graph edges, executing skills to transition
    between nodes and collecting states for each reachable atom set.

    Args:
        system: The TAMP system (should be reset to initial state before calling)
        planning_graph: The planning graph with nodes and edges
        max_steps_per_skill: Maximum low-level steps to execute each skill

    Returns:
        Dictionary mapping frozenset(atoms) -> list of states for those atoms
    """
    from collections import deque

    # Get initial state and node
    obs, info = system.reset()
    _, initial_atoms, _ = system.perceiver.reset(obs, info)
    initial_atoms_frozen = frozenset(initial_atoms)

    # Find initial node in planning graph
    print(planning_graph.nodes)
    initial_node = None
    for node in planning_graph.nodes:
        if frozenset(node.atoms) == initial_atoms_frozen:
            initial_node = node
            break

    if initial_node is None:
        print(f"Warning: Could not find node matching initial atoms: {initial_atoms}")
        return {}

    print(f"Starting BFS exploration from node {initial_node.id}: {initial_atoms}")

    # Initialize with initial state
    atom_states: dict[frozenset, list[Any]] = {initial_atoms_frozen: [obs]}

    # BFS queue: (node, state_at_node)
    queue: deque[tuple[PlanningGraphNode, Any]] = deque([(initial_node, obs)])
    visited_nodes: set[int] = {initial_node.id}

    explored_count = 0

    while queue:
        current_node, current_state = queue.popleft()
        explored_count += 1

        current_atoms_frozen = frozenset(current_node.atoms)
        print(f"\n[{explored_count}] Exploring from node {current_node.id}: {current_atoms_frozen}")
        print(f"  Queue size: {len(queue)}, Visited nodes: {len(visited_nodes)}")

        # Try each outgoing edge from current node
        for edge in planning_graph.node_to_outgoing_edges.get(current_node, []):
            # Skip shortcut edges (only use skill-based transitions)
            if edge.is_shortcut:
                continue

            target_node = edge.target
            target_atoms_frozen = frozenset(target_node.atoms)

            # Skip if already visited this node
            if target_node.id in visited_nodes:
                continue

            # Find skill that can execute this edge's operator
            if not edge.operator:
                continue

            skill = None
            for s in system.skills:
                if s.can_execute(edge.operator):
                    skill = s
                    break

            if not skill:
                print(f"  Warning: No skill found for operator {edge.operator.name}")
                continue

            print(f"  Trying edge {current_node.id} → {target_node.id} (operator: {edge.operator.name})")

            # Reset environment to current state
            obs, _ = system.env.reset_from_state(current_state)

            # Reset skill with the edge's ground operator
            skill.reset(edge.operator)

            # Execute the skill
            success = False

            for step in range(max_steps_per_skill):
                action = skill.get_action(obs)
                obs, _, _, _, _ = system.env.step(action)
                atoms = system.perceiver.step(obs)
                atoms_frozen = frozenset(atoms)

                # Check if we reached target node
                if atoms_frozen == target_atoms_frozen:
                    success = True
                    print(f"    → Reached target node {target_node.id} in {step + 1} steps")
                    break


            if success:
                # Mark node as visited
                visited_nodes.add(target_node.id)

                # Add state to atom_states
                if target_atoms_frozen not in atom_states:
                    atom_states[target_atoms_frozen] = []
                atom_states[target_atoms_frozen].append(obs)

                # Add to queue for further exploration
                queue.append((target_node, obs))
                print(f"    ✓ Added to queue for exploration")
            else:
                print(f"    ✗ Failed to reach target node")

    print(f"\n{'='*80}")
    print(f"BFS Exploration Complete")
    print(f"{'='*80}")
    print(f"Visited {len(visited_nodes)}/{len(planning_graph.nodes)} nodes")
    print(f"Discovered {len(atom_states)} unique atom sets")
    print(f"Total states collected: {sum(len(states) for states in atom_states.values())}")
    print(f"\nAtom sets discovered:")
    for atoms_frozen in sorted(atom_states.keys(), key=lambda x: len(x)):
        atoms_str = ", ".join(sorted([str(atom) for atom in atoms_frozen]))
        if not atoms_str:
            atoms_str = "(initial state)"
        print(f"  • {atoms_str} ({len(atom_states[atoms_frozen])} states)")
    print(f"{'='*80}\n")

    return atom_states


def collect_all_shortcuts(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> GoalConditionedTrainingData:
    """Collect ALL possible shortcuts without any pruning.

    This is the base collection function that builds planning graphs and
    identifies all valid shortcuts. No pruning is applied - that should be
    done separately using functions from pruning.py.

    Args:
        system: The TAMP system
        approach: The improvisational TAMP approach
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        GoalConditionedTrainingData with all possible shortcuts
    """
    if rng is None:
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

    approach.training_mode = True
    collect_episodes = config.get("collect_episodes", 10)

    # We'll collect data across multiple episodes and aggregate
    all_states = []
    all_current_atoms = []
    all_goal_atoms = []
    all_node_states = {}
    all_valid_shortcuts = []
    all_node_atoms = {}
    all_shortcut_info = []

    print(f"\n{'=' * 80}")
    print(f"Collecting shortcuts from {collect_episodes} episodes")
    print(f"{'=' * 80}")

    for episode in range(collect_episodes):
        print(f"\n=== Episode {episode + 1}/{collect_episodes} ===", flush=True)

        # Reset and build planning graph
        print("[DEBUG] Resetting system...", flush=True)
        obs, info = system.reset()
        print("[DEBUG] Building planning graph (this may take a while)...", flush=True)
        _ = approach.reset(obs, info)
        print("[DEBUG] Planning graph built", flush=True)

        assert (
            hasattr(approach, "planning_graph") and approach.planning_graph is not None
        ), "Planning graph not created"
        planning_graph = approach.planning_graph

        # Collect states at each node
        if (
            hasattr(approach, "observed_states")
            and approach.observed_states is not None
        ):
            print("Using observed states from approach")
            observed_states = approach.observed_states
        else:
            print("Collecting states for all nodes via BFS...")
            # Use BFS to collect states (returns {frozenset(atoms): [states]})
            atom_states = collect_states_for_all_nodes(system, planning_graph, max_steps_per_skill=50)

            # Convert to node ID format: {node_id: [states]}
            observed_states = {}
            for node in planning_graph.nodes:
                node_atoms_frozen = frozenset(node.atoms)
                if node_atoms_frozen in atom_states:
                    observed_states[node.id] = atom_states[node_atoms_frozen]

        print(
            f"Graph has {len(planning_graph.nodes)} nodes, "
            f"{len(planning_graph.edges)} edges"
        )
        print(f"Collected states for {len(observed_states)} nodes")

        # Identify ALL shortcut candidates (no pruning)
        shortcut_candidates = identify_shortcut_candidates(
            planning_graph,
            observed_states,
        )

        print(f"Identified {len(shortcut_candidates)} shortcut candidates")

        # Add all shortcuts to the training data
        for candidate in shortcut_candidates:
            source_id = candidate.source_node.id
            target_id = candidate.target_node.id

            if source_id in observed_states and observed_states[source_id] is not None:
                # Use the first state from source node
                for source_state in observed_states[source_id]:
                    all_states.append(source_state)
                    all_current_atoms.append(candidate.source_atoms)
                    all_goal_atoms.append(candidate.target_atoms)
                    # Store shortcut info with node IDs (needed for MultiRL policy keys)
                    all_shortcut_info.append({
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                    })

                # Track this shortcut
                all_valid_shortcuts.append((source_id, target_id))

                # Register signature with approach (matches collect_graph_based_training_data)
                signature = ShortcutSignature.from_context(
                    candidate.source_atoms,
                    candidate.target_atoms,
                )
                if signature not in approach.trained_signatures:
                    approach.trained_signatures.append(signature)

        # Merge node states (using episode-specific node IDs if needed)
        # For simplicity, we only keep the last episode's planning graph
        # This is consistent with collect_goal_conditioned_training_data
        all_node_states = observed_states
        for node in planning_graph.nodes:
            if node.id in observed_states:
                all_node_atoms[node.id] = set(node.atoms)

    print(
        f"\n{'=' * 80}\n"
        f"Collection complete: {len(all_states)} training examples, "
        f"{len(all_valid_shortcuts)} shortcuts\n"
        f"{'=' * 80}"
    )

    approach.training_mode = False

    # Return as GoalConditionedTrainingData
    return GoalConditionedTrainingData(
        states=all_states,
        current_atoms=all_current_atoms,
        goal_atoms=all_goal_atoms,
        config={
            **config,
            "collection_method": "all_shortcuts",
            "collect_episodes": collect_episodes,
            "num_shortcuts_collected": len(all_valid_shortcuts),
            "shortcut_info": all_shortcut_info,
        },
        node_states=all_node_states,
        valid_shortcuts=all_valid_shortcuts,
        node_atoms=all_node_atoms,
    )


def collect_total_shortcuts(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    cfg: DictConfig,
    rng: np.random.Generator | None = None,
) -> GoalConditionedTrainingData:
    """Collect all shortcuts from a unified total planning graph.

    This is more efficient than collect_all_shortcuts because it:
    1. Builds one large planning graph across multiple episodes
    2. Collects states for all nodes in the unified graph
    3. Identifies shortcuts once from the complete graph

    Args:
        system: The TAMP system
        approach: The improvisational TAMP approach
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        GoalConditionedTrainingData with all shortcuts from the total graph
    """
    if rng is None:
        seed = cfg.seed
        rng = np.random.default_rng(seed)

    approach.training_mode = True
    collect_episodes = cfg.collection.num_episodes

    print(f"\n{'=' * 80}")
    print(f"Collecting total planning graph from {collect_episodes} episodes")
    print(f"{'=' * 80}")

    # Build unified planning graph across all episodes
    total_graph, node_states = collect_total_planning_graph(
        system=system,
        collect_episodes=collect_episodes,
        seed=cfg.seed,
        planner_id=cfg.collection.planner_id,
        max_steps_per_edge=cfg.collection.max_steps_per_edge
    )

    # Store the total graph in the approach
    approach.planning_graph = total_graph

    print(f"\n{'=' * 80}")
    print(f"Identifying shortcuts from total graph")
    print(f"{'=' * 80}")

    # Identify all shortcut candidates from the total graph
    # Note: node_states keys are frozensets of atoms, need to convert to node IDs
    # First, create a mapping from atoms to node IDs
    atom_to_node_id = {node.atoms: node.id for node in total_graph.nodes}

    # Convert node_states to use node IDs
    observed_states = {}
    for atom_set, states in node_states.items():
        if atom_set in atom_to_node_id:
            node_id = atom_to_node_id[atom_set]
            observed_states[node_id] = states

    shortcut_candidates = identify_shortcut_candidates(
        total_graph,
        observed_states,
    )

    print(f"Identified {len(shortcut_candidates)} shortcut candidates")

    # Build training data
    all_states = []
    all_current_atoms = []
    all_goal_atoms = []
    all_valid_shortcuts = []
    all_unique_shortcuts = []  # One per node-node pair
    all_node_atoms = {}
    all_shortcut_info = []

    for candidate in shortcut_candidates:
        source_id = candidate.source_node.id
        target_id = candidate.target_node.id

        if source_id in observed_states and observed_states[source_id] is not None:
            # Add to unique shortcuts (once per node-node pair)
            all_unique_shortcuts.append((source_id, target_id))

            for source_state in observed_states[source_id]:
                all_states.append(source_state)
                all_current_atoms.append(candidate.source_atoms)
                all_goal_atoms.append(candidate.target_atoms)
                all_shortcut_info.append({
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                })
                # Add one entry to valid_shortcuts per state pair
                all_valid_shortcuts.append((source_id, target_id))

            # Register signature with approach (once per node pair)
            signature = ShortcutSignature.from_context(
                candidate.source_atoms,
                candidate.target_atoms,
            )
            if signature not in approach.trained_signatures:
                approach.trained_signatures.append(signature)

    # Store node atoms
    for node in total_graph.nodes:
        if node.id in observed_states:
            all_node_atoms[node.id] = set(node.atoms)

    print(
        f"\n{'=' * 80}\n"
        f"Collection complete: {len(all_states)} training examples, "
        f"{len(all_unique_shortcuts)} unique shortcuts ({len(all_valid_shortcuts)} state-node pairs)\n"
        f"{'=' * 80}"
    )

    approach.training_mode = False

    return GoalConditionedTrainingData(
        states=all_states,
        current_atoms=all_current_atoms,
        goal_atoms=all_goal_atoms,
        config={
            "collection_method": "total_shortcuts",
            "collect_episodes": collect_episodes,
            "num_shortcuts_collected": len(all_valid_shortcuts),
            "num_unique_shortcuts": len(all_unique_shortcuts),
            "shortcut_info": all_shortcut_info,
            "max_training_steps_per_shortcut": cfg.policy.max_episode_steps,
            "episodes_per_scenario": cfg.policy.episodes_per_scenario,
            "early_stopping": cfg.policy.early_stopping,
            "early_stopping_patience": cfg.policy.early_stopping_patience,
            "training_record_interval": cfg.policy.training_record_interval,
        },
        node_states=observed_states,
        valid_shortcuts=all_valid_shortcuts,
        unique_shortcuts=all_unique_shortcuts,
        node_atoms=all_node_atoms,
        graph=total_graph,
    )


def collect_shortcuts_single_episode(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
) -> GoalConditionedTrainingData:
    """Collect all shortcuts from a single episode.

    This is a simplified version that works with a single planning graph.
    Useful for quick tests or when you want to work with one graph at a time.

    Args:
        system: The TAMP system
        approach: The improvisational TAMP approach (should be reset already)

    Returns:
        GoalConditionedTrainingData with all shortcuts from this episode
    """
    approach.training_mode = True

    assert (
        hasattr(approach, "planning_graph") and approach.planning_graph is not None
    ), "Planning graph not created - call approach.reset() first"
    planning_graph = approach.planning_graph

    # Collect states at each node
    if hasattr(approach, "observed_states") and approach.observed_states is not None:
        observed_states = approach.observed_states
    else:
        # Use BFS to collect states (returns {frozenset(atoms): [states]})
        atom_states = collect_states_for_all_nodes(system, planning_graph, max_steps_per_skill=50)

        # Convert to node ID format: {node_id: [states]}
        observed_states = {}
        for node in planning_graph.nodes:
            node_atoms_frozen = frozenset(node.atoms)
            if node_atoms_frozen in atom_states:
                observed_states[node.id] = atom_states[node_atoms_frozen]

    # Identify all shortcut candidates
    shortcut_candidates = identify_shortcut_candidates(
        planning_graph,
        observed_states,
    )

    # Build training data
    states = []
    current_atoms_list = []
    goal_atoms_list = []
    valid_shortcuts = []
    node_atoms = {}

    for candidate in shortcut_candidates:
        source_id = candidate.source_node.id
        target_id = candidate.target_node.id

        if source_id in observed_states and observed_states[source_id] is not None:
            for source_state in observed_states[source_id]:
                states.append(source_state)
                current_atoms_list.append(candidate.source_atoms)
                goal_atoms_list.append(candidate.target_atoms)

            valid_shortcuts.append((source_id, target_id))

    for node in planning_graph.nodes:
        if node.id in observed_states:
            node_atoms[node.id] = set(node.atoms)

    approach.training_mode = False

    return GoalConditionedTrainingData(
        states=states,
        current_atoms=current_atoms_list,
        goal_atoms=goal_atoms_list,
        config={
            "collection_method": "single_episode",
            "num_shortcuts_collected": len(valid_shortcuts),
        },
        node_states=observed_states,
        valid_shortcuts=valid_shortcuts,
        node_atoms=node_atoms,
    )


def collect_total_planning_graph(
    system: ImprovisationalTAMPSystem,
    collect_episodes: int,
    seed: int = 42,
    planner_id: str = "pyperplan",
    compute_costs: bool = True,
    cost_samples: int = 25,
    max_steps_per_edge: int = 100,
) -> tuple[PlanningGraph, dict[frozenset, list]]:
    """Build a unified planning graph across multiple episodes.

    This creates a single, comprehensive planning graph by running BFS
    collection across multiple random episodes. Unlike per-episode graphs,
    this gives a consistent atom-based representation that doesn't depend
    on episode-specific node IDs.

    Args:
        system: The TAMP system
        collect_episodes: Number of episodes to collect across
        seed: Random seed for environment resets
        planner_id: Planner to use for BFS graph construction
        compute_costs: Whether to compute edge costs after building graph
        cost_samples: Number of samples per edge for cost computation
        max_steps_per_edge: Maximum steps for edge execution during cost computation

    Returns:
        Tuple of:
        - total_graph: Unified planning graph with all discovered nodes/edges
        - node_states: Dict mapping frozenset[atoms] -> list of states
    """
    rng = np.random.default_rng(seed)
    total_graph = PlanningGraph()
    node_states: dict[frozenset, list] = {}

    print(f"Building total planning graph from {collect_episodes} episodes...")

    for episode_idx in range(collect_episodes):
        print(f"\n=== Episode {episode_idx + 1}/{collect_episodes} ===")

        # Reset environment with random seed
        obs, info = system.reset(seed=int(rng.integers(0, 2**31)))
        objects, atoms, goal = system.perceiver.reset(obs, info)

        # Create a temporary approach just for this episode's planning graph
        from tamp_improv.approaches.improvisational.policies.base import Policy

        # Create a dummy policy (not used for collection)
        class DummyPolicy(Policy):
            def can_initiate(self, obs, context):
                return False

            def step(self, obs, context):
                raise NotImplementedError

        dummy_policy = MultiRLPolicy(seed=seed)
        temp_approach = ImprovisationalTAMPApproach(
            system=system,
            policy=dummy_policy,
            seed=seed + episode_idx,
            planner_id=planner_id,
        )
        temp_approach.training_mode = True

        # Build planning graph to determine nodes and edges
        temp_approach._goal = goal
        episode_graph = temp_approach._create_planning_graph(objects, atoms)
        temp_approach.planning_graph = episode_graph

        # Collect states using BFS exploration (returns {frozenset(atoms): [states]})
        # This fills in the actual low-level states for nodes discovered in episode_graph
        episode_states = collect_states_for_all_nodes(
            system, episode_graph, max_steps_per_skill=max_steps_per_edge
        )

        print(f"Episode graph has {len(episode_graph.nodes)} nodes, {len(episode_graph.edges)} edges")
        print(f"BFS collected states for {len(episode_states)} atom sets")

        # Merge nodes into total graph
        for node in episode_graph.nodes:
            # Get or create node in total graph (atom-based, so consistent across episodes)
            total_node = total_graph.get_or_add_node(set(node.atoms))

            # Accumulate states for this atom set
            if total_node.atoms not in node_states:
                node_states[total_node.atoms] = []

            # Add states from this episode (episode_states uses frozenset(atoms) as keys)
            node_atoms_frozen = frozenset(node.atoms)
            if node_atoms_frozen in episode_states:
                for new_state in episode_states[node_atoms_frozen]:
                    # Check if this state is a duplicate
                    is_duplicate = False
                    if hasattr(new_state, "nodes"):
                        for existing_state in node_states[total_node.atoms]:
                            if hasattr(existing_state, "nodes") and np.array_equal(
                                existing_state.nodes, new_state.nodes
                            ):
                                is_duplicate = True
                                break
                    elif isinstance(new_state, np.ndarray):
                        for existing_state in node_states[total_node.atoms]:
                            if isinstance(existing_state, np.ndarray) and np.array_equal(
                                existing_state, new_state
                            ):
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        node_states[total_node.atoms].append(new_state)

        # Merge edges (only add if not already present)
        for edge in episode_graph.edges:
            source_atoms = edge.source.atoms
            target_atoms = edge.target.atoms

            # Get the corresponding nodes in total graph
            source_node = total_graph.node_map[source_atoms]
            target_node = total_graph.node_map[target_atoms]

            # Check if this edge already exists
            edge_exists = any(
                e.source == source_node
                and e.target == target_node
                and e.operator == edge.operator
                for e in total_graph.edges
            )

            if not edge_exists:
                total_graph.add_edge(
                    source=source_node,
                    target=target_node,
                    operator=edge.operator,
                    cost=edge.cost,
                    is_shortcut=edge.is_shortcut,
                )

        if (episode_idx + 1) % 5 == 0 or episode_idx == collect_episodes - 1:
            print(
                f"  Episode {episode_idx + 1}/{collect_episodes}: "
                f"{len(total_graph.nodes)} nodes, {len(total_graph.edges)} edges"
            )

    print(
        f"Total graph complete: {len(total_graph.nodes)} nodes, "
        f"{len(total_graph.edges)} edges, "
        f"{sum(len(states) for states in node_states.values())} total states"
    )

    # Print node atoms for understanding
    print("\nNode ID -> Atoms mapping:")
    print("=" * 80)
    for node in sorted(total_graph.nodes, key=lambda n: n.id):
        atoms_str = ", ".join(sorted([str(atom) for atom in node.atoms]))
        if not atoms_str:
            atoms_str = "(empty - initial state)"
        num_states = len(node_states.get(node.atoms, []))
        print(f"  Node {node.id:3d} ({num_states:2d} states): {atoms_str}")
    print("=" * 80)

    # Compute edge costs if requested
    if compute_costs:
        from tamp_improv.approaches.improvisational.analyze import compute_all_edge_costs

        compute_all_edge_costs(
            system=system,
            graph=total_graph,
            node_states=node_states,
            num_samples=cost_samples,
            max_steps=max_steps_per_edge,
        )

    return total_graph, node_states
