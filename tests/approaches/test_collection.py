"""Tests for collection module."""

import pytest
import numpy as np

from tamp_improv.approaches.improvisational.collection import (
    collect_all_shortcuts,
    collect_shortcuts_single_episode,
    collect_total_planning_graph,
    collect_total_shortcuts,
)
from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem


def test_collection_imports():
    """Test that collection functions can be imported."""
    assert callable(collect_all_shortcuts)
    assert callable(collect_shortcuts_single_episode)


def test_collect_and_group_shortcuts():
    """Test collection and grouping to understand policy key generation."""
    # Set up system and approach
    config = {
        "seed": 42,
        "collect_episodes": 10,
        "n_blocks": 2,
        "render_mode": None,
    }

    system = GraphObstacle2DTAMPSystem.create_default(
        n_blocks=config["n_blocks"],
        seed=config["seed"],
        render_mode=config.get("render_mode"),
    )

    policy = MultiRLPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])
    approach.training_mode = True

    # Collect shortcuts
    print("\n" + "=" * 80)
    print("COLLECTING SHORTCUTS")
    print("=" * 80)

    rng = np.random.default_rng(config["seed"])
    train_data = collect_all_shortcuts(system, approach, config, rng=rng)

    print(f"\nCollection complete:")
    print(f"  Total training examples: {len(train_data.states)}")
    print(f"  Valid shortcuts: {len(train_data.valid_shortcuts)}")
    print(f"  Node states: {len(train_data.node_states)}")
    print(f"  Node atoms: {len(train_data.node_atoms)}")

    # Check shortcut_info
    shortcut_info = train_data.config.get("shortcut_info", [])
    print(f"  Shortcut info entries: {len(shortcut_info)}")

    # Show first few shortcut_info entries
    print("\nFirst 10 shortcut_info entries:")
    for i, info in enumerate(shortcut_info[:10]):
        print(f"  {i}: {info}")

    print("Number of states:", len(train_data.states))
    for i, states in train_data.node_states.items():
        print("Number of states for node", i, ":", len(states))
    # Test pruning to see if it breaks shortcut_info alignment
    print("\n" + "=" * 80)
    print("PRUNING WITH ROLLOUTS")
    print("=" * 80)

    from tamp_improv.approaches.improvisational.pruning import prune_training_data

    # Use rollout pruning like the pipeline does
    pruning_config = {**config, "pruning_method": "rollouts"}
    pruned_data = prune_training_data(
        train_data,
        system,
        approach.planning_graph,
        pruning_config,
        rng
    )

    # pruned_data = train_data

    print(f"\nAfter pruning:")
    print(f"  Training examples: {len(pruned_data.states)}")
    print(f"  Shortcut info entries: {len(pruned_data.config.get('shortcut_info', []))}")

    # Now test grouping with MultiRLPolicy on PRUNED data
    print("\n" + "=" * 80)
    print("GROUPING WITH MultiRLPolicy (AFTER PRUNING)")
    print("=" * 80)

    policy = MultiRLPolicy(system.env)

    # Access the private method to see grouping
    grouped = policy._group_training_data(pruned_data)

    print(f"\nGrouping results:")
    print(f"  Number of unique policy keys: {len(grouped)}")
    print(f"  Total training examples: {len(pruned_data.states)}")
    print(f"  Average examples per policy: {len(pruned_data.states) / len(grouped):.2f}")

    # Show all policy keys and their counts
    print("\nPolicy keys and example counts:")
    for key, group_data in grouped.items():
        num_examples = len(group_data.states)
        print(f"  {key}: {num_examples} examples")

    # Group by node pair to see if there are collisions
    node_pair_to_keys = {}
    for key in grouped.keys():
        # Extract node pair from key (format: n{source}-to-n{target}_{hash1}_{hash2})
        if key.startswith("n"):
            node_pair = key.split("_")[0]  # Get "nX-to-nY" part
            if node_pair not in node_pair_to_keys:
                node_pair_to_keys[node_pair] = []
            node_pair_to_keys[node_pair].append(key)

    print(f"\nNode pairs with multiple policy keys:")
    for node_pair, keys in node_pair_to_keys.items():
        if len(keys) > 1:
            print(f"  {node_pair}: {len(keys)} different keys")
            for key in keys:
                num_examples = len(grouped[key].states)
                print(f"    - {key}: {num_examples} examples")

    # Verify we have training data
    print("Number of states:", len(train_data.states))
    for i, states in train_data.node_states.items():
        print("Number of states for node", i, ":", len(states))

    assert len(train_data.states) > 0
    assert len(grouped) > 0


def test_collect_total_planning_graph():
    """Test that collect_total_planning_graph builds a unified graph across episodes."""
    print("\n" + "=" * 80)
    print("TESTING COLLECT_TOTAL_PLANNING_GRAPH")
    print("=" * 80)

    # Set up system
    config = {
        "seed": 42,
        "collect_episodes": 20,
        "n_blocks": 2,
        "render_mode": None,
    }

    # system = GraphObstacle2DTAMPSystem.create_default(
    #     n_blocks=config["n_blocks"],
    #     seed=config["seed"],
    #     render_mode=config.get("render_mode"),
    # )
    
    # Use a gridworld system
    system = GridworldTAMPSystem.create_default(
        num_cells=10,
        num_states_per_cell=10,
        num_teleporters=1,
        seed=config["seed"],
    )

    # Collect total planning graph
    total_graph, node_states = collect_total_planning_graph(
        system=system,
        collect_episodes=config["collect_episodes"],
        seed=config["seed"],
        planner_id="pyperplan"
    )

    print(f"\nTotal Graph Stats:")
    print(f"  Nodes: {len(total_graph.nodes)}")
    print(f"  Edges: {len(total_graph.edges)}")
    print(f"  Unique atom sets: {len(node_states)}")
    print(f"  Total states collected: {sum(len(states) for states in node_states.values())}")

    # Verify basic properties
    assert len(total_graph.nodes) > 0, "Should have collected some nodes"
    assert len(total_graph.edges) > 0, "Should have collected some edges"
    assert len(node_states) > 0, "Should have states for nodes"

    # Verify node_states keys match graph nodes (both use frozenset[atoms])
    graph_atom_sets = {node.atoms for node in total_graph.nodes}
    state_atom_sets = set(node_states.keys())

    print(f"\nAtom Set Consistency Check:")
    print(f"  Graph atom sets: {len(graph_atom_sets)}")
    print(f"  State atom sets: {len(state_atom_sets)}")
    print(f"  Intersection: {len(graph_atom_sets & state_atom_sets)}")

    # All state atom sets should be in the graph
    assert state_atom_sets.issubset(graph_atom_sets), \
        "All states should correspond to nodes in the graph"

    # Show some example atom sets and state counts
    print(f"\nExample atom sets and state counts:")
    for i, (atoms, states) in enumerate(list(node_states.items())[:5]):
        print(f"  {i+1}. {len(atoms)} atoms, {len(states)} states")
        # Show first few atoms
        for j, atom in enumerate(list(atoms)[:3]):
            print(f"     - {atom}")
        if len(atoms) > 3:
            print(f"     - ... ({len(atoms) - 3} more)")

    print(node_states[list(node_states.keys())[0]])

    # Verify states accumulate across episodes (should have multiple states per node)
    states_per_node = [len(states) for states in node_states.values()]
    avg_states = sum(states_per_node) / len(states_per_node)
    max_states = max(states_per_node)

    print(f"\nStates per node statistics:")
    print(f"  Average: {avg_states:.2f}")
    print(f"  Maximum: {max_states}")
    print(f"  Minimum: {min(states_per_node)}")
    print(states_per_node)

    # With 5 episodes, we expect many nodes to have multiple states
    nodes_with_multiple_states = sum(1 for count in states_per_node if count > 1)
    print(f"  Nodes with >1 state: {nodes_with_multiple_states}/{len(states_per_node)}")

    # Verify that the graph uses atom-based identity
    print(f"\nVerifying atom-based node identity:")
    for node in list(total_graph.nodes)[:3]:
        # Check that we can look up the node by its atoms
        assert node.atoms in total_graph.node_map, \
            "Node should be in node_map by atoms"
        assert total_graph.node_map[node.atoms] == node, \
            "node_map lookup should return the same node"
        print(f"  ✓ Node {node.id} with {len(node.atoms)} atoms: lookup works")

    print("\n✓ Total planning graph collection test passed!")


def test_collect_total_shortcuts():
    """Test collect_total_shortcuts on gridworld."""
    print("\n" + "=" * 80)
    print("TESTING COLLECT_TOTAL_SHORTCUTS")
    print("=" * 80)

    # Set up gridworld system
    config = {
        "seed": 42,
        "collect_episodes": 5,
        "planner_id": "pyperplan",
    }

    # system = GraphObstacle2DTAMPSystem.create_default()

    system = GridworldTAMPSystem.create_default(
        num_cells=10,
        num_states_per_cell=10,
        num_teleporters=1,
        seed=config["seed"],
    )

    # Create approach
    policy = MultiRLPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])

    # Collect total shortcuts
    rng = np.random.default_rng(config["seed"])
    train_data = collect_total_shortcuts(system, approach, config, rng=rng)

    print(f"\nCollection Results:")
    print(f"  Total training examples: {len(train_data.states)}")
    print(f"  Valid shortcuts: {len(train_data.valid_shortcuts)}")
    print(f"  Node states: {len(train_data.node_states)}")
    print(f"  Node atoms: {len(train_data.node_atoms)}")
    print(f"  Collection method: {train_data.config.get('collection_method')}")

    # Verify basic properties
    assert len(train_data.states) > 0, "Should have training examples"
    assert len(train_data.valid_shortcuts) > 0, "Should have shortcuts"
    assert len(train_data.node_states) > 0, "Should have node states"
    assert len(train_data.node_atoms) > 0, "Should have node atoms"
    assert train_data.config.get("collection_method") == "total_shortcuts"

    # Verify data consistency
    assert len(train_data.states) == len(train_data.current_atoms), \
        "States and current_atoms should match"
    assert len(train_data.states) == len(train_data.goal_atoms), \
        "States and goal_atoms should match"

    # Verify shortcut_info matches states
    shortcut_info = train_data.config.get("shortcut_info", [])
    assert len(shortcut_info) == len(train_data.states), \
        "Shortcut info should match number of training examples"

    # Show some example shortcuts
    print(f"\nFirst 5 shortcuts:")
    for i in range(min(5, len(train_data.valid_shortcuts))):
        source_id, target_id = train_data.valid_shortcuts[i]
        print(f"  {i+1}. Node {source_id} -> Node {target_id}")

    # Check that approach has the planning graph stored
    assert approach.planning_graph is not None, "Approach should have planning graph"
    print(f"\nApproach planning graph: {len(approach.planning_graph.nodes)} nodes, "
          f"{len(approach.planning_graph.edges)} edges")

    # Check trained signatures were registered
    print(f"Trained signatures registered: {len(approach.trained_signatures)}")

    # Verify node_states format (should be dict[int, list[obs]])
    for node_id, states in list(train_data.node_states.items())[:3]:
        assert isinstance(node_id, int), "Node states should be keyed by int"
        assert isinstance(states, list), "States should be a list"
        print(f"  Node {node_id}: {len(states)} states")

    print("\n✓ collect_total_shortcuts test passed!")


# Note: Full integration tests would require setting up a real TAMP system
# and approach, which is more appropriate for integration tests.
# For now, we just verify the functions exist and are callable.

if __name__ == "__main__":
    # test_collection_imports()
    # test_collect_and_group_shortcuts()
    # test_collect_total_planning_graph()
    test_collect_total_shortcuts()
    print("\nAll tests passed!")