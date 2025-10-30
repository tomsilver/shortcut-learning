"""Test V2 collection pipeline."""

import pytest

from shortcut_learning.configs import ApproachConfig, CollectionConfig, PolicyConfig
from shortcut_learning.methods.collection_v2 import (
    collect_diverse_states_per_node,
    collect_training_data_v2,
    select_shortcut_pairs,
)
from shortcut_learning.methods.policies.multi_rl_ppo_v2 import MultiRLPolicyV2
from shortcut_learning.methods.slap_approach_v2 import SLAPApproachV2
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem
from shortcut_learning.problems.obstacle2d_hard.system import (
    BaseObstacle2DTAMPSystem as BaseObstacle2DHardTAMPSystem,
)

@pytest.mark.parametrize("system_cls", [BaseObstacle2DTAMPSystem, BaseObstacle2DHardTAMPSystem])
def test_collect_diverse_states_per_node(system_cls):
    """Test that we can collect multiple states per node."""
    # Setup
    system = system_cls.create_default(seed=42)
    approach_config = ApproachConfig(
        approach_type="slap",
        approach_name="test_v2",
        debug_videos=False,
        seed=42,
    )
    policy_config = PolicyConfig(policy_type="rl_ppo")
    policy = MultiRLPolicyV2(seed=42, config=policy_config)
    approach = SLAPApproachV2(system, approach_config, policy)

    # Build planning graph (don't use reset() since it tries to find a path)
    obs, info = system.reset()
    if not approach.graph_built:
        approach.planning_graph = approach._create_planning_graph(obs, info)
        approach.graph_built = True

    assert approach.planning_graph is not None
    num_nodes = len(approach.planning_graph.nodes)
    print(f"\nPlanning graph has {num_nodes} nodes")

    # Use smaller parameters for larger graphs (like obstacle_tower)
    # This keeps unit tests fast while still verifying functionality
    states_per_node = 1 if num_nodes > 50 else 2
    print(f"Using states_per_node={states_per_node}")

    # Collect states
    collect_diverse_states_per_node(
        approach,
        states_per_node=states_per_node,
    )

    # Verify states were collected
    nodes_with_states = 0
    total_states = 0
    for node in approach.planning_graph.nodes:
        if node.states:
            nodes_with_states += 1
            total_states += len(node.states)
            print(f"Node {node.id}: {len(node.states)} states")

    print(f"\nTotal: {nodes_with_states}/{num_nodes} nodes have states")
    print(f"Total states collected: {total_states}")

    # At least the initial node should have states
    assert nodes_with_states > 0, "No states collected"
    assert total_states > 0, "No states collected"

    # Initial node should have at least 1 state
    initial_node = approach.planning_graph.nodes[0]
    assert len(initial_node.states) > 0, "Initial node has no states"


@pytest.mark.parametrize("system_cls", [BaseObstacle2DTAMPSystem])
def test_select_shortcut_pairs(system_cls):
    """Test shortcut pair selection."""
    import numpy as np

    # Setup
    system = system_cls.create_default(seed=42)
    approach_config = ApproachConfig(
        approach_type="slap",
        approach_name="test_v2",
        debug_videos=False,
        seed=42,
    )
    policy_config = PolicyConfig(policy_type="rl_ppo")
    policy = MultiRLPolicyV2(seed=42, config=policy_config)
    approach = SLAPApproachV2(system, approach_config, policy)

    # Build planning graph (don't use reset() since it tries to find a path)
    obs, info = system.reset()
    if not approach.graph_built:
        approach.planning_graph = approach._create_planning_graph(obs, info)
        approach.graph_built = True

    # Collect states (use smaller parameters for larger graphs)
    num_nodes = len(approach.planning_graph.nodes)
    states_per_node = 1 if num_nodes > 50 else 2
    collect_diverse_states_per_node(approach, states_per_node=states_per_node)

    # Select shortcuts
    rng = np.random.default_rng(42)
    max_shortcuts = 5
    shortcut_pairs = select_shortcut_pairs(
        approach.planning_graph,
        max_shortcuts=max_shortcuts,
        rng=rng,
    )

    print(f"\nSelected {len(shortcut_pairs)} shortcut pairs")
    for source, target in shortcut_pairs:
        print(f"  {source.id} -> {target.id}")

    # Verify shortcuts are valid
    assert len(shortcut_pairs) > 0, "No shortcuts selected"
    assert len(shortcut_pairs) <= max_shortcuts, "Too many shortcuts selected"

    for source, target in shortcut_pairs:
        # Source should come before target
        assert source.id < target.id, f"Invalid pair: {source.id} -> {target.id}"

        # Both should have states
        assert len(source.states) > 0, f"Source node {source.id} has no states"
        assert len(target.states) > 0, f"Target node {target.id} has no states"

        # Should not be a direct regular edge
        has_direct_edge = any(
            edge.target == target and not edge.is_shortcut
            for edge in approach.planning_graph.node_to_outgoing_edges.get(source, [])
        )
        assert not has_direct_edge, f"Direct edge exists: {source.id} -> {target.id}"


@pytest.mark.parametrize("system_cls", [BaseObstacle2DTAMPSystem])
def test_collect_training_data_v2_full_pipeline(system_cls):
    """Test the full V2 collection pipeline."""
    # Setup
    system = system_cls.create_default(seed=42)
    approach_config = ApproachConfig(
        approach_type="slap",
        approach_name="test_v2",
        debug_videos=False,
        seed=42,
    )
    policy_config = PolicyConfig(policy_type="rl_ppo")
    policy = MultiRLPolicyV2(seed=42, config=policy_config)
    approach = SLAPApproachV2(system, approach_config, policy)

    # Build planning graph (don't use reset() since it tries to find a path)
    obs, info = system.reset()
    if not approach.graph_built:
        approach.planning_graph = approach._create_planning_graph(obs, info)
        approach.graph_built = True

    # Collect training data with minimal parameters for fast unit tests
    num_nodes = len(approach.planning_graph.nodes)
    states_per_node = 1 if num_nodes > 50 else 2

    collection_config = CollectionConfig(
        seed=42,
        states_per_node=states_per_node,
        max_shortcuts_per_graph=3,
    )

    # Use a temporary file for caching
    import tempfile
    import os
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
    os.close(tmp_fd)  # Close the file descriptor

    training_data = collect_training_data_v2(approach, collection_config, save_path=tmp_path)

    print(f"\nCollection complete!")
    print(f"  Number of shortcuts: {len(training_data)}")
    print(f"  Total training examples: {training_data.num_training_examples()}")

    # Verify training data
    assert len(training_data) > 0, "No shortcuts collected"
    assert training_data.num_training_examples() > 0, "No training examples"

    # Check each shortcut
    for idx, (source_node, target_node) in enumerate(training_data.shortcuts):
        print(f"  Shortcut {idx}: {source_node.id} -> {target_node.id}")
        print(f"    Source states: {len(source_node.states)}")
        print(f"    Target states: {len(target_node.states)}")

        assert len(source_node.states) > 0, f"Shortcut {idx} has no source states"
        assert source_node.id < target_node.id, f"Invalid shortcut {idx}"

    # Check config
    assert "num_shortcuts" in training_data.config
    assert training_data.config["num_shortcuts"] == len(training_data)


@pytest.mark.parametrize(
    "states_per_node",
    [
        (100),   # Multi-start BFS with 100 episodes (matches experiment)
    ],
)
@pytest.mark.parametrize("system_cls", [BaseObstacle2DTAMPSystem, BaseObstacle2DHardTAMPSystem])
def test_states_per_node_ablation(system_cls, states_per_node):
    """Ablation study: Test how many states we can actually collect per node.

    This test helps us understand the practical limits of state collection and
    whether requesting 50 states per node is achievable.
    """
    # Setup
    system = system_cls.create_default(seed=42)
    approach_config = ApproachConfig(
        approach_type="slap",
        approach_name="test_ablation",
        debug_videos=False,
        seed=42,
    )
    policy_config = PolicyConfig(policy_type="rl_ppo")
    policy = MultiRLPolicyV2(seed=42, config=policy_config)
    approach = SLAPApproachV2(system, approach_config, policy)

    # Build planning graph
    obs, info = system.reset()
    if not approach.graph_built:
        approach.planning_graph = approach._create_planning_graph(obs, info)
        approach.graph_built = True

    num_nodes = len(approach.planning_graph.nodes)
    strategy = "MULTI-START"
    print(f"\n{'='*70}")
    print(f"ABLATION: {strategy}, states_per_node={states_per_node}")
    print(f"{'='*70}")

    # Collect states
    collect_diverse_states_per_node(
        approach,
        states_per_node=states_per_node,
    )

    # Analyze results
    states_by_node = {}
    total_states = 0
    min_states = float('inf')
    max_states = 0

    for node in approach.planning_graph.nodes:
        num_states = len(node.states)
        states_by_node[node.id] = num_states
        total_states += num_states
        min_states = min(min_states, num_states)
        max_states = max(max_states, num_states)

    avg_states = total_states / num_nodes

    print(f"\nResults:")
    print(f"  Requested: {states_per_node} states/node")
    print(f"  Collected: {total_states} total states across {num_nodes} nodes")
    print(f"  Average:   {avg_states:.2f} states/node")
    print(f"  Min:       {min_states} states/node")
    print(f"  Max:       {max_states} states/node")
    print(f"  Success rate: {avg_states/states_per_node*100:.1f}%")

    print(f"\nPer-node breakdown:")
    for node_id, num_states in states_by_node.items():
        progress_bar = "█" * num_states + "░" * (states_per_node - num_states)
        print(f"  Node {node_id}: {num_states:3d}/{states_per_node} {progress_bar[:50]}")

    # Assertions: We should get at least some states for each node
    assert min_states > 0, f"Node with 0 states found (min={min_states})"

    # We should achieve at least 20% of requested states on average
    # (this is a loose bound to catch catastrophic failures)
    # assert avg_states >= states_per_node * 0.2, \
    #     f"Collection too inefficient: {avg_states:.1f}/{states_per_node} = {avg_states/states_per_node*100:.1f}%"


@pytest.mark.parametrize("system_cls", [BaseObstacle2DTAMPSystem])
def test_planning_graph_nodes_have_states_field(system_cls):
    """Test that planning graph nodes have the new states field."""
    system = system_cls.create_default(seed=42)
    approach_config = ApproachConfig(
        approach_type="slap",
        approach_name="test_v2",
        debug_videos=False,
        seed=42,
    )
    policy_config = PolicyConfig(policy_type="rl_ppo")
    policy = MultiRLPolicyV2(seed=42, config=policy_config)
    approach = SLAPApproachV2(system, approach_config, policy)

    # Build planning graph (first reset builds it)
    obs, info = system.reset()

    # Just build the graph directly without running full reset
    if not approach.graph_built:
        approach.planning_graph = approach._create_planning_graph(obs, info)
        approach.graph_built = True

    # Check that nodes have states field
    assert approach.planning_graph is not None
    for node in approach.planning_graph.nodes:
        assert hasattr(node, "states"), f"Node {node.id} missing states field"
        assert isinstance(node.states, list), f"Node {node.id} states is not a list"


if __name__ == "__main__":
    # Run tests manually
    print("=" * 60)
    print("Test 1: Collect diverse states per node")
    print("=" * 60)
    test_collect_diverse_states_per_node(BaseObstacle2DTAMPSystem)

    print("\n" + "=" * 60)
    print("Test 2: Select shortcut pairs")
    print("=" * 60)
    test_select_shortcut_pairs(BaseObstacle2DTAMPSystem)

    print("\n" + "=" * 60)
    print("Test 3: Full collection pipeline")
    print("=" * 60)
    test_collect_training_data_v2_full_pipeline(BaseObstacle2DTAMPSystem)

    print("\n" + "=" * 60)
    print("Test 4: Nodes have states field")
    print("=" * 60)
    test_planning_graph_nodes_have_states_field(BaseObstacle2DTAMPSystem)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
