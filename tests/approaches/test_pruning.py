"""Tests for pruning module."""

import numpy as np
import pytest

from tamp_improv.approaches.improvisational.graph import (
    GroundAtom,
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.benchmarks.base import Predicate
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.approaches.improvisational.pruning import (
    prune_none,
    prune_random,
    prune_training_data,
    prune_with_rollouts,
)
from tamp_improv.approaches.improvisational.collection import collect_total_shortcuts
from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from tamp_improv.benchmarks.gridworld import GridworldTAMPSystem

def _make_test_atom(name: str) -> GroundAtom:
    """Create a test ground atom with a given name."""
    pred = Predicate(name, [])  # Empty list for 0-arity predicate
    return GroundAtom(pred, [])


def test_prune_none():
    """Test that prune_none returns all shortcuts unchanged."""
    # Create minimal training data
    training_data = GoalConditionedTrainingData(
        states=["state1", "state2", "state3"],
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("b")}, {_make_test_atom("c")}],
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("c")}, {_make_test_atom("d")}],
        config={},
        node_states={0: ["state1"], 1: ["state2"], 2: ["state3"]},
        valid_shortcuts=[(0, 1), (0, 2), (1, 2)],
        node_atoms={0: {_make_test_atom("a")}, 1: {_make_test_atom("b")}, 2: {_make_test_atom("c")}},
    )

    result = prune_none(training_data)

    assert len(result.states) == 3
    assert len(result.valid_shortcuts) == 3
    assert result.states == training_data.states


def test_prune_random():
    """Test that prune_random selects correct number of shortcuts."""
    rng = np.random.default_rng(42)

    # Create training data with 10 shortcuts
    training_data = GoalConditionedTrainingData(
        states=[f"state{i}" for i in range(10)],
        current_atoms=[{_make_test_atom(f"a{i}")} for i in range(10)],
        goal_atoms=[{_make_test_atom(f"b{i}")} for i in range(10)],
        config={},
        node_states={i: [f"state{i}"] for i in range(10)},
        valid_shortcuts=[(i, i + 1) for i in range(10)],
        node_atoms={i: {_make_test_atom(f"a{i}")} for i in range(10)},
    )

    # Prune to 5 shortcuts
    result = prune_random(training_data, max_shortcuts=5, rng=rng)

    assert len(result.states) == 5
    assert len(result.valid_shortcuts) == 5
    assert result.config["pruning_method"] == "random"
    assert result.config["max_shortcuts"] == 5


def test_prune_random_under_limit():
    """Test that prune_random keeps all shortcuts if under limit."""
    rng = np.random.default_rng(42)

    # Create training data with 3 shortcuts
    training_data = GoalConditionedTrainingData(
        states=["state1", "state2", "state3"],
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("b")}, {_make_test_atom("c")}],
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("c")}, {_make_test_atom("d")}],
        config={},
        node_states={0: ["state1"], 1: ["state2"], 2: ["state3"]},
        valid_shortcuts=[(0, 1), (0, 2), (1, 2)],
        node_atoms={0: {_make_test_atom("a")}, 1: {_make_test_atom("b")}, 2: {_make_test_atom("c")}},
    )

    # Try to prune to 10 shortcuts (but only 3 exist)
    result = prune_random(training_data, max_shortcuts=10, rng=rng)

    assert len(result.states) == 3
    assert len(result.valid_shortcuts) == 3


def test_prune_training_data_dispatch():
    """Test that prune_training_data dispatches to correct method."""
    rng = np.random.default_rng(42)

    # Create minimal training data with 2 NODE PAIRS (0->1 and 1->2)
    training_data = GoalConditionedTrainingData(
        states=["state1", "state2"],
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("b")}],
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("c")}],
        config={},
        node_states={0: ["state1"], 1: ["state2"], 2: ["state3"]},
        valid_shortcuts=[(0, 1), (1, 2)],  # 2 different node pairs
        node_atoms={0: {_make_test_atom("a")}, 1: {_make_test_atom("b")}, 2: {_make_test_atom("c")}},
    )

    # Create minimal planning graph
    node0 = PlanningGraphNode(id=0, atoms=frozenset([_make_test_atom("a")]))
    node1 = PlanningGraphNode(id=1, atoms=frozenset([_make_test_atom("b")]))
    node2 = PlanningGraphNode(id=2, atoms=frozenset([_make_test_atom("c")]))
    planning_graph = PlanningGraph()
    planning_graph.nodes = [node0, node1, node2]

    # Mock system (not used for 'none' pruning)
    system = None

    # Test 'none' method
    config = {"pruning_method": "none", "seed": 42}
    result = prune_training_data(training_data, system, planning_graph, config, rng)
    assert len(result.states) == 2

    # Test 'random' method - prune to 1 node pair
    config = {"pruning_method": "random", "max_shortcuts_per_graph": 1, "seed": 42}
    result = prune_training_data(training_data, system, planning_graph, config, rng)
    assert len(result.states) == 1  # Should keep 1 state (from 1 node pair)
    assert result.config["pruning_method"] == "random"


def test_prune_training_data_unknown_method():
    """Test that unknown pruning method raises error."""
    rng = np.random.default_rng(42)

    # Create minimal training data
    training_data = GoalConditionedTrainingData(
        states=["state1"],
        current_atoms=[{_make_test_atom("a")}],
        goal_atoms=[{_make_test_atom("b")}],
        config={},
        node_states={0: ["state1"]},
        valid_shortcuts=[(0, 1)],
        node_atoms={0: {_make_test_atom("a")}},
    )

    # Create minimal planning graph
    node0 = PlanningGraphNode(id=0, atoms=frozenset([_make_test_atom("a")]))
    planning_graph = PlanningGraph()
    planning_graph.nodes = [node0]

    config = {"pruning_method": "invalid_method", "seed": 42}

    with pytest.raises(ValueError, match="Unknown pruning method"):
        prune_training_data(training_data, None, planning_graph, config, rng)


def test_pruning_preserves_multiple_states_per_node_pair():
    """Test that pruning operates on node pairs, not state pairs.

    If a node pair (A, B) has multiple training examples (different states),
    then pruning should either keep ALL of them or remove ALL of them.
    It should NOT split them up by treating each state pair independently.
    """
    rng = np.random.default_rng(42)

    # Create training data where node pair (0, 1) has 2 state pairs
    # This simulates: node 0 has 2 states, node 1 has 1 state
    training_data = GoalConditionedTrainingData(
        states=["state0a", "state0b"],  # 2 states for the same shortcut
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("a")}],  # Same source node
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("b")}],  # Same target node
        config={"shortcut_info": [
            {"source_node_id": 0, "target_node_id": 1},
            {"source_node_id": 0, "target_node_id": 1},
        ]},
        node_states={
            0: ["state0a", "state0b"],  # Node 0 has 2 states
            1: ["state1"],  # Node 1 has 1 state
        },
        valid_shortcuts=[(0, 1), (0, 1)],  # Same node pair appears twice (once per state)
        node_atoms={
            0: {_make_test_atom("a")},
            1: {_make_test_atom("b")},
        },
    )

    # If we prune randomly to keep just 1 "shortcut", we should either:
    # - Keep BOTH state pairs (because they're for the same node pair), OR
    # - Remove BOTH state pairs
    # We should NOT end up with just 1 state pair

    # The correct behavior: since both entries represent the same node pair (0, 1),
    # they should be treated as a group
    result = prune_random(training_data, max_shortcuts=1, rng=rng)

    # After fixing the bug, this should keep 0 or 2 states, not 1
    # (Because pruning by node pair means all states for a node pair move together)
    print(f"Result has {len(result.states)} states")
    print(f"Result has {len(result.valid_shortcuts)} shortcuts")

    # The bug would cause this to be 1 (treating state pairs independently)
    # The correct behavior should give us 2 (keeping all states for the node pair)
    # or 0 (rejecting the entire node pair)
    assert len(result.states) in [0, 2], \
        f"Expected 0 or 2 states (pruning by node pair), got {len(result.states)}"

def test_prune_total_shortcuts():
    """Test collect_total_shortcuts on gridworld."""
    print("\n" + "=" * 80)
    print("TESTING COLLECT_TOTAL_SHORTCUTS")
    print("=" * 80)

    # Set up gridworld system
    config = {
        "seed": 43,
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

    
    # prune with rollouts
    print("\nPruning training data with rollouts...")

    # Count state pairs per node pair before pruning
    from collections import Counter
    original_node_pair_counts = Counter(train_data.valid_shortcuts)
    print(f"\nNode pair distribution before pruning:")
    for (src, tgt), count in sorted(original_node_pair_counts.items()):
        print(f"  ({src}, {tgt}): {count} state pairs")

    # Configure rollout pruning
    planning_graph = train_data.graph
    pruning_config = {
        **config,
        "num_rollouts_per_node": 1,
        "max_steps_per_rollout": 20,
        "shortcut_success_threshold": 1,
    }

    # Apply rollout pruning
    pruned_data = prune_with_rollouts(
        train_data,
        system,
        planning_graph,
        pruning_config,
        rng,
    )

    print(f"\nPruning Results:")
    print(f"  Kept {len(set(pruned_data.valid_shortcuts))} unique node pairs")
    print(f"  Total state pairs: {len(pruned_data.states)}")

    # Count state pairs per node pair after pruning
    pruned_node_pair_counts = Counter(pruned_data.valid_shortcuts)
    print(f"\nNode pair distribution after pruning:")
    for (src, tgt), count in sorted(pruned_node_pair_counts.items()):
        original_count = original_node_pair_counts[(src, tgt)]
        print(f"  ({src}, {tgt}): {count} state pairs (was {original_count})")

        # KEY ASSERTION: If a node pair survived pruning,
        # it should have the SAME number of state pairs as before
        assert count == original_count, \
            f"Node pair ({src}, {tgt}) had {original_count} state pairs before " \
            f"pruning but {count} after - state pairs were split!"

    print("\n✓ test_prune_total_shortcuts passed!")

if __name__ == "__main__":
    # test_prune_none()
    # test_prune_random()
    # test_prune_random_under_limit()
    # test_prune_training_data_dispatch()
    # test_prune_training_data_unknown_method()
    # test_pruning_preserves_multiple_states_per_node_pair()
    test_prune_total_shortcuts()
    print("\nAll tests passed!")