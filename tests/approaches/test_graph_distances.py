"""Tests for graph distance computation."""

import pytest
from relational_structs import GroundAtom, Object, Predicate

from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.graph_training import compute_graph_distances


def _make_test_atom(name: str) -> GroundAtom:
    """Create a test ground atom with a given name."""
    pred = Predicate(name, [])  # Empty list for 0-arity predicate
    return GroundAtom(pred, [])


def test_compute_graph_distances_simple():
    """Test graph distance computation on a simple graph."""
    # Create a simple linear graph: 0 -> 1 -> 2
    graph = PlanningGraph()

    node0 = graph.add_node({_make_test_atom("at0")})
    node1 = graph.add_node({_make_test_atom("at1")})
    node2 = graph.add_node({_make_test_atom("at2")})

    edge01 = graph.add_edge(node0, node1, None, cost=10.0)
    edge12 = graph.add_edge(node1, node2, None, cost=5.0)

    # Compute distances
    distances = compute_graph_distances(graph)

    # Check distances
    assert distances[(0, 0)] == 0.0  # Distance to self
    assert distances[(0, 1)] == 10.0  # Direct edge
    assert distances[(0, 2)] == 15.0  # Via node 1
    assert distances[(1, 1)] == 0.0
    assert distances[(1, 2)] == 5.0
    assert distances[(2, 2)] == 0.0


def test_compute_graph_distances_with_branching():
    """Test graph distance computation with multiple paths."""
    # Create a graph with multiple paths:
    #     2 (cost 12)
    #    / \
    #   0   4
    #    \ /
    #     3 (cost 8)
    graph = PlanningGraph()

    node0 = graph.add_node({_make_test_atom("at0")})
    node2 = graph.add_node({_make_test_atom("at2")})
    node3 = graph.add_node({_make_test_atom("at3")})
    node4 = graph.add_node({_make_test_atom("at4")})

    # Two paths from 0 to 4
    edge02 = graph.add_edge(node0, node2, None, cost=12.0)
    edge24 = graph.add_edge(node2, node4, None, cost=3.0)
    edge03 = graph.add_edge(node0, node3, None, cost=8.0)
    edge34 = graph.add_edge(node3, node4, None, cost=5.0)

    # Compute distances
    distances = compute_graph_distances(graph)

    # Check that shortest path is chosen (0 -> 3 -> 4 = 13, better than 0 -> 2 -> 4 = 15)
    # Use actual node IDs from the graph
    assert distances[(node0.id, node4.id)] == 13.0
    assert distances[(node0.id, node2.id)] == 12.0
    assert distances[(node0.id, node3.id)] == 8.0


def test_compute_graph_distances_excludes_shortcuts():
    """Test that shortcuts are excluded from distance computation."""
    graph = PlanningGraph()

    node0 = graph.add_node({_make_test_atom("at0")})
    node1 = graph.add_node({_make_test_atom("at1")})
    node2 = graph.add_node({_make_test_atom("at2")})

    # Regular path: 0 -> 1 -> 2
    edge01 = graph.add_edge(node0, node1, None, cost=10.0, is_shortcut=False)
    edge12 = graph.add_edge(node1, node2, None, cost=10.0, is_shortcut=False)

    # Shortcut: 0 -> 2 (should be excluded)
    edge02_shortcut = graph.add_edge(node0, node2, None, cost=5.0, is_shortcut=True)

    # Compute distances with shortcuts excluded
    distances = compute_graph_distances(graph, exclude_shortcuts=True)

    # Should use the longer path (0 -> 1 -> 2 = 20) instead of shortcut (5)
    assert distances[(0, 2)] == 20.0

    # Now compute with shortcuts included
    distances_with_shortcuts = compute_graph_distances(graph, exclude_shortcuts=False)

    # Should use the shortcut
    assert distances_with_shortcuts[(0, 2)] == 5.0


def test_compute_graph_distances_with_path_dependent_costs():
    """Test that path-dependent costs are used correctly."""
    graph = PlanningGraph()

    node0 = graph.add_node({_make_test_atom("at0")})
    node1 = graph.add_node({_make_test_atom("at1")})
    node2 = graph.add_node({_make_test_atom("at2")})

    # Create edge with path-dependent costs
    edge01 = graph.add_edge(node0, node1, None, cost=10.0)
    edge12 = graph.add_edge(node1, node2, None, cost=100.0)  # Default cost

    # Set path-dependent cost (cheaper when coming via path from node 0)
    edge12.costs[(tuple([0]), 1)] = 5.0

    # Compute distances
    distances = compute_graph_distances(graph)

    # Should use path-dependent cost: 0 -> 1 (10) -> 2 (5) = 15
    assert distances[(0, 2)] == 15.0


def test_compute_graph_distances_unreachable_nodes():
    """Test handling of unreachable nodes."""
    graph = PlanningGraph()

    node0 = graph.add_node({_make_test_atom("at0")})
    node1 = graph.add_node({_make_test_atom("at1")})
    node2 = graph.add_node({_make_test_atom("at2")})  # Unreachable

    # Only connect 0 -> 1
    edge01 = graph.add_edge(node0, node1, None, cost=10.0)

    # Compute distances
    distances = compute_graph_distances(graph)

    # Node 2 is unreachable from nodes 0 and 1
    assert (0, 2) not in distances
    assert (1, 2) not in distances

    # But reachable nodes should work
    assert distances[(0, 1)] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
