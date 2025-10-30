"""Quick test for obstacle_tower collection compatibility.

This test verifies that obstacle_tower works with the collection pipeline
without running the full expensive test.
"""

import pytest

from shortcut_learning.configs import ApproachConfig, PolicyConfig
from shortcut_learning.methods.policies.multi_rl_ppo_v2 import MultiRLPolicyV2
from shortcut_learning.methods.slap_approach_v2 import SLAPApproachV2
from shortcut_learning.problems.obstacle_tower.system import (
    BaseObstacleTowerTAMPSystem,
)


@pytest.mark.slow
def test_obstacle_tower_planning_graph():
    """Test that obstacle_tower can build a planning graph."""
    system = BaseObstacleTowerTAMPSystem.create_default(
        num_obstacle_blocks=3,
        stack_blocks=True,
        seed=42
    )
    approach_config = ApproachConfig(
        approach_type="slap",
        approach_name="test_obstacle_tower",
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

    assert approach.planning_graph is not None
    num_nodes = len(approach.planning_graph.nodes)
    print(f"\nPlanning graph has {num_nodes} nodes")

    # Check that all nodes have the states field
    for node in approach.planning_graph.nodes:
        assert hasattr(node, "states"), f"Node {node.id} missing states field"
        assert isinstance(node.states, list), f"Node {node.id} states is not a list"
        assert node.states == [], f"Node {node.id} should start with empty states"

    # Verify graph is non-trivial
    assert num_nodes > 10, f"Expected >10 nodes for obstacle_tower, got {num_nodes}"


if __name__ == "__main__":
    print("Testing obstacle_tower planning graph...")
    test_obstacle_tower_planning_graph()
    print("\n✓ Test passed!")
