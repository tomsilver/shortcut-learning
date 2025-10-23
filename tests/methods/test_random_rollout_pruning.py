"""Test random rollout pruning for V2 shortcut selection."""

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    PolicyConfig,
)
from shortcut_learning.methods.pipeline import (
    initialize_approach,
    initialize_policy,
)
from shortcut_learning.methods.collection_v2 import collect_training_data_v2
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem


def test_random_rollout_pruning():
    """Test that random rollout pruning reduces shortcuts to promising ones."""
    print("\n" + "="*60)
    print("TEST: Random Rollout Pruning")
    print("="*60)

    # Create system
    system = BaseObstacle2DTAMPSystem.create_default(seed=42)

    # Create approach
    approach_config = ApproachConfig(
        approach_type='slap_v2',
        approach_name='test_pruning',
        debug_videos=False,
        seed=42
    )
    policy_config = PolicyConfig(policy_type='rl_ppo')
    approach = initialize_approach(system, approach_config, policy_config)

    # Build planning graph
    obs, info = approach.system.reset()
    approach.build_planning_graph(obs, info)

    print(f"\nPlanning graph: {len(approach.planning_graph.nodes)} nodes")

    # First: Collect WITHOUT random rollout pruning
    print("\n" + "="*60)
    print("PART 1: Collection WITHOUT random rollout pruning")
    print("="*60)

    collect_config_no_pruning = CollectionConfig(
        seed=42,
        states_per_node=10,
        perturbation_steps=5,
        use_random_rollouts=False,  # No pruning
        max_shortcuts_per_graph=100,
    )

    train_data_no_pruning = collect_training_data_v2(approach, collect_config_no_pruning)
    num_shortcuts_no_pruning = len(train_data_no_pruning)

    print(f"\n📊 WITHOUT pruning: {num_shortcuts_no_pruning} shortcuts selected")

    # Second: Collect WITH random rollout pruning
    print("\n" + "="*60)
    print("PART 2: Collection WITH random rollout pruning")
    print("="*60)

    # Reset states in nodes (they were populated by first collection)
    for node in approach.planning_graph.nodes:
        node.states = []

    collect_config_with_pruning = CollectionConfig(
        seed=42,
        states_per_node=10,
        perturbation_steps=5,
        use_random_rollouts=True,  # Use pruning
        num_rollouts_per_node=50,
        max_steps_per_rollout=30,
        shortcut_success_threshold=1,
        action_scale=1.0,
        max_shortcuts_per_graph=100,
    )

    train_data_with_pruning = collect_training_data_v2(approach, collect_config_with_pruning)
    num_shortcuts_with_pruning = len(train_data_with_pruning)

    print(f"\n📊 WITH pruning: {num_shortcuts_with_pruning} shortcuts selected")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    print(f"\nShortcuts without pruning: {num_shortcuts_no_pruning}")
    print(f"Shortcuts with pruning:    {num_shortcuts_with_pruning}")

    if num_shortcuts_with_pruning < num_shortcuts_no_pruning:
        reduction_pct = 100 * (1 - num_shortcuts_with_pruning / num_shortcuts_no_pruning)
        print(f"Reduction: {reduction_pct:.1f}%")
        print("\n✅ SUCCESS: Pruning reduced the number of shortcuts!")
        print("   This means we're filtering out unpromising shortcuts.")
    else:
        print(f"\n⚠️  UNEXPECTED: Pruning did not reduce shortcuts")
        print("   All shortcuts might be reachable via random exploration,")
        print("   or the threshold might be too low.")

    # Show which shortcuts were selected with pruning
    print("\n📋 Shortcuts selected WITH pruning:")
    for i, (source, target) in enumerate(train_data_with_pruning.shortcuts):
        num_examples = len(source.states)
        print(f"  Shortcut {i}: node {source.id} → {target.id} ({num_examples} training states)")

    # Verify basic properties
    assert num_shortcuts_with_pruning > 0, "Should find at least some promising shortcuts"
    assert num_shortcuts_with_pruning <= num_shortcuts_no_pruning, \
        "Pruning should not increase number of shortcuts"

    print("\n" + "="*60)
    print("TEST PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    test_random_rollout_pruning()
