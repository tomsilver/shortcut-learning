"""Unit tests for heuristic interface and implementations.

This test module verifies that all heuristic implementations satisfy the BaseHeuristic
interface and produce valid outputs. Tests can be run on any heuristic type.
"""

import numpy as np
import pytest

from tamp_improv.approaches.improvisational.collection import collect_total_shortcuts
from tamp_improv.approaches.improvisational.graph_training import (
    compute_graph_distances,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_none import (
    NoneHeuristic,
)
from tamp_improv.approaches.improvisational.heuristics.heuristic_crl_v2 import (
    CRLv2Heuristic,
    CRLV2HeuristicConfig,
)
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedTAMPSystem


@pytest.fixture
def gridworld_system():
    """Create a simple gridworld system for testing."""
    return GridworldFixedTAMPSystem.create_default(seed=42)


@pytest.fixture
def training_data(gridworld_system):
    """Collect training data from gridworld."""
    training_data, _, _ = collect_total_shortcuts(
        system=gridworld_system,
        num_training_tasks=2,  # Small for testing
        max_episode_steps=50,
        seed=42,
    )
    return training_data


@pytest.fixture
def graph_distances(training_data):
    """Compute graph distances."""
    return compute_graph_distances(training_data.graph)


class TestHeuristicInterface:
    """Test that heuristics satisfy the BaseHeuristic interface."""

    def test_none_heuristic_interface(self, training_data, graph_distances):
        """Test NoneHeuristic satisfies interface."""
        rng = np.random.default_rng(42)
        heuristic = NoneHeuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            rng=rng,
        )

        # Test multi_train returns dict
        result = heuristic.multi_train()
        assert isinstance(result, dict), "multi_train should return dict"

        # Test estimate_distance returns float
        if len(training_data.states) > 0 and len(training_data.unique_shortcuts) > 0:
            state = training_data.states[0]
            target_node = training_data.valid_shortcuts[0][1]
            distance = heuristic.estimate_distance(state, target_node)
            assert isinstance(distance, (int, float)), "estimate_distance should return numeric"
            assert distance >= 0, "distance should be non-negative"

        # Test estimate_node_distance returns float
        if len(training_data.unique_shortcuts) > 0:
            source_node, target_node = training_data.unique_shortcuts[0]
            distance = heuristic.estimate_node_distance(source_node, target_node)
            assert isinstance(distance, (int, float)), "estimate_node_distance should return numeric"
            assert distance >= 0, "node distance should be non-negative"

        # Test prune returns GoalConditionedTrainingData
        pruned = heuristic.prune(max_shortcuts=5)
        assert hasattr(pruned, "states"), "pruned data should have states"
        assert hasattr(pruned, "valid_shortcuts"), "pruned data should have valid_shortcuts"
        assert hasattr(pruned, "unique_shortcuts"), "pruned data should have unique_shortcuts"
        assert len(pruned.unique_shortcuts) <= 5, "should respect max_shortcuts limit"

    def test_crlv2_heuristic_interface(self, training_data, graph_distances, gridworld_system):
        """Test CRLv2Heuristic satisfies interface."""
        rng = np.random.default_rng(42)

        # Create minimal config for fast testing
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            hidden_dims=[16],
            actor_hidden_dims=[16],
            batch_size=8,
            buffer_size=100,
            num_rounds=1,
            num_epochs_per_round=2,
            trajectories_per_epoch=2,
            max_episode_steps=10,
            learn_frequency=1,
            iters_per_epoch=1,
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Test multi_train returns dict
        result = heuristic.multi_train()
        assert isinstance(result, dict), "multi_train should return dict"
        assert "critic_losses" in result or "rounds" in result, "should have training metrics"

        # Test estimate_distance returns float
        if len(training_data.states) > 0 and len(training_data.unique_shortcuts) > 0:
            state = training_data.states[0]
            target_node = training_data.valid_shortcuts[0][1]
            distance = heuristic.estimate_distance(state, target_node)
            assert isinstance(distance, (int, float)), "estimate_distance should return numeric"
            assert distance >= 0, "distance should be non-negative"
            assert not np.isnan(distance), "distance should not be NaN"
            assert not np.isinf(distance), "distance should not be infinite"

        # Test estimate_node_distance returns float
        if len(training_data.unique_shortcuts) > 0:
            source_node, target_node = training_data.unique_shortcuts[0]
            distance = heuristic.estimate_node_distance(source_node, target_node)
            assert isinstance(distance, (int, float)), "estimate_node_distance should return numeric"
            assert distance >= 0, "node distance should be non-negative"
            assert not np.isnan(distance), "node distance should not be NaN"

        # Test prune returns GoalConditionedTrainingData
        pruned = heuristic.prune(max_shortcuts=5)
        assert hasattr(pruned, "states"), "pruned data should have states"
        assert hasattr(pruned, "valid_shortcuts"), "pruned data should have valid_shortcuts"
        assert hasattr(pruned, "unique_shortcuts"), "pruned data should have unique_shortcuts"
        assert len(pruned.unique_shortcuts) <= 5, "should respect max_shortcuts limit"


class TestCRLv2Specifics:
    """Test CRLv2-specific functionality."""

    def test_network_initialization(self, training_data, graph_distances, gridworld_system):
        """Test that networks are initialized correctly."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=16,
            hidden_dims=[32, 32],
            actor_hidden_dims=[32, 32],
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Check network dimensions
        assert heuristic.sa_encoder.latent_dim == 16
        assert heuristic.g_encoder.shape == (heuristic.num_nodes, 16)

        # Check actor type based on action space
        if heuristic.action_space_type == "discrete":
            assert hasattr(heuristic.actor, "num_actions")
        else:
            assert hasattr(heuristic.actor, "action_dim")

    def test_discrete_action_space(self, training_data, graph_distances, gridworld_system):
        """Test that discrete action space is handled correctly."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            hidden_dims=[16],
            actor_hidden_dims=[16],
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Gridworld has discrete actions
        assert heuristic.action_space_type == "discrete"
        assert heuristic.action_dim > 0

    def test_trajectory_collection(self, training_data, graph_distances, gridworld_system):
        """Test that trajectory collection works."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            hidden_dims=[16],
            actor_hidden_dims=[16],
            max_episode_steps=10,
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Collect a trajectory
        trajectory = heuristic._collect_trajectory()

        # Verify trajectory structure
        assert isinstance(trajectory, list), "trajectory should be a list"
        if len(trajectory) > 0:
            state, action, node_id = trajectory[0]
            assert isinstance(state, np.ndarray), "state should be ndarray"
            assert isinstance(node_id, (int, np.integer)), "node_id should be int"
            # Action can be int or ndarray depending on action space

    def test_replay_buffer(self, training_data, graph_distances, gridworld_system):
        """Test replay buffer functionality."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            buffer_size=10,
            gamma=0.9,
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Add some trajectories
        for _ in range(5):
            traj = heuristic._collect_trajectory()
            if len(traj) > 0:
                heuristic.replay_buffer.add_trajectory(traj)

        # Check buffer size
        assert len(heuristic.replay_buffer) > 0, "buffer should contain trajectories"
        assert len(heuristic.replay_buffer) <= 10, "buffer should respect max_size"

        # Test sampling
        if len(heuristic.replay_buffer) > 0:
            states, actions, future_nodes = heuristic.replay_buffer.sample_batch_crtr(
                batch_size=2, repetition_factor=2
            )
            assert states.shape[0] == 4, "should have batch_size * repetition_factor samples"
            assert actions.shape[0] == 4
            assert future_nodes.shape[0] == 4

    def test_training_step(self, training_data, graph_distances, gridworld_system):
        """Test that a single training step runs without errors."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            hidden_dims=[16],
            actor_hidden_dims=[16],
            batch_size=4,
            buffer_size=100,
            num_rounds=1,
            num_epochs_per_round=2,
            trajectories_per_epoch=3,
            max_episode_steps=10,
            learn_frequency=1,
            iters_per_epoch=1,
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Fill buffer
        for _ in range(5):
            traj = heuristic._collect_trajectory()
            if len(traj) > 0:
                heuristic.replay_buffer.add_trajectory(traj)

        # Perform one update step
        if len(heuristic.replay_buffer) >= config.batch_size:
            critic_loss, actor_loss = heuristic._update_networks()
            assert isinstance(critic_loss, float), "critic_loss should be float"
            assert isinstance(actor_loss, float), "actor_loss should be float"
            assert not np.isnan(critic_loss), "critic_loss should not be NaN"
            assert not np.isnan(actor_loss), "actor_loss should not be NaN"

    def test_distance_estimation_consistency(self, training_data, graph_distances, gridworld_system):
        """Test that distance estimates are consistent."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            hidden_dims=[16],
            actor_hidden_dims=[16],
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        if len(training_data.states) > 0 and len(training_data.unique_shortcuts) > 0:
            state = training_data.states[0]
            target_node = training_data.valid_shortcuts[0][1]

            # Same state and target should give same distance
            dist1 = heuristic.estimate_distance(state, target_node)
            dist2 = heuristic.estimate_distance(state, target_node)

            # May not be exactly equal due to stochastic policy, but should be close
            assert abs(dist1 - dist2) < 100, "distances should be reasonably consistent"

    def test_pruning_reduces_shortcuts(self, training_data, graph_distances, gridworld_system):
        """Test that pruning reduces number of shortcuts."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            hidden_dims=[16],
            actor_hidden_dims=[16],
            threshold=100.0,  # High threshold to keep most shortcuts
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        initial_shortcuts = len(training_data.unique_shortcuts)
        max_shortcuts = max(1, initial_shortcuts // 2)

        pruned = heuristic.prune(max_shortcuts=max_shortcuts)

        assert len(pruned.unique_shortcuts) <= max_shortcuts
        assert len(pruned.unique_shortcuts) <= initial_shortcuts


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trajectory(self, training_data, graph_distances, gridworld_system):
        """Test handling of empty trajectories."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            buffer_size=10,
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Add empty trajectory
        heuristic.replay_buffer.add_trajectory([])

        # Buffer should not store empty trajectories
        assert len(heuristic.replay_buffer) == 0

    def test_single_step_trajectory(self, training_data, graph_distances, gridworld_system):
        """Test handling of single-step trajectories."""
        rng = np.random.default_rng(42)
        config = CRLV2HeuristicConfig(
            latent_dim=8,
            buffer_size=10,
            gamma=0.9,
            device="cpu",
        )

        heuristic = CRLv2Heuristic(
            training_data=training_data,
            graph_distances=graph_distances,
            system=gridworld_system,
            rng=rng,
            config=config,
        )

        # Create single-step trajectory
        if len(training_data.states) > 0:
            state = training_data.states[0]
            action = gridworld_system.env.action_space.sample()
            node_id = list(training_data.node_states.keys())[0]

            single_step_traj = [(state, action, node_id)]
            heuristic.replay_buffer.add_trajectory(single_step_traj)

            # Should be able to sample from buffer
            if len(heuristic.replay_buffer) > 0:
                states, actions, nodes = heuristic.replay_buffer.sample_batch_crtr(
                    batch_size=1, repetition_factor=1
                )
                assert len(states) == 1
                assert len(actions) == 1
                assert len(nodes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
