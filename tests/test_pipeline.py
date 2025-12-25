"""Unit tests for the modular pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tamp_improv.approaches.improvisational.pipeline import (
    get_or_collect_full_data,
    get_or_train_heuristic,
    prune_full_data,
    train_and_evaluate_with_pipeline,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem


@pytest.fixture
def test_config():
    """Minimal config for testing pipeline."""
    return {
        "seed": 42,
        "collect_episodes": 2,  # Very small for fast testing
        "use_random_rollouts": True,
        "num_rollouts_per_node": 10,
        "max_steps_per_rollout": 20,
        "shortcut_success_threshold": 1,
        "pruning_method": "none",  # Start with no pruning
        "max_training_steps_per_shortcut": 10,  # Very small
        "learning_rate": 3.0e-4,
        "rl_batch_size": 32,
        "n_epochs": 1,  # Very small
        "gamma": 0.99,
        "ent_coef": 0.01,
        "num_episodes": 2,  # Very small for evaluation
        "max_steps": 50,
        "episodes_per_scenario": 10,
        "action_scale": 1.0,
        "deterministic": False,
    }


@pytest.fixture
def test_system():
    """Create a minimal test system."""
    return GraphObstacle2DTAMPSystem.create_default(seed=42, render_mode=None)


@pytest.fixture
def test_policy_factory():
    """Create a policy factory for testing."""
    def factory(seed):
        rl_config = RLConfig(
            learning_rate=3.0e-4,
            batch_size=32,
            n_epochs=1,
            gamma=0.99,
            ent_coef=0.01,
            deterministic=False,
            device="cpu",
        )
        return MultiRLPolicy(seed=seed, config=rl_config)
    return factory


def test_pipeline_imports():
    """Test that all pipeline functions can be imported."""
    assert get_or_collect_full_data is not None
    assert get_or_train_heuristic is not None
    assert prune_full_data is not None
    assert train_and_evaluate_with_pipeline is not None


def test_collection_stage(test_system, test_config):
    """Test Stage 1: Collection."""
    from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
    from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
    from tamp_improv.approaches.improvisational.policies.rl import RLConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        rng = np.random.default_rng(42)

        # Create minimal policy and approach
        rl_config = RLConfig(device="cpu")
        policy = MultiRLPolicy(seed=42, config=rl_config)
        approach = ImprovisationalTAMPApproach(test_system, policy, seed=42)

        # Test collection
        full_data = get_or_collect_full_data(
            test_system,
            approach,
            test_config,
            save_dir,
            rng,
            force_collect=False,
        )

        # Check that we got some data
        assert full_data is not None
        assert len(full_data.valid_shortcuts) > 0

        # Check that cache was created
        assert (save_dir / "full_training_data" / "data.pkl").exists()
        assert (save_dir / "full_training_data" / "config.json").exists()

        # Test cache loading
        full_data2 = get_or_collect_full_data(
            test_system,
            approach,
            test_config,
            save_dir,
            rng,
            force_collect=False,
        )

        # Should load from cache (same number of shortcuts)
        assert len(full_data2.valid_shortcuts) == len(full_data.valid_shortcuts)


def test_pipeline_with_config_dict(test_system, test_policy_factory, test_config):
    """Test that pipeline works with config dict (not TrainingConfig object)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # This should not raise an error
        try:
            metrics = train_and_evaluate_with_pipeline(
                system=test_system,
                policy_factory=test_policy_factory,
                config=test_config,  # Pass dict, not TrainingConfig
                save_dir=save_dir,
                policy_name="MultiRL_Test",
                num_eval_episodes=2,
            )

            # Check that we got valid metrics
            assert 0.0 <= metrics.success_rate <= 1.0
            assert metrics.avg_episode_length > 0
            assert metrics.training_time > 0

        except TypeError as e:
            pytest.fail(f"Pipeline failed with config dict: {e}")


def test_pipeline_with_rollouts_pruning(test_system, test_policy_factory, test_config):
    """Test pipeline with rollouts pruning."""
    test_config["pruning_method"] = "rollouts"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        try:
            metrics = train_and_evaluate_with_pipeline(
                system=test_system,
                policy_factory=test_policy_factory,
                config=test_config,
                save_dir=save_dir,
                policy_name="MultiRL_Rollouts",
                num_eval_episodes=2,
            )

            assert 0.0 <= metrics.success_rate <= 1.0

        except Exception as e:
            pytest.fail(f"Pipeline with rollouts pruning failed: {e}")


@pytest.mark.slow
def test_pipeline_with_heuristic_pruning(test_system, test_policy_factory, test_config):
    """Test pipeline with distance heuristic pruning (slow)."""
    test_config["pruning_method"] = "distance_heuristic"
    test_config["heuristic_training_pairs"] = 5  # Very small
    test_config["heuristic_training_steps"] = 100  # Very small
    test_config["heuristic_learning_rate"] = 3.0e-4
    test_config["heuristic_batch_size"] = 32
    test_config["heuristic_buffer_size"] = 1000
    test_config["heuristic_max_steps"] = 50
    test_config["heuristic_practical_horizon"] = 50

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        try:
            metrics = train_and_evaluate_with_pipeline(
                system=test_system,
                policy_factory=test_policy_factory,
                config=test_config,
                save_dir=save_dir,
                policy_name="MultiRL_Heuristic",
                num_eval_episodes=2,
            )

            assert 0.0 <= metrics.success_rate <= 1.0

        except Exception as e:
            pytest.fail(f"Pipeline with heuristic pruning failed: {e}")


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running pipeline smoke test...")

    config = {
        "seed": 42,
        "collect_episodes": 1,
        "use_random_rollouts": True,
        "num_rollouts_per_node": 5,
        "max_steps_per_rollout": 10,
        "shortcut_success_threshold": 1,
        "max_shortcuts_per_graph": 5,
        "pruning_method": "distance_heuristic",
        "max_training_steps_per_shortcut": 10,
        "learning_rate": 3.0e-4,
        "rl_batch_size": 32,
        "n_epochs": 1,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "num_episodes": 1,
        "max_steps": 50,
        "episodes_per_scenario": 5,
        "action_scale": 1.0,
        "deterministic": False,
        "debug": True
    }

    system = GraphObstacle2DTAMPSystem.create_default(seed=42, render_mode=None)

    def policy_factory(seed):
        rl_config = RLConfig(device="cpu", learning_rate=3.0e-4, batch_size=32, n_epochs=1)
        return MultiRLPolicy(seed=seed, config=rl_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        print("Testing pipeline with config dict...")
        metrics = train_and_evaluate_with_pipeline(
            system=system,
            policy_factory=policy_factory,
            config=config,
            save_dir=save_dir,
            policy_name="Test",
            num_eval_episodes=1,
        )

        print(f"✓ Pipeline completed!")
        print(f"  Success rate: {metrics.success_rate:.2%}")
        print(f"  Training time: {metrics.training_time:.1f}s")
