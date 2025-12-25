"""Tests for distance heuristic."""

import numpy as np
import pytest

from tamp_improv.approaches.improvisational.distance_heuristic import (
    DistanceHeuristicConfig,
    DistanceHeuristicWrapper,
    GoalConditionedDistanceHeuristic,
)


class MockPerceiver:
    """Mock perceiver for testing."""

    def __init__(self):
        # Build a simple vocabulary
        self._atom_vocabulary = ["at(robot,table)", "at(robot,target)", "holding(robot,block)"]
        self._atom_to_idx = {atom: idx for idx, atom in enumerate(self._atom_vocabulary)}

    def step(self, obs):
        """Convert observation to atoms."""
        # For testing, just return a set based on observation values
        atoms = set()
        if obs[0] > 0:
            atoms.add("at(robot,table)")
        if obs[1] > 0:
            atoms.add("at(robot,target)")
        if obs[2] > 0:
            atoms.add("holding(robot,block)")
        return atoms

    def encode_atoms_to_vector(self, atoms):
        """Encode atoms to binary vector."""
        vector = np.zeros(len(self._atom_vocabulary), dtype=np.float32)
        for atom in atoms:
            if atom in self._atom_to_idx:
                vector[self._atom_to_idx[atom]] = 1.0
        return vector


def test_distance_heuristic_wrapper_creation():
    """Test that wrapper can be created with state pairs."""
    # Create mock environment
    import gymnasium as gym

    class MockEnv(gym.Env):
        """Mock environment for testing."""

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(10,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            )

        def reset_from_state(self, state, seed=None):
            return state, {}

        def step(self, action):
            obs = self.observation_space.sample()
            return obs, 0.0, False, False, {}

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

    env = MockEnv()
    perceiver = MockPerceiver()
    state_pairs = [
        (np.random.randn(10).astype(np.float32), np.random.randn(10).astype(np.float32))
        for _ in range(5)
    ]

    wrapper = DistanceHeuristicWrapper(env, state_pairs, perceiver, max_episode_steps=50)

    # Check observation space
    assert isinstance(wrapper.observation_space, gym.spaces.Dict)
    assert "observation" in wrapper.observation_space.spaces
    assert "achieved_goal" in wrapper.observation_space.spaces
    assert "desired_goal" in wrapper.observation_space.spaces

    # Goal spaces should be atom vectors (default max_atom_size=50)
    assert wrapper.observation_space.spaces["desired_goal"].shape == (50,)
    assert wrapper.observation_space.spaces["achieved_goal"].shape == (50,)


def test_distance_heuristic_wrapper_reset():
    """Test that wrapper can reset and sample state pairs."""
    import gymnasium as gym

    class MockEnv(gym.Env):
        """Mock environment for testing."""

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(10,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            )

        def reset_from_state(self, state, seed=None):
            return state, {}

        def step(self, action):
            obs = self.observation_space.sample()
            return obs, 0.0, False, False, {}

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

    env = MockEnv()
    perceiver = MockPerceiver()
    state_pairs = [
        (np.random.randn(10).astype(np.float32), np.random.randn(10).astype(np.float32))
        for _ in range(5)
    ]

    wrapper = DistanceHeuristicWrapper(env, state_pairs, perceiver, max_episode_steps=50)
    obs, info = wrapper.reset()

    # Check observation structure
    assert isinstance(obs, dict)
    assert "observation" in obs
    assert "achieved_goal" in obs
    assert "desired_goal" in obs
    assert obs["observation"].shape == (10,)

    # Goals should be atom vectors (default max_atom_size=50)
    assert obs["desired_goal"].shape == (50,)
    assert obs["achieved_goal"].shape == (50,)
    # Atom vectors should be binary
    assert np.all((obs["desired_goal"] == 0) | (obs["desired_goal"] == 1))
    assert np.all((obs["achieved_goal"] == 0) | (obs["achieved_goal"] == 1))


def test_distance_heuristic_wrapper_step():
    """Test that wrapper returns -1 reward per step."""
    import gymnasium as gym

    class MockEnv(gym.Env):
        """Mock environment for testing."""

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(10,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            )
            self.state = None

        def reset_from_state(self, state, seed=None):
            self.state = state
            return state, {}

        def step(self, action):
            obs = self.observation_space.sample()
            return obs, 0.0, False, False, {}

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

    env = MockEnv()
    perceiver = MockPerceiver()
    state_pairs = [
        (np.random.randn(10).astype(np.float32), np.random.randn(10).astype(np.float32))
        for _ in range(5)
    ]

    wrapper = DistanceHeuristicWrapper(env, state_pairs, perceiver, max_episode_steps=50)
    obs, info = wrapper.reset()
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = wrapper.step(action)

    # Key test: reward should always be -1
    assert reward == -1.0
    assert isinstance(obs, dict)
    assert "is_success" in info

    # Goals should be atom vectors (default max_atom_size=50)
    assert obs["desired_goal"].shape == (50,)
    assert obs["achieved_goal"].shape == (50,)


def test_distance_heuristic_initialization():
    """Test that distance heuristic can be initialized."""
    config = DistanceHeuristicConfig(
        learning_rate=3e-4,
        batch_size=128,
        device="cpu",
    )

    heuristic = GoalConditionedDistanceHeuristic(config=config, seed=42)

    assert heuristic.config.learning_rate == 3e-4
    assert heuristic.config.batch_size == 128
    assert heuristic.device == "cpu"
    assert heuristic.model is None  # Not trained yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
