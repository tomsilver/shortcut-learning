"""Goal-conditioned distance heuristic V3 using contrastive learning.

This module implements a learned distance function f(s, s') that estimates
the number of steps required to reach target state s' from source state s.

V3 uses contrastive learning to learn an embedding space where L2 distance
correlates with trajectory distance. This avoids the sparse reward problem
that plagued the DQN-based approaches (V1 and V2).

The approach is based on the contrastive learning method from the
interp-planning-v2 project, which has been validated on Manhattan gridworlds.

Key components:
- StateEncoder: Neural network that maps states to k-dimensional embeddings
- Contrastive loss: Aligns (current, future) pairs and separates negative pairs
- Distance estimation: L2 distance in embedding space ≈ trajectory distance
"""
import time
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class DistanceHeuristicV3Config:
    """Configuration for contrastive learning distance heuristic."""

    # Network architecture
    latent_dim: int = 32  # Dimension of embedding space (k)
    hidden_dims: list[int] | None = None  # Hidden layer sizes [64, 64]

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    buffer_size: int = 10000
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Contrastive learning
    c_target: float = 1.0  # Target L2 norm constraint
    gamma: float = 0.99  # For geometric sampling of future states
    repetition_factor: int = 4  # CRTR: repeat each trajectory this many times in batch
    policy_temperature: float = 1.0
    
    # Training
    iters_per_epoch: int = 1  # Gradient steps per training call
    learn_frequency: int = 10  # Learn every N epochs

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


class StateEncoder(nn.Module):
    """Neural network that encodes states to k-dimensional embeddings.

    Maps state (state_dim-dimensional vector) -> embedding (k-dimensional vector)
    """

    def __init__(self, state_dim: int, latent_dim: int, hidden_dims: list[int] | None = None):
        """Initialize the state encoder network.

        Args:
            state_dim: Dimension of input state
            latent_dim: Dimension of output embedding (k)
            hidden_dims: List of hidden layer sizes
        """
        super(StateEncoder, self).__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Default hidden dims
        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Encode states to embeddings.

        Args:
            states: Batch of states (batch_size, state_dim) - float tensor

        Returns:
            Embeddings (batch_size, latent_dim)
        """
        return self.network(states)


class ContrastiveReplayBuffer:
    """Replay buffer for storing complete trajectories.

    Implements CRTR (Contrastive Representations for Temporal Reasoning) sampling
    to encourage learning temporal dynamics rather than static context.
    """

    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        repetition_factor: int = 4,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of trajectories to store
            gamma: Discount factor (used for geometric sampling of future states)
            repetition_factor: Number of times to repeat each trajectory in a batch
                              (for CRTR within-trajectory negatives)
        """
        self.capacity = capacity
        self.gamma = gamma
        self.repetition_factor = repetition_factor

        # Storage for complete trajectories
        # Each trajectory is a list of states (NDArray)
        self.trajectories: deque = deque(maxlen=capacity)

        # Store metadata for each trajectory (start state, goal state)
        self.trajectory_metadata: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return total number of trajectories stored."""
        return len(self.trajectories)

    def store_trajectory(
        self,
        trajectory: list[NDArray],
        start_state: NDArray | None = None,
        goal_state: NDArray | None = None
    ) -> None:
        """Store complete trajectory with metadata.

        Args:
            trajectory: List of states visited during episode
            start_state: Starting state of the trajectory
            goal_state: Goal state for the trajectory
        """
        if len(trajectory) < 2:
            return

        # Store the entire trajectory
        self.trajectories.append(trajectory)

        # Store metadata
        self.trajectory_metadata.append({
            'start_state': start_state,
            'goal_state': goal_state,
        })

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample batch using CRTR strategy.

        Implements Algorithm 1 from "Contrastive Representations for Temporal Reasoning":
        - Sample batch_size/repetition_factor unique trajectories
        - Repeat each trajectory ID repetition_factor times
        - Sample different (t0, t1) time points for each repetition

        This creates within-trajectory negatives that force the model to learn
        temporal dynamics rather than static context.

        Args:
            batch_size: Number of state pairs to sample

        Returns:
            Dictionary with keys:
                - current_states: (batch_size, state_dim)
                - future_states: (batch_size, state_dim)
        """
        if len(self.trajectories) == 0:
            raise ValueError("Buffer is empty")

        # Ensure batch_size is divisible by repetition_factor
        assert batch_size % self.repetition_factor == 0, \
            f"batch_size ({batch_size}) must be divisible by repetition_factor ({self.repetition_factor})"

        # Sample unique trajectory IDs
        num_unique_trajs = batch_size // self.repetition_factor
        unique_traj_ids = np.random.choice(
            len(self.trajectories),
            size=num_unique_trajs,
            replace=False if num_unique_trajs <= len(self.trajectories) else True
        )

        # Repeat each trajectory ID repetition_factor times
        # This is the key CRTR modification!
        traj_ids = np.repeat(unique_traj_ids, self.repetition_factor)

        current_states = []
        future_states = []

        # For each trajectory in the batch, sample (t0, t1) pair
        for traj_id in traj_ids:
            trajectory = self.trajectories[traj_id]
            traj_len = len(trajectory)

            if traj_len < 2:
                # Edge case: trajectory too short, use first and last state
                current_states.append(trajectory[0])
                future_states.append(trajectory[-1])
                continue

            # Sample t0 uniformly from trajectory
            t0 = np.random.randint(0, traj_len - 1)

            # Sample t1 using geometric distribution
            # p = gamma gives bias toward NEARBY future states (like interp-planning)
            # Higher p → samples closer to t0
            num_future = traj_len - t0 - 1
            p = self.gamma  # Use gamma directly, NOT (1 - gamma)!
            # print("Geometric probability:", p)
            offset = min(
                np.random.geometric(p) - 1,  # -1 because geometric starts at 1
                num_future - 1
            )
            t1 = t0 + 1 + offset

            current_states.append(trajectory[t0])
            future_states.append(trajectory[t1])

        # Stack into tensors
        current_states = torch.FloatTensor(np.array(current_states))
        future_states = torch.FloatTensor(np.array(future_states))

        return {
            "current_states": current_states,
            "future_states": future_states,
        }


def contrastive_loss_with_dual(
    psi_net: StateEncoder,
    A: torch.Tensor,
    log_lambda: torch.Tensor,
    current_states: torch.Tensor,
    future_states: torch.Tensor,
    c_target: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute contrastive loss with dual formulation.

    This implements the loss from the interp-planning-v2 project:
    - phi = A @ psi(x0) - transformed current state embeddings
    - psi = psi(xT) - future state embeddings
    - l_align: ||phi - psi||^2 (align positive pairs)
    - l_unif: logsumexp over negative pair distances (uniformity)
    - l2: mean squared norm of embeddings
    - dual_loss: log_lambda * (c - l2) to constrain L2 norm

    Args:
        psi_net: Neural network encoder
        A: Transformation matrix (k x k)
        log_lambda: Scalar log of Lagrange multiplier (for dual formulation)
        current_states: Batch of current states (batch_size, state_dim)
        future_states: Batch of future states (batch_size, state_dim)
        c_target: Target L2 norm constraint

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Get embeddings
    # phi = A @ psi(x0) - transformed current state embeddings
    # psi = psi(xT) - future state embeddings
    current_emb = psi_net(current_states)  # (batch_size, k)
    phi = current_emb @ A.T  # (batch_size, k)

    future_emb = psi_net(future_states)  # (batch_size, k)
    psi = future_emb  # (batch_size, k)

    batch_size = phi.shape[0]

    # L2 regularization: average squared norm
    l2 = (torch.mean(psi**2) + torch.mean(current_emb**2)) / 2

    # Alignment loss: ||phi - psi||^2 for positive pairs
    l_align = torch.sum((phi - psi) ** 2, dim=1)  # (batch_size,)

    # Pairwise distances: ||phi[i] - psi[j]||^2 for all i, j
    # phi[:, None] is (batch_size, 1, k)
    # psi[None] is (1, batch_size, k)
    # Result is (batch_size, batch_size)
    pdist = torch.mean((phi[:, None] - psi[None]) ** 2, dim=-1)

    # Uniformity loss: logsumexp over negative pairs
    # Mask out diagonal (positive pairs) with identity matrix
    I = torch.eye(batch_size, device=phi.device)

    # For each row i: logsumexp over j != i of -pdist[i, j]
    # For each col j: logsumexp over i != j of -pdist[i, j]
    l_unif = (
        torch.logsumexp(-(pdist * (1 - I)), dim=1) +
        torch.logsumexp(-(pdist.T * (1 - I)), dim=1)
    ) / 2.0

    # Combined contrastive loss
    loss = l_align + l_unif

    # Accuracy: how often is the closest future state the correct one?
    # Use unscaled distances for accuracy metric
    accuracy = torch.mean((torch.argmin(pdist, dim=1) == torch.arange(batch_size, device=phi.device)).float())

    # Dual loss to constrain L2 norm
    # dual_loss = log_lambda * (c_target - stop_gradient(l2))
    dual_loss = log_lambda * (c_target - l2.detach())

    # Total loss
    # loss + exp(log_lambda) * l2 + dual_loss
    total_loss = (
        loss.mean() +
        torch.exp(log_lambda).detach() * l2 +
        dual_loss
    )

    # Metrics
    metrics = {
        'loss': loss.mean().item(),
        'l_unif': l_unif.mean().item(),
        'l_align': l_align.mean().item(),
        'accuracy': accuracy.item(),
        'l2': l2.item(),
        'lambda': torch.exp(log_lambda).item(),
        'total_loss': total_loss.item()
    }

    return total_loss, metrics


class DistanceHeuristicV3:
    """Contrastive learning-based distance heuristic.

    Learns an embedding space where L2 distance correlates with trajectory distance.
    """

    def __init__(
        self,
        config: DistanceHeuristicV3Config | None = None,
        seed: int | None = None,
    ):
        """Initialize distance heuristic V3.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config or DistanceHeuristicV3Config()
        self.seed = seed

        if self.config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Networks
        self.psi_net: StateEncoder | None = None
        self.A: torch.Tensor | None = None  # Transformation matrix (k x k)
        self.log_lambda: torch.Tensor | None = None  # Lagrange multiplier
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ContrastiveReplayBuffer | None = None

        self.perceiver: Any | None = None
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

        # Training state
        self.total_episodes = 0
        self.policy_temperature = config.policy_temperature  # Softmax temperature for action selection

    def _select_action_greedy(
        self,
        env: Any,
        current_state: ObsType,
        goal_state: ObsType,
    ) -> int | None:
        """Select action using greedy policy over learned embedding space.

        Uses softmax over distances in embedding space (lower distance → higher probability).

        Args:
            env: Environment
            current_state: Current state
            goal_state: Goal state

        Returns:
            Selected action, or None if no valid actions
        """
        # Get available actions
        available_actions = list(range(env.action_space.n))

        if len(available_actions) == 0:
            return None

        # If networks not trained yet, use random policy
        if self.psi_net is None or self.A is None:
            return random.choice(available_actions)

        # Compute goal embedding
        goal_flat = self._flatten_state(goal_state)
        with torch.no_grad():
            goal_tensor = torch.FloatTensor(goal_flat).unsqueeze(0).to(self.device)
            goal_emb = self.psi_net(goal_tensor).squeeze(0).cpu().numpy()  # (k,)

        # Compute distances in embedding space for each action
        distances = []
        for action in available_actions:
            # Save current state
            env.reset_from_state(current_state)

            # Take action to get next state
            next_state, _, _, _, _ = env.step(action)

            # Encode next state and transform: A @ psi(next_state)
            next_flat = self._flatten_state(next_state)
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_flat).unsqueeze(0).to(self.device)
                next_emb = self.psi_net(next_tensor).squeeze(0).cpu().numpy()  # (k,)
                next_transformed = self.A.detach().cpu().numpy() @ next_emb  # (k,)

            # Compute L2 distance to goal in embedding space
            dist = np.linalg.norm(next_transformed - goal_emb)
            distances.append(dist)

        distances = np.array(distances)

        # Softmax over negative distances (lower distance = higher probability)
        logits = -distances / self.policy_temperature
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample action
        action = np.random.choice(available_actions, p=probabilities)

        return action

    def collect_trajectories(
        self,
        env: Any,
        state_pairs: list[tuple[ObsType, ObsType]],
        num_trajectories: int,
        max_episode_steps: int = 100,
    ) -> tuple[list[list[NDArray]], dict]:
        """Collect trajectories by rolling out from state pairs.

        Uses learned policy (greedy over embedding space) if available,
        otherwise uses random policy.

        Args:
            env: Base environment (must support reset_from_state)
            state_pairs: List of (source, target) state pairs
            num_trajectories: Number of trajectories to collect
            max_episode_steps: Maximum steps per trajectory

        Returns:
            Tuple of (trajectories, stats_dict) where stats_dict contains:
                - success_rate: Fraction of trajectories that reached goal
                - avg_length: Average trajectory length
                - avg_success_length: Average length of successful trajectories
        """
        trajectories = []
        trajectory_metadata = []
        successes = []
        lengths = []

        for _ in range(num_trajectories):
            # Sample random state pair
            source_state, target_state = random.choice(state_pairs)

            # Reset environment to source state
            current_state, _ = env.reset_from_state(source_state)

            # Collect trajectory using learned policy (or random if not trained yet)
            trajectory = []
            trajectory.append(self._flatten_state(current_state))

            # Flatten target state for comparison
            target_flat = self._flatten_state(target_state)

            success = False
            for _ in range(max_episode_steps):

                # Select action using greedy policy over embedding space
                action = self._select_action_greedy(env, current_state, target_state)

                if action is None:
                    break

                # Step environment
                current_state, _, done, truncated, _ = env.step(action)

                # Check if we've reached the goal (within epsilon)
                current_flat = self._flatten_state(current_state)
                if np.linalg.norm(current_flat - target_flat) < 1e-3:
                    success = True
                    break

                # Store state
                trajectory.append(current_flat)

            trajectories.append(trajectory)
            trajectory_metadata.append({
                'start_state': self._flatten_state(source_state),
                'goal_state': self._flatten_state(target_state),
            })
            successes.append(success)
            lengths.append(len(trajectory))

        # Compute statistics
        success_rate = np.mean(successes) if successes else 0.0
        avg_length = np.mean(lengths) if lengths else 0.0
        success_lengths = [l for l, s in zip(lengths, successes) if s]
        avg_success_length = np.mean(success_lengths) if success_lengths else 0.0

        stats = {
            'success_rate': success_rate,
            'avg_length': avg_length,
            'avg_success_length': avg_success_length,
            'num_successes': sum(successes),
            'num_trajectories': num_trajectories
        }

        return trajectories, trajectory_metadata, stats

    def train(
        self,
        env: Any,
        state_pairs: list[tuple[ObsType, ObsType]],
        perceiver: Any,
        num_epochs: int = 1000,
        trajectories_per_epoch: int = 10,
        max_episode_steps: int = 100,
    ) -> None:
        """Train distance heuristic using contrastive learning.

        Args:
            env: Base environment
            state_pairs: List of (source, target) state pairs
            perceiver: Perceiver to convert states to atoms (not used in V3)
            num_epochs: Number of training epochs
            trajectories_per_epoch: Number of trajectories to collect per epoch
            max_episode_steps: Maximum steps per episode
        """
        print(f"\nTraining distance heuristic V3 on {len(state_pairs)} state pairs...")
        print(f"Device: {self.device}")

        # Store perceiver (not used, kept for compatibility)
        self.perceiver = perceiver

        # Determine state dimension
        sample_state = state_pairs[0][0]
        state_flat = self._flatten_state(sample_state)
        state_dim = state_flat.shape[0]

        print(f"State dimension: {state_dim}")
        print(f"Latent dimension: {self.config.latent_dim}")

        # Compute observation statistics for normalization
        self._compute_obs_statistics(state_pairs)

        # Create encoder network
        self.psi_net = StateEncoder(
            state_dim=state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        # Initialize learnable parameters
        self.A = nn.Parameter(torch.eye(self.config.latent_dim, device=self.device, requires_grad=True))
        self.log_lambda = nn.Parameter(torch.tensor(0.0, device=self.device, requires_grad=True))

        # Create optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            list(self.psi_net.parameters()) + [self.A, self.log_lambda],
            lr=self.config.learning_rate
        )

        # Create cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01  # Decay to 1% of initial LR
        )

        # Create replay buffer with CRTR sampling
        self.replay_buffer = ContrastiveReplayBuffer(
            capacity=self.config.buffer_size,
            gamma=self.config.gamma,
            repetition_factor=self.config.repetition_factor,
        )

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Collecting {trajectories_per_epoch} trajectories per epoch")
        print(f"Using cosine annealing LR scheduler: {self.config.learning_rate} -> {self.config.learning_rate * 0.01}")

        # Temperature annealing parameters (like interp-planning)
        initial_temp = self.config.policy_temperature
        min_temp = 0.1 * initial_temp # Same as eval temperature in interp-planning

        # Training loop
        for epoch in range(num_epochs):
            # Update policy temperature using cosine annealing
            progress = epoch / num_epochs
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            self.policy_temperature = min_temp + (initial_temp - min_temp) * cosine_factor

            # Collect trajectories
            trajectories, trajectory_metadata, traj_stats = self.collect_trajectories(
                env,
                state_pairs,
                num_trajectories=trajectories_per_epoch,
                max_episode_steps=max_episode_steps,
            )

            # Store trajectories in replay buffer with metadata
            for trajectory, metadata in zip(trajectories, trajectory_metadata):
                self.replay_buffer.store_trajectory(
                    trajectory,
                    start_state=metadata['start_state'],
                    goal_state=metadata['goal_state']
                )
                self.total_episodes += 1

            # Train if we have enough data (only every learn_frequency epochs)
            if (epoch > 0 and
                epoch % self.config.learn_frequency == 0 and
                len(self.replay_buffer) >= self.config.batch_size):

                metrics = self._train_step()

                # Logging after learning
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n[Epoch {epoch}/{num_epochs}]")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                print(f"  Learning rate: {current_lr:.6f}")
                print(f"  --- Policy Performance ---")
                print(f"  Success rate: {traj_stats['success_rate']:.2%} ({traj_stats['num_successes']}/{traj_stats['num_trajectories']})")
                print(f"  Avg trajectory length: {traj_stats['avg_length']:.1f}")
                if traj_stats['num_successes'] > 0:
                    print(f"  Avg successful length: {traj_stats['avg_success_length']:.1f}")
                print(f"  --- Training Metrics ---")
                print(f"  Total loss: {metrics['total_loss']:.4f}")
                print(f"  Alignment loss: {metrics['l_align']:.4f}")
                print(f"  Uniformity loss: {metrics['l_unif']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  L2 norm: {metrics['l2']:.4f}")
                print(f"  Lambda: {metrics['lambda']:.4f}")

            # Step the learning rate scheduler every epoch
            self.scheduler.step()

        print(f"\nTraining complete!")

    def _train_step(self) -> dict:
        """Perform one training step (batch update).

        Returns:
            Dictionary of training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        # Move to device
        current_states = batch["current_states"].to(self.device)
        future_states = batch["future_states"].to(self.device)

        # Run gradient descent
        for _ in range(self.config.iters_per_epoch):
            self.optimizer.zero_grad()

            # Compute loss
            loss, metrics = contrastive_loss_with_dual(
                self.psi_net,
                self.A,
                self.log_lambda,
                current_states,
                future_states,
                c_target=self.config.c_target,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.psi_net.parameters()) + [self.A, self.log_lambda],
                max_norm=self.config.grad_clip
            )

            # Update parameters
            self.optimizer.step()

        return metrics
    
    def s_encode(self, state: ObsType) -> np.ndarray:
        flat = self._flatten_state(state)
        with torch.no_grad():
            ten = torch.FloatTensor(flat).unsqueeze(0).to(self.device)
            emb = self.A @ self.psi_net(ten).squeeze(0)
        return emb.numpy()
    
    def g_encode(self, state: ObsType) -> np.ndarray:
        flat = self._flatten_state(state)
        with torch.no_grad():
            ten = torch.FloatTensor(flat).unsqueeze(0).to(self.device)
            emb = self.psi_net(ten).squeeze(0)
        return emb.numpy()

    def latent_dist(self, source_state: ObsType, target_state: ObsType) -> float:
        """Estimate distance from source to target state.

        Uses L2 distance in learned embedding space as proxy for trajectory distance.

        Args:
            source_state: Starting state
            target_state: Goal state

        Returns:
            Estimated distance (L2 distance in embedding space)
        """
        assert self.psi_net is not None, "Model must be trained before estimation"
        assert self.A is not None, "Model must be trained before estimation"

        # Flatten states
        source_flat = self._flatten_state(source_state)
        target_flat = self._flatten_state(target_state)

        # Encode states
        with torch.no_grad():
            source_tensor = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            target_tensor = torch.FloatTensor(target_flat).unsqueeze(0).to(self.device)

            # Get embeddings
            source_emb = self.psi_net(source_tensor).squeeze(0)  # (k,)
            target_emb = self.psi_net(target_tensor).squeeze(0)  # (k,)

            # Transform source embedding: A @ psi(source)
            source_transformed = self.A @ source_emb  # (k,)

            # Compute L2 distance in embedding space
            distance = torch.norm(source_transformed - target_emb).item()

        return float(distance)
    
    def estimate_distance(self, source_state: ObsType, target_state: ObsType):
        d_sg = self.latent_dist(source_state, target_state)
        d_gg = self.latent_dist(target_state, target_state)
        factor = (1 / (2 * np.log(self.config.gamma)))
        return 2 * np.sqrt(max(factor * (d_gg**2 - d_sg**2), 0))

    def _flatten_state(self, state: ObsType) -> np.ndarray:
        """Flatten state to array."""
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32)
        return np.array(state).flatten().astype(np.float32)

    def _compute_obs_statistics(self, state_pairs: list[tuple[ObsType, ObsType]]) -> None:
        """Compute mean and std of observations for normalization."""
        all_obs = []
        for source, target in state_pairs:
            all_obs.append(self._flatten_state(source))
            all_obs.append(self._flatten_state(target))

        all_obs_array = np.array(all_obs, dtype=np.float32)
        self._obs_mean = np.mean(all_obs_array, axis=0)
        self._obs_std = np.std(all_obs_array, axis=0)

        print(f"Computed observation statistics: mean={self._obs_mean.mean():.3f}, std={self._obs_std.mean():.3f}")

    def save(self, path: str) -> None:
        """Save the distance heuristic.

        Args:
            path: Directory path to save model
        """
        assert self.psi_net is not None, "Model must be trained before saving"

        os.makedirs(path, exist_ok=True)

        # Save encoder network weights
        torch.save(self.psi_net.state_dict(), f"{path}/psi_net.pt")

        # Save A and log_lambda
        torch.save(self.A, f"{path}/A.pt")
        torch.save(self.log_lambda, f"{path}/log_lambda.pt")

        # Save network architecture info
        with open(f"{path}/network_config.pkl", "wb") as f:
            pickle.dump(
                {
                    "state_dim": self.psi_net.state_dim,
                    "latent_dim": self.psi_net.latent_dim,
                    "hidden_dims": self.config.hidden_dims,
                },
                f,
            )

        # Save observation statistics
        with open(f"{path}/obs_stats.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_mean": self._obs_mean,
                    "obs_std": self._obs_std,
                },
                f,
            )

        print(f"Distance heuristic V3 saved to {path}")

    def load(self, path: str, perceiver: Any) -> None:
        """Load a pre-trained distance heuristic.

        Args:
            path: Directory path containing saved model
            perceiver: Perceiver to convert states to atoms (not used in V3)
        """
        # Store perceiver
        self.perceiver = perceiver

        # Load observation statistics
        with open(f"{path}/obs_stats.pkl", "rb") as f:
            stats = pickle.load(f)
            self._obs_mean = stats["obs_mean"]
            self._obs_std = stats["obs_std"]

        # Load network config
        with open(f"{path}/network_config.pkl", "rb") as f:
            network_config = pickle.load(f)

        # Create encoder network with saved architecture
        self.psi_net = StateEncoder(
            state_dim=network_config["state_dim"],
            latent_dim=network_config["latent_dim"],
            hidden_dims=network_config["hidden_dims"],
        ).to(self.device)

        # Load weights
        self.psi_net.load_state_dict(torch.load(f"{path}/psi_net.pt", map_location=self.device))
        self.psi_net.eval()

        # Load A and log_lambda
        self.A = torch.load(f"{path}/A.pt", map_location=self.device)
        self.log_lambda = torch.load(f"{path}/log_lambda.pt", map_location=self.device)

        print(f"Distance heuristic V3 loaded from {path}")
