import time
import os
import pickle
import random
import wandb
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar, Callable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
import gymnasium as gym


from tamp_improv.approaches.improvisational.heuristics.base import BaseHeuristic
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from tamp_improv.approaches.improvisational.policies.base import ObsType
    from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class CMDHeuristicConfig:
    """Configuration for contrastive state-node distance heuristic."""
    wandb_enabled: bool = False  # Whether to enable Weights & Biases logging

    # Pruning
    threshold: float = 0.05
    beta: float = 1 # Complex distance scaling parameter

    # Network architecture
    latent_dim: int = 32  # Dimension of embedding space (k)
    hidden_dims: list[int] | None = None  # Hidden layer sizes [64, 64]
    normalize_embeddings: bool = True  # Whether to L2-normalize embeddings to unit sphere

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    buffer_size: int = 10000
    grad_clip: float = 1.0  # Gradient clipping threshold

    # Contrastive learning
    gamma: float = 0.99  # For geometric sampling of future states
    repetition_factor: int = 4  # CRTR: repeat each trajectory this many times in batch
    policy_temperature: float = 1.0
    eval_temperature: float = 0.0
    num_action_samples: int = 4

    # Training
    iters_per_epoch: int = 1  # Gradient steps per training call
    learn_frequency: int = 10  # Learn every N epochs
    num_rounds: int = 5
    num_epochs_per_round: int = 200
    trajectories_per_epoch: int = 10
    max_episode_steps: int = 100
    keep_fraction: float = 0.5

    # Multi-round training
    num_rounds: int = 1  # Number of pruning rounds (1 = no pruning, just normal train)
    keep_fraction: float = 0.1  # Fraction of node-node pairs to keep each round
    exploration_factor: float = 0

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

class GoalEncoder(nn.Module):
    """Embedding matrix for discrete goal nodes.

    Maps node ID (integer) -> embedding (k-dimensional vector).
    Since nodes are discrete, we use a simple embedding matrix rather than a neural network.
    """

    def __init__(self, num_nodes: int, latent_dim: int):
        """Initialize goal encoder.

        Args:
            num_nodes: Number of discrete goal nodes
            latent_dim: Dimension of output embedding (k)
        """
        super(GoalEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.out_dim = latent_dim

        # Learnable embedding matrix (num_nodes x latent_dim)
        self.embedding = nn.Embedding(num_nodes, latent_dim)

        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Encode node IDs to embeddings.

        Args:
            node_ids: Tensor of node IDs (batch_size,) or scalar

        Returns:
            Embeddings (batch_size, latent_dim) or (latent_dim,)
        """
        return self.embedding(node_ids)


class CostNet(nn.Module):
    """Scalar network that outputs c(g) for each goal node.

    This is a simple embedding that maps each goal node to a scalar cost value.
    In CMD-1, the energy function is f(s,g) = c(g) - d(s,g), where c(g) is
    an upper bound on the cost-to-go from any state to goal g.
    """

    def __init__(self, g_encoder: GoalEncoder):
        """Initialize cost network.

        Args:
            num_nodes: Number of discrete goal nodes
        """
        super(CostNet, self).__init__()

        self.g_encoder = g_encoder

        # Get output dimensions from encoders
        goal_enc_dim = g_encoder.latent_dim

        self.scalar_net = nn.Sequential(nn.Linear(goal_enc_dim, goal_enc_dim // 2),
                                        nn.ReLU(),
                                        nn.Linear(goal_enc_dim // 2, 1))


    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Get cost value c(g) for goal nodes.

        Args:
            g: Tensor of node ids (batch_size,) or scalar

        Returns:
            Cost values (batch_size, 1) or (1,)
        """

        goal_encodings = self.g_encoder(g)

        return self.scalar_net(goal_encodings)
    

class MRN(nn.Module):
    """Metric Residual Network for learning quasi-metric distances.

    Combines symmetric and asymmetric components to learn d(s,g) that
    respects triangle inequality. Returns NEGATIVE distance (for energy function).
    """

    def __init__(self, s_encoder: StateEncoder, g_encoder: GoalEncoder, sym_dim: int = 64, asym_dim: int = 16):
        """Initialize MRN.

        Args:
            s_encoder: State encoder network
            g_encoder: Goal encoder network
            sym_dim: Dimension of symmetric component
            asym_dim: Dimension of asymmetric component
        """
        super().__init__()
        self.s_encoder = s_encoder
        self.g_encoder = g_encoder

        # Get output dimensions from encoders
        state_enc_dim = s_encoder.latent_dim
        goal_enc_dim = g_encoder.latent_dim

        # Symmetrical head
        self.s_sym = nn.Linear(state_enc_dim, sym_dim)
        self.g_sym = nn.Linear(goal_enc_dim, sym_dim)

        # Asymmetrical head
        self.s_asym = nn.Linear(state_enc_dim, asym_dim)
        self.g_asym = nn.Linear(goal_enc_dim, asym_dim)

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Compute negative distance -d(s,g).

        Args:
            s: State tensor (batch_size, state_dim) or (state_dim,)
            g: Goal node IDs (batch_size,) or scalar

        Returns:
            Negative distances (batch_size, 1) or (1,)
        """
        # Encode state and goal
        s_enc = self.s_encoder(s)
        g_enc = self.g_encoder(g)

        # Symmetric component
        sym_s = self.s_sym(s_enc)
        sym_g = self.g_sym(g_enc)
        dist_sym = (sym_s - sym_g).pow(2).mean(-1, keepdim=True)

        # Asymmetric component
        asym_s = self.s_asym(s_enc)
        asym_g = self.g_asym(g_enc)
        res = F.relu(asym_s - asym_g)
        dist_asym = (F.softmax(res, -1) * res).sum(-1, keepdim=True)

        # Total distance (both components are non-negative)
        dist = dist_sym + dist_asym

        # Return NEGATIVE distance for energy function
        return -dist
    
class ContrastiveReplayBuffer:
    """Replay buffer for storing complete trajectories.

    Implements CRTR (Contrastive Representations for Temporal Reasoning) sampling
    to encourage learning temporal dynamics rather than static context.

    For V5: Stores trajectories as sequences of node IDs rather than states.
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
        # Each trajectory is a dict with:
        #   - 'states': list of flattened states (NDArray)
        #   - 'nodes': list of node IDs (int)
        self.trajectories: deque = deque(maxlen=capacity)

        # Store metadata for each trajectory (start state, goal node)
        self.trajectory_metadata: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return total number of trajectories stored."""
        return len(self.trajectories)

    def store_trajectory(
        self,
        states: list[NDArray],
        nodes: list[int],
        start_state: NDArray | None = None,
        goal_node: int | None = None
    ) -> None:
        """Store complete trajectory with metadata.

        Args:
            states: List of flattened states visited during episode
            nodes: List of node IDs corresponding to states
            start_state: Starting state of the trajectory
            goal_node: Goal node ID for the trajectory
        """
        if len(states) < 2 or len(states) != len(nodes):
            return

        # Store the trajectory
        self.trajectories.append({
            'states': states,
            'nodes': nodes,
        })

        # Store metadata
        self.trajectory_metadata.append({
            'start_state': start_state,
            'goal_node': goal_node,
        })

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample batch using CRTR strategy.

        Implements Algorithm 1 from "Contrastive Representations for Temporal Reasoning":
        For each sample in batch:
        1. Sample a trajectory uniformly
        2. Sample t0 uniformly from [0, T-1]
        3. Sample t1 from geometric distribution starting at t0+1

        Returns:
            Dictionary containing:
                - current_states: (batch_size, state_dim) - states at time t0
                - future_nodes: (batch_size,) - node IDs at time t1
        """
        if len(self.trajectories) == 0:
            raise ValueError("Buffer is empty")

        # Ensure batch_size is divisible by repetition_factor
        assert batch_size % self.repetition_factor == 0

        # Sample unique trajectory IDs
        num_unique_trajs = batch_size // self.repetition_factor
        unique_traj_ids = np.random.choice(
            len(self.trajectories),
            size=num_unique_trajs,
            replace=False if num_unique_trajs <= len(self.trajectories) else True
        )

        # Repeat each trajectory ID repetition_factor times
        traj_ids = np.repeat(unique_traj_ids, self.repetition_factor)

        current_states = []
        future_nodes = []

        # For each trajectory in the batch, sample (t0, t1) pair
        for traj_id in traj_ids:
            trajectory = self.trajectories[traj_id]
            states = trajectory['states']
            nodes = trajectory['nodes']
            traj_len = len(states)

            if traj_len < 2:
                current_states.append(states[0])
                future_nodes.append(nodes[-1])
                continue

            # Sample t0 uniformly from trajectory
            t0 = np.random.randint(0, traj_len - 1)

            # Sample t1 using geometric distribution
            num_future = traj_len - t0 - 1
            p = 1 - self.gamma
            offset = min(
                np.random.geometric(p) - 1,
                num_future - 1
            )
            # print("Offset:", offset)
            t1 = t0 + 1 + offset

            # print("Start node:", nodes[t0])
            # print("Goal node:", nodes[t1])

            current_states.append(states[t0])
            future_nodes.append(nodes[t1])

        # Stack into tensors
        current_states = torch.FloatTensor(np.array(current_states))
        future_nodes = torch.LongTensor(future_nodes)

        return {
            "current_states": current_states,
            "future_nodes": future_nodes,
        }



def cmd_contrastive_loss(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    current_states: torch.Tensor,
    future_nodes: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    CMD-1 contrastive loss with duplicate-goal collapse fixed,
    and ignoring goals marked as -1.
    """

    device = current_states.device
    B = current_states.shape[0]

    # --------------------------------------------------
    # 1. Mask invalid goals (-1)
    # --------------------------------------------------
    valid_mask = future_nodes != -1
    if valid_mask.sum() == 0:
        # All goals invalid, return zero loss
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "loss": 0.0,
            "accuracy": 1.0,
            "pos_energy_mean": 0.0,
            "neg_energy_mean": 0.0,
            "energy_gap": 0.0,
            "num_unique_goals": 0,
        }

    valid_states = current_states[valid_mask]
    valid_goals = future_nodes[valid_mask]

    # --------------------------------------------------
    # 2. Deduplicate goals
    # --------------------------------------------------
    unique_goals, inverse_indices = torch.unique(
        valid_goals, return_inverse=True
    )
    G = unique_goals.shape[0]

    if G == 1:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "loss": 0.0,
            "accuracy": 1.0,
            "pos_energy_mean": 0.0,
            "neg_energy_mean": 0.0,
            "energy_gap": 0.0,
            "num_unique_goals": 1,
        }

    # --------------------------------------------------
    # 3. Build (state, unique-goal) energy matrix
    # --------------------------------------------------
    states_expanded = valid_states.unsqueeze(1).expand(-1, G, -1)
    goals_expanded = unique_goals.unsqueeze(0).expand(valid_states.shape[0], G)

    states_flat = states_expanded.reshape(-1, current_states.shape[-1])
    goals_flat = goals_expanded.reshape(-1)

    energies = energy_fn(states_flat, goals_flat)
    energy_matrix = energies.view(valid_states.shape[0], G)

    logits = energy_matrix / temperature
    labels = inverse_indices

    # --------------------------------------------------
    # 4. Compute loss
    # --------------------------------------------------
    loss = F.cross_entropy(logits, labels)

    # --------------------------------------------------
    # 5. Metrics
    # --------------------------------------------------
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        pos_energies = energy_matrix[
            torch.arange(valid_states.shape[0], device=device), labels
        ]

        neg_mask = torch.ones_like(energy_matrix, dtype=bool)
        neg_mask[torch.arange(valid_states.shape[0]), labels] = False
        neg_energies = energy_matrix[neg_mask]

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "pos_energy_mean": pos_energies.mean().item(),
        "neg_energy_mean": neg_energies.mean().item(),
        "energy_gap": (pos_energies.mean() - neg_energies.mean()).item(),
        "num_unique_goals": G,
        "num_valid_states": valid_states.shape[0],
    }

    return loss, metrics



class CMDHeuristic(BaseHeuristic):
    """Contrastive state-node distance heuristic.

    Learns an embedding space where L2 distance correlates with trajectory distance.
    Conditions policy on goal NODES rather than goal STATES.
    """

    def __init__(
        self,
        training_data: "GoalConditionedTrainingData",
        graph_distances: dict[tuple[int, int], float],
        system: "ImprovisationalTAMPSystem",
        config: CMDHeuristicConfig,
        seed: int | None = None,
    ):
        """Initialize distance heuristic V4.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        self.env = system.env
        self.perceiver = system.perceiver
        self.node_to_states = training_data.node_states
        self.graph_distances= graph_distances
        self.training_data = training_data
        
        
        # Extract state-node pairs (state from node A, target node B)
        self.atoms_to_node = {}
        planning_graph = training_data.graph
        for node in planning_graph.nodes:
            self.atoms_to_node[node.atoms] = node.id

        state_node_pairs = []
        for node_id, states in self.node_to_states.items():
            # Find node by atoms

            for state in states:

                # Add pairs to all other nodes
                for target_node in planning_graph.nodes:
                    if target_node.id != node_id:
                        state_node_pairs.append((state, target_node.id))

        self.state_node_pairs = state_node_pairs

        # Create get_node function
        def get_node(state: ObsType) -> int:
            """Convert state to node ID."""
            atoms = self.perceiver.step(state)
            # print(atoms)
            # print(self.atoms_to_node)
            if frozenset(atoms) not in self.atoms_to_node:
                return -1
            return self.atoms_to_node[frozenset(atoms)]



        self.get_node = get_node
        # Callable[[ObsType], int]
        self.num_nodes = len(self.atoms_to_node)

        if self.config.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Training state
        self.total_episodes = 0
        self.policy_temperature = config.policy_temperature

        # Determine state dimension
        sample_state = state_node_pairs[0][0]
        state_flat = self._flatten_state(sample_state)
        state_dim = state_flat.shape[0]

        print(f"State dimension: {state_dim}")
        print(f"Latent dimension: {self.config.latent_dim}")

        # Recreate networks
        self.s_encoder = StateEncoder(
            state_dim=state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        self.g_encoder = GoalEncoder(
            num_nodes=self.num_nodes,
            latent_dim=self.config.latent_dim,
        ).to(self.device)


        # Create cost network c(g)
        self.c_net = CostNet(g_encoder=self.g_encoder).to(self.device)

        # Create distance network d(s,g) using MRN
        self.d_net = MRN(
            s_encoder=self.s_encoder,
            g_encoder=self.g_encoder,
            sym_dim=64,
            asym_dim=16,
        ).to(self.device)

        # Create optimizer for all parameters
        all_params = (
            list(self.s_encoder.parameters()) +
            list(self.g_encoder.parameters()) +
            list(self.c_net.parameters()) +
            list(self.d_net.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.learning_rate)

        # Create cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs_per_round * self.config.num_rounds,
            eta_min=self.config.learning_rate
        )

        # Create replay buffer
        self.replay_buffer = ContrastiveReplayBuffer(
            capacity=self.config.buffer_size,
            gamma=self.config.gamma,
            repetition_factor=self.config.repetition_factor,
        )


    def energy(self, states: torch.Tensor, goal_nodes: torch.Tensor) -> torch.Tensor:
        """Compute energy function f(s,g) = c(g) - d(s,g).

        Args:
            states: State tensor (batch_size, state_dim)
            goal_nodes: Goal node IDs (batch_size,)

        Returns:
            Energy values (batch_size,)
        """
        # c(g): cost upper bound for each goal
        c_g = self.c_net(goal_nodes).squeeze(-1)  # (batch_size,)

        # -d(s,g): negative distance (MRN returns negative distance)
        neg_d_sg = self.d_net(states, goal_nodes).squeeze(-1)  # (batch_size,)

        print("C:", c_g.mean().item(), " -D:", neg_d_sg.mean().item())

        # f(s,g) = c(g) - d(s,g) = c(g) + (-d(s,g))
        energy = c_g + neg_d_sg

        return energy


    def _select_action_greedy(
            self,
            env: Any,
            current_state,
            goal_node: int,
        ) -> Any | None:

                """
                Soft greedy action selection over embedding distances.

                policy_temperature == 0:
                    - Discrete action space  -> argmin distance
                    - Continuous action space -> weighted mean of sampled actions

                policy_temperature > 0:
                    - Sample from softmax distribution (both discrete and continuous)
                """
                num_action_samples = self.config.num_action_samples
                action_space = env.action_space
                tau = self.policy_temperature

                # ------------------------------------------------------------
                # 1. Collect candidate actions
                # ------------------------------------------------------------

                if isinstance(action_space, gym.spaces.Discrete):
                    candidate_actions = list(range(action_space.n))
                    is_discrete = True

                elif isinstance(action_space, gym.spaces.Box):
                    candidate_actions = [
                        action_space.sample() for _ in range(num_action_samples)
                    ]
                    is_discrete = False

                else:
                    raise NotImplementedError(
                        f"Unsupported action space type: {type(action_space)}"
                    )

                if len(candidate_actions) == 0:
                    return None

                # ------------------------------------------------------------
                # 2. Fallback: random policy if encoders are uninitialized
                # ------------------------------------------------------------

                if self.s_encoder is None or self.g_encoder is None:
                    return random.choice(candidate_actions)

                # Create goal node tensor
                goal_tensor = torch.tensor([goal_node], dtype=torch.long, device=self.device)

                # ------------------------------------------------------------
                # 4. Score candidate actions
                # ------------------------------------------------------------

                 # Compute distances for each action
                distances = []
                for action in candidate_actions:
                    # Save current state
                    env.reset_from_state(current_state)

                    # Take action to get next state
                    next_state, _, _, _, _ = env.step(action)

                    # Flatten and tensorize next state
                    next_flat = self._flatten_state(next_state)
                    next_tensor = torch.FloatTensor(next_flat).unsqueeze(0).to(self.device)

                    # Compute distance: d(s',g) = -MRN(s',g)
                    with torch.no_grad():
                        neg_dist = self.d_net(next_tensor, goal_tensor).squeeze()  # MRN returns -d
                        dist = -neg_dist.item()  # Get positive distance

                    distances.append(dist)

                distances = np.asarray(distances)  # (N,)

                # ------------------------------------------------------------
                # 5. Zero-temperature limit
                # ------------------------------------------------------------

                if tau == 0:
                    if is_discrete:
                        # Hard greedy
                        idx = int(np.argmin(distances))
                        return candidate_actions[idx]

                    else:
                        # Continuous: weighted mean (MPPI-style)
                        # Use exp(-d) *without* dividing by tau
                        weights = np.exp(-distances)
                        weights /= np.sum(weights)

                        actions = np.stack(candidate_actions)  # (N, action_dim)
                        action = np.sum(actions * weights[:, None], axis=0)

                        return np.clip(
                            action,
                            action_space.low,
                            action_space.high,
                        )

                # ------------------------------------------------------------
                # 6. Soft (stochastic) policy
                # ------------------------------------------------------------

                logits = -distances / tau
                logits -= np.max(logits)  # numerical stability
                probs = np.exp(logits)
                probs /= np.sum(probs)

                idx = np.random.choice(len(candidate_actions), p=probs)
                return candidate_actions[idx]

    def rollout(
        self,
        env: Any,
        start_state: ObsType,
        goal_node: int,
        max_steps: int = 100,
        temperature: float | None = None,
    ) -> tuple[list[NDArray], list[int], bool]:
        """Rollout the learned policy from start state to goal node.

        Args:
            env: Environment
            start_state: Starting state
            goal_node: Goal node ID to reach
            max_steps: Maximum number of steps
            temperature: Policy temperature (uses self.policy_temperature if None)

        Returns:
            Tuple of (states, nodes, success) where:
                - states: List of flattened states visited
                - nodes: List of node IDs visited
                - success: Whether the goal node was reached
        """
        if temperature is not None:
            old_temp = self.policy_temperature
            self.policy_temperature = temperature

        # Reset to start state
        current_state, _ = env.reset_from_state(start_state)

        states = [self._flatten_state(current_state)]
        # print(self.get_node)
        nodes = [self.get_node(current_state)]

        success = False


        for _ in range(max_steps):
            # Check if reached goal
            current_node = self.get_node(current_state)
            if current_node == goal_node:
                success = True
                break

            # Select action
            action = self._select_action_greedy(env, current_state, goal_node)
            if action is None:
                break

            # Step environment
            current_state, _, _, _, _ = env.step(action)

            # Store state and node
            states.append(self._flatten_state(current_state))
            nodes.append(self.get_node(current_state))

        # Restore temperature
        if temperature is not None:
            self.policy_temperature = old_temp

        return states, nodes, success

    def collect_trajectories(
        self,
        env: Any,
        state_node_pairs: list[tuple[ObsType, int]],
        num_trajectories: int,
        max_episode_steps: int = 100,
    ) -> tuple[list[dict], list[dict], dict]:
        """Collect trajectories by rolling out from state-node pairs.

        Uses learned policy (greedy over embedding space) if available,
        otherwise uses random policy.

        Args:
            env: Base environment (must support reset_from_state)
            state_node_pairs: List of (source_state, target_node) pairs
            num_trajectories: Number of trajectories to collect
            max_episode_steps: Maximum steps per trajectory

        Returns:
            Tuple of (trajectories, trajectory_metadata, stats_dict)
        """
        trajectories = []
        trajectory_metadata = []
        successes = []
        lengths = []

        for _ in range(num_trajectories):
            # Sample random state-node pair
            source_state, target_node = random.choice(state_node_pairs)

            # Rollout from source to target
            states, nodes, success = self.rollout(
                env, source_state, target_node, max_steps=max_episode_steps
            )

            trajectories.append({
                'states': states,
                'nodes': nodes,
            })
            trajectory_metadata.append({
                'start_state': self._flatten_state(source_state),
                'goal_node': target_node,
            })
            successes.append(success)
            lengths.append(len(states))

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
        state_node_pairs: list[tuple[ObsType, int]],
        num_epochs: int = 1000,
        trajectories_per_epoch: int = 10,
        max_episode_steps: int = 100,
    ) -> dict:
        """Train distance heuristic using contrastive state-node learning.

        Args:
            env: Base environment
            state_node_pairs: List of (source_state, target_node) pairs
            perceiver: Perceiver to convert states to atoms
            atoms_to_node: Dictionary mapping frozenset of atoms to node ID
            num_epochs: Number of training epochs
            trajectories_per_epoch: Number of trajectories to collect per epoch
            max_episode_steps: Maximum steps per episode

        Returns:
            Dictionary containing training history with keys:
                - total_loss: List of total loss values per epoch
                - alignment_loss: List of alignment loss values per epoch
                - uniformity_loss: List of uniformity loss values per epoch
                - accuracy: List of accuracy values per epoch
                - success_rate: List of policy success rates per epoch
                - avg_success_length: List of avg successful trajectory lengths per epoch
                - learning_rate: List of learning rates per epoch
                - policy_temperature: List of policy temperatures per epoch
        """
        print(f"\nTraining distance heuristic V4 on {len(state_node_pairs)} state-node pairs...")
        print(f"Device: {self.device}")
        print(f"Number of nodes: {self.num_nodes}")


        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Collecting {trajectories_per_epoch} trajectories per epoch")
        print(f"Using cosine annealing LR scheduler: {self.config.learning_rate} -> {self.config.learning_rate * 0.01}")

        initial_temp = self.config.policy_temperature
        min_temp = self.config.eval_temperature

        # Initialize training history tracking
        training_history = {
            'total_loss': [],
            'accuracy': [],
            'pos_energy_mean': [],
            'neg_energy_mean': [],
            'energy_gap': [],
            'success_rate': [],
            'avg_success_length': [],
            'learning_rate': [],
            'policy_temperature': []
        }

        # Training loop
        for epoch in range(num_epochs):
            print("Epoch", epoch)

            if self.config.num_rounds == 1:
                # Update policy temperature using cosine annealing (per-epoch)
                progress = (epoch+1) / num_epochs
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                self.policy_temperature = min_temp + (initial_temp - min_temp) * cosine_factor
            

            # Collect trajectories
            trajectories, trajectory_metadata, traj_stats = self.collect_trajectories(
                self.env,
                state_node_pairs,
                num_trajectories=trajectories_per_epoch,
                max_episode_steps=max_episode_steps,
            )

            # Store trajectories in replay buffer with metadata
            for trajectory, metadata in zip(trajectories, trajectory_metadata):
                self.replay_buffer.store_trajectory(
                    states=trajectory['states'],
                    nodes=trajectory['nodes'],
                    start_state=metadata['start_state'],
                    goal_node=metadata['goal_node']
                )
                self.total_episodes += 1

            # Train if we have enough data (only every learn_frequency epochs)
            if (epoch > 0 and
                epoch % self.config.learn_frequency == 0 and
                len(self.replay_buffer) >= self.config.batch_size):

                metrics = self._train_step()

                # Track metrics in history
                current_lr = self.optimizer.param_groups[0]['lr']
                training_history['total_loss'].append(metrics['loss'])
                training_history['accuracy'].append(metrics['accuracy'])
                training_history['pos_energy_mean'].append(metrics['pos_energy_mean'])
                training_history['neg_energy_mean'].append(metrics['neg_energy_mean'])
                training_history['energy_gap'].append(metrics['energy_gap'])
                training_history['success_rate'].append(traj_stats['success_rate'])
                training_history['avg_success_length'].append(traj_stats['avg_success_length'])
                training_history['learning_rate'].append(current_lr)
                training_history['policy_temperature'].append(self.policy_temperature)

                if self.config.wandb_enabled:
                    wandb.log({
                        "train/total_loss": metrics['loss'],
                        "train/accuracy": metrics['accuracy'],
                        "train/pos_energy_mean": metrics['pos_energy_mean'],
                        "train/neg_energy_mean": metrics['neg_energy_mean'],
                        "train/energy_gap": metrics['energy_gap'],
                        "train/success_rate": traj_stats['success_rate'],
                        "train/avg_success_length": traj_stats['avg_success_length'],
                        "train/learning_rate": current_lr,
                        "train/policy_temperature": self.policy_temperature,
                    })

                # Logging after learning
                print(f"\n[Epoch {epoch}/{num_epochs}]")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                print(f"  Learning rate: {current_lr:.6f}")
                print(f"  Policy temperature: {self.policy_temperature:.4f}")
                print(f"  --- Policy Performance ---")
                print(f"  Success rate: {traj_stats['success_rate']:.2%} ({traj_stats['num_successes']}/{traj_stats['num_trajectories']})")
                print(f"  Avg trajectory length: {traj_stats['avg_length']:.1f}")
                if traj_stats['num_successes'] > 0:
                    print(f"  Avg successful length: {traj_stats['avg_success_length']:.1f}")
                print(f"  --- Training Metrics ---")
                print(f"  Total loss: {metrics['loss']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  Positive energy: {metrics['pos_energy_mean']:.4f}")
                print(f"  Negative energy: {metrics['neg_energy_mean']:.4f}")
                print(f"  Energy gap: {metrics['energy_gap']:.4f}")

            # Step the learning rate scheduler every epoch
            self.scheduler.step()

        print(f"\nTraining complete!")
        return training_history


    def _train_step(self) -> dict:
        """Perform one training step (batch update).

        Returns:
            Dictionary of training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        # Move to device
        current_states = batch["current_states"].to(self.device)
        future_nodes = batch["future_nodes"].to(self.device)

        # Run gradient descent
        for _ in range(self.config.iters_per_epoch):
            self.optimizer.zero_grad()

            # Compute loss using CMD-1 contrastive loss
            loss, metrics = cmd_contrastive_loss(
                self.energy,
                current_states,
                future_nodes,
                temperature=self.policy_temperature,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            all_params = (
                list(self.s_encoder.parameters()) +
                list(self.g_encoder.parameters()) +
                list(self.c_net.parameters()) +
                list(self.d_net.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.config.grad_clip)

            # Update parameters
            self.optimizer.step()

        return metrics
    

    def estimate_distance(self, source_state: ObsType, target_node: int) -> float:
        """Estimate trajectory distance from state to node using learned distance function.

        In CMD-1/V5, we directly use the learned distance network d(s,g).

        Args:
            source_state: Source state
            target_node: Target node ID

        Returns:
            Estimated distance (number of steps) to reach target from source
        """
        if self.d_net is None:
            return 0.0

        source_flat = self._flatten_state(source_state)

        # Compute d(s,g) using the MRN
        with torch.no_grad():
            source_tensor = torch.FloatTensor(source_flat).unsqueeze(0).to(self.device)
            target_tensor = torch.tensor([target_node], dtype=torch.long, device=self.device)

            # MRN returns -d(s,g), so negate to get positive distance
            neg_distance = self.d_net(source_tensor, target_tensor).squeeze()
            distance = -neg_distance.item()
        
        return -(1/np.log(self.config.gamma)) * float(distance)

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        source_states = self.node_to_states[source_node]
        avg = 0
        n = 0
        for source_state in random.sample(source_states, min(100, len(source_states))):
            avg += self.estimate_distance(source_state, target_node)
            n += 1
        return avg / n
    
    def estimate_probability(self, source_node: int, target_node: int) -> float:
        est_dist = self.estimate_node_distance(source_node, target_node)

        if est_dist <= 0:
            p_rr = 1.0
        else:
            p_rr = np.clip(self.config.beta * np.exp(-(est_dist)**2 / (2 * self.config.max_episode_steps)), 0, 1)
        

        print("Rollout probability of reaching node", target_node, "from node", source_node, ":", p_rr, "for distance", est_dist)

        k = np.log(0.5) / np.log(1 - self.config.threshold)
        return 1 - (1 - p_rr)**k
    
    # def estimate_gain(self, source_node: int, target_node: int) -> float:
    #     """Estimate gain of training on a shortcut, relative to distance in the
    #     initial graph. Higher gain means more useful shortcut."""

    #     graph_distance = self.graph_distances.get((source_node, target_node), float('inf'))
    #     estimated_distance = self.estimate_node_distance(source_node, target_node)
        
    #     # Use this if we're learning too many backwards shortcuts. Might not be a great idea in general.
    #     # gain = max(graph_distance - estimated_distance, 0)
    #     # if math.isinf(gain):
    #     #     gain = 0
        
    #     gain = np.clip(graph_distance - estimated_distance, 0, self.config.max_episode_steps)
    #     return gain

    def estimate_gain(self, source_node: int, target_node: int) -> float:
        """Estimate gain of training on a shortcut, relative to distance in the
        initial graph. Higher gain means more useful shortcut."""

        # graph_distance = self.graph_distances.get((source_node, target_node), float('inf'))
        length = self.estimate_node_distance(source_node, target_node)
        
        # Use this if we're learning too many backwards shortcuts. Might not be a great idea in general.
        # gain = max(graph_distance - estimated_distance, 0)
        # if math.isinf(gain):
        #     gain = 0

        x = source_node
        y = target_node

        nodes = self.training_data.node_states.keys()


        def d(u, v):
            return self.graph_distances[(u, v)]

        # --- Phase 1: affected node sets ---
        A_x = [u for u in nodes if d(u, y) + length < d(u, x)]
        A_y = [v for v in nodes if d(v, x) + length < d(v, y)]

        # Iterate over smaller set (optimization)
        if len(A_y) < len(A_x):
            A_x, A_y = A_y, A_x
            x, y = y, x

        RG = 0.0

        # --- Phase 2: affected pairs ---
        for u in A_x:
            du_y = d(u, y)
            for v in A_y:
                d_old = d(u, v)
                d_new = du_y + length + d(x, v)

                if d_new < d_old:
                    RG += (d_old - d_new)

        gain = np.clip(RG, 0, self.config.max_episode_steps)
        return gain

    def prune_explore(self) -> GoalConditionedTrainingData:
        print(f"\n[DEBUG] Pruning with keep_fraction={self.config.keep_fraction}, max_episode_steps={self.config.max_episode_steps}")

        # Score all node pairs: score = estimated_distance - graph_distance
        # Negative scores = shortcuts (estimated < graph)
        # Lower scores = more promising

        score_tuples = []
        for source_id, target_id in self.training_data.unique_shortcuts:
            p = self.estimate_probability(source_id, target_id)
            g = self.estimate_gain(source_id, target_id)
            score = p * g
            score_tuples.append((source_id, target_id, score, p, g))

        # Sort score tuples first by score, and then by probability
        score_tuples.sort(key=lambda x: (x[2], x[3]), reverse=True)
        print("  Shortcut scores (source -> target: score (dist, prob, gain)):")
        for source_id, target_id, score, p, g in score_tuples:
            d = self.estimate_node_distance(source_id, target_id)
            print(f"    ({source_id} -> {target_id}): {score:.4f} ({d:.2f}, {p:.2f}, {g:.2f})")
        
        

        # # Strategy 1: Keep all negative scores (shortcuts)
        # shortcuts = set()
        # for score, source_node, target_node, est_dist in node_pair_scores:
        #     if score < 0:
        #         shortcuts.add((source_node, target_node))

        # print(f"[DEBUG] Found {len(shortcuts)} shortcuts (negative scores)")

        # Strategy 2: Keep top keep_fraction by score
        num_to_keep = max(1, int(len(score_tuples) * self.config.keep_fraction))
        top_pairs = set()
        for source_node, target_node, score, p, g in score_tuples[:num_to_keep]:
            top_pairs.add((source_node, target_node))

        print(f"[DEBUG] Keeping top {num_to_keep} pairs ({self.config.keep_fraction:.1%})")

        # Strategy 3: Add exploration pairs randomly
        num_exploration = int(len(self.training_data.unique_shortcuts) * self.config.exploration_factor)  # 5% exploration
        all_pairs = set(self.training_data.unique_shortcuts)
        unexplored_pairs = all_pairs - top_pairs
        exploration_pairs = random.sample(list(unexplored_pairs), min(num_exploration, len(unexplored_pairs)))
        for source_node, target_node in exploration_pairs:
            top_pairs.add((source_node, target_node))
        print(f"[DEBUG] Added {len(exploration_pairs)} exploration pairs")

        # Return union
        result = top_pairs
        print(f"[DEBUG] Total pairs to keep: {len(result)} (union of shortcuts and top)")


        # Combine result with self.training_data to get back a new TrainingData
        selected_set = set(result)
        selected_indices = []
        for i, (source_id, target_id) in enumerate(self.training_data.valid_shortcuts):
            if (source_id, target_id) in selected_set:
                selected_indices.append(i)
        
        print(f"  ({len(selected_indices)} state-node pairs)")
        # Filter shortcut_info to match the pruned data
        original_shortcut_info = self.training_data.config.get("shortcut_info", [])
        pruned_shortcut_info = (
            [original_shortcut_info[i] for i in selected_indices]
            if original_shortcut_info
            else []
        )


        pruned_data = GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[self.training_data.current_atoms[i] for i in selected_indices],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[self.training_data.valid_shortcuts[i] for i in selected_indices],
            unique_shortcuts=list(selected_set),  # Unique node-node pairs
            node_states=self.training_data.node_states,  # Keep all node states
            node_atoms=self.training_data.node_atoms,  # Keep all node atoms
            graph=self.training_data.graph,  # Keep planning graph
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "crl",
            },
        )

        return pruned_data

    def prune(self, max_shortcuts: int | None, ) -> GoalConditionedTrainingData:
        """Like prune_explore, but only prunes the top max_shortcuts shortcuts from the list
        instead of keep_fraction/explore_fraction."""

        if max_shortcuts is None:
            return self.training_data

        print(f"\n[DEBUG] Pruning to max_shortcuts={max_shortcuts}")

        # Compute success rates and select shortcuts (use unique_shortcuts for node-node pairs)
        score_tuples = []
        for source_id, target_id in self.training_data.unique_shortcuts:
            p = self.estimate_probability(source_id, target_id)
            g = self.estimate_gain(source_id, target_id)
            score = p * g
            score_tuples.append((source_id, target_id, score, p, g))
        
        # Sort score tuples first by score, and then by probability
        score_tuples.sort(key=lambda x: (x[2], x[3]), reverse=True)
        print("  Shortcut scores (source -> target: score (dist, prob, gain)):")
        for source_id, target_id, score, p, g in score_tuples:
            d = self.estimate_node_distance(source_id, target_id)
            dg = self.graph_distances.get((source_id, target_id), np.inf)
            print(f"    ({source_id} -> {target_id}): {score:.4f} ({d:.2f}, {dg:.2f}, {p:.2f}, {g:.2f})")
        
        # Select top max_shortcuts shortcuts
        selected_shortcuts = score_tuples[:max_shortcuts]
        selected_unique_shortcuts = [
            (source_id, target_id) for source_id, target_id, _, _, _ in selected_shortcuts
        ]

        # Filter training data to match selected unique shortcuts
        # Keep all state-node pairs that correspond to selected node-node pairs
        selected_set = set(selected_unique_shortcuts)
        selected_indices = []

        for i, (source_id, target_id) in enumerate(self.training_data.valid_shortcuts):
            if (source_id, target_id) in selected_set:
                selected_indices.append(i)

        print(f"  ({len(selected_indices)} state-node pairs)")

        # Filter shortcut_info to match the pruned data
        original_shortcut_info = self.training_data.config.get("shortcut_info", [])
        pruned_shortcut_info = (
            [original_shortcut_info[i] for i in selected_indices]
            if original_shortcut_info
            else []
        )

        pruned_data = GoalConditionedTrainingData(
            states=[self.training_data.states[i] for i in selected_indices],
            current_atoms=[self.training_data.current_atoms[i] for i in selected_indices],
            goal_atoms=[self.training_data.goal_atoms[i] for i in selected_indices],
            valid_shortcuts=[self.training_data.valid_shortcuts[i] for i in selected_indices],
            unique_shortcuts=selected_unique_shortcuts,  # Unique node-node pairs
            node_states=self.training_data.node_states,  # Keep all node states
            node_atoms=self.training_data.node_atoms,  # Keep all node atoms
            graph=self.training_data.graph,  # Keep planning graph
            config={
                **self.training_data.config,
                "shortcut_info": pruned_shortcut_info,
                "pruning_method": "crl",
                "threshold": self.config.threshold,
            },
        )

        return pruned_data

    def multi_train(
        self,
    ) -> dict:
        """Multi-round training that iteratively prunes to promising shortcuts.

        For each round:
        1. Train on current subset of state-node pairs
        2. Compute estimated vs graph distances at NODE-NODE level
        3. Keep top keep_fraction of NODE-NODE pairs with best differences
        4. Keep ALL state-node pairs for the selected node-node pairs
        5. Continue training on this pruned subset

        This focuses learning on node-node pairs that show promise for shortcuts.

        Args:
            state_node_pairs: List of (source_state, target_node) pairs
            graph_distances: Dict mapping (source_node, target_node) -> graph distance
            num_rounds: Number of pruning rounds
            num_epochs_per_round: Training epochs per round
            trajectories_per_epoch: Trajectories collected per epoch
            max_episode_steps: Max steps per trajectory
            keep_fraction: Fraction of NODE-NODE pairs to keep each round (0.5 = keep top 50%)

        Returns:
            Dictionary with combined training history across all rounds
        """
        state_node_pairs = self.state_node_pairs
        graph_distances = self.graph_distances


        num_rounds = self.config.num_rounds
        num_epochs_per_round = self.config.num_epochs_per_round
        trajectories_per_epoch = self.config.trajectories_per_epoch
        max_episode_steps = self.config.max_episode_steps
        keep_fraction = self.config.keep_fraction

        # wandb.init(project="slap_crl_heuristic", config=self.config.__dict__)

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING: {num_rounds} rounds, {num_epochs_per_round} epochs/round")
        print(f"Starting with {len(state_node_pairs)} state-node pairs")
        print(f"Keep fraction: {keep_fraction} (pruning {1-keep_fraction:.1%} each round)")
        print(f"{'='*80}\n")

        # Build mapping: (source_node, target_node) -> list of (source_state, target_node) pairs
        print("Building node-node to state-node mapping...")
        node_pair_to_state_pairs: dict[tuple[int, int], list[tuple[ObsType, int]]] = {}
        for state, target_node in state_node_pairs:
            source_node = self.get_node(state)
            key = (source_node, target_node)
            if key not in node_pair_to_state_pairs:
                node_pair_to_state_pairs[key] = []
            node_pair_to_state_pairs[key].append((state, target_node))

        print(f"Found {len(node_pair_to_state_pairs)} unique node-node pairs")
        print(f"Average {len(state_node_pairs) / len(node_pair_to_state_pairs):.1f} state-node pairs per node-node pair")

        # Combined history across all rounds
        combined_history = {
            'total_loss': [],
            'accuracy': [],
            'pos_energy_mean': [],
            'neg_energy_mean': [],
            'energy_gap': [],
            'success_rate': [],
            'avg_success_length': [],
            'learning_rate': [],
            'policy_temperature': [],
            'round_boundaries': [],  # Track where each round starts
            'num_state_pairs_per_round': [],  # Track state-node dataset size per round
            'num_node_pairs_per_round': [],  # Track node-node dataset size per round
            'distance_matrices': [],  # Distance matrix after each round (for debugging)
            'graph_distances': graph_distances,  # Store graph distances for reference
        }

        # Start with all node-node pairs
        current_node_pairs = list(node_pair_to_state_pairs.keys())

        initial_temp = self.config.policy_temperature
        min_temp = self.config.eval_temperature


        for round_idx in range(num_rounds):
            if num_rounds > 1:
                # Update policy temperature using cosine annealing (per-round)
                progress = (round_idx+0.5) / num_rounds
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                self.policy_temperature = min_temp + (initial_temp - min_temp) * cosine_factor

            # Get all state-node pairs for current node-node pairs
            current_state_pairs = []
            for node_pair in current_node_pairs:
                current_state_pairs.extend(node_pair_to_state_pairs[node_pair])

            print(f"\n{'='*80}")
            print(f"ROUND {round_idx + 1}/{num_rounds}")
            print(f"Training on {len(current_node_pairs)} node-node pairs")
            print(f"  -> {len(current_state_pairs)} state-node pairs")
            print(f"{'='*80}\n")

            # Mark start of this round in history
            combined_history['round_boundaries'].append(len(combined_history['total_loss']))
            combined_history['num_state_pairs_per_round'].append(len(current_state_pairs))
            combined_history['num_node_pairs_per_round'].append(len(current_node_pairs))

            # Train on current subset
            round_history = self.train(
                state_node_pairs=current_state_pairs,
                num_epochs=num_epochs_per_round,
                trajectories_per_epoch=trajectories_per_epoch,
                max_episode_steps=self.config.max_episode_steps,
            )

            # Append this round's history
            for key in round_history.keys():
                combined_history[key].extend(round_history[key])

            # Compute distance matrix for ALL node-node pairs (not just current ones)
            # This allows us to "bring back" previously pruned pairs if they become promising
            print(f"\n[DEBUG] Computing distance matrix for ALL {len(node_pair_to_state_pairs)} node pairs after round {round_idx + 1}...")
            all_node_pairs = list(node_pair_to_state_pairs.keys())
            distance_matrix = self._compute_distance_matrix(all_node_pairs)
            combined_history['distance_matrices'].append({
                'round': round_idx + 1,
                'distances': distance_matrix,
                'active_node_pairs': current_node_pairs.copy(),  # Which pairs we trained on
            })
            print(f"[DEBUG] Distance matrix computed with {len(distance_matrix)} entries")

            # If not the last round, prune to most promising NODE-NODE pairs
            # IMPORTANT: Evaluate ALL node pairs, not just current ones!
            if round_idx < num_rounds - 1:
                print(f"\n{'='*80}")
                print(f"PRUNING after round {round_idx + 1}")
                print(f"{'='*80}\n")

                # Call prune() to get selected node pairs
                new_data = self.prune_explore()
                selected_node_pairs_set = new_data.unique_shortcuts
                current_node_pairs = list(selected_node_pairs_set)

                # Count resulting state-node pairs
                resulting_state_pairs = sum(
                    len(node_pair_to_state_pairs.get((src, tgt), []))
                    for src, tgt in current_node_pairs
                )

                print(f"Pruned node-node pairs: {len(all_node_pairs)} -> {len(current_node_pairs)}")
                print(f"Resulting state-node pairs: {resulting_state_pairs}")

        # Final state-node pairs
        final_state_pairs = []
        for node_pair in current_node_pairs:
            final_state_pairs.extend(node_pair_to_state_pairs[node_pair])

        print(f"\n{'='*80}")
        print(f"MULTI-ROUND TRAINING COMPLETE")
        print(f"Final node-node pairs: {len(current_node_pairs)}")
        print(f"Final state-node pairs: {len(final_state_pairs)}")
        print(f"{'='*80}\n")

        # wandb.finish()

        return combined_history
    
    def _compute_distance_matrix(self, node_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Compute estimated distances for all node pairs.

        Args:
            node_pairs: List of (source_node, target_node) pairs to compute distances for

        Returns:
            Dictionary mapping (source_node, target_node) -> estimated distance
        """
        distance_matrix = {}
        for source_node, target_node in node_pairs:
            est_dist = self.estimate_node_distance(source_node, target_node)
            distance_matrix[(source_node, target_node)] = est_dist
        return distance_matrix


    def _flatten_state(self, state: ObsType) -> np.ndarray:
        """Flatten state to array."""
        if hasattr(state, "nodes"):
            return state.nodes.flatten().astype(np.float32) #[1:3]
        return np.array(state).flatten().astype(np.float32)

    def save(self, path: str) -> None:
        """Save distance heuristic to disk."""
        os.makedirs(path, exist_ok=True)

        # Save networks
        torch.save(self.s_encoder.state_dict(), f"{path}/s_encoder.pt")
        torch.save(self.g_encoder.state_dict(), f"{path}/g_encoder.pt")
        torch.save(self.c_net.state_dict(), f"{path}/c_net.pt")
        torch.save(self.d_net.state_dict(), f"{path}/d_net.pt")

        # Save config
        with open(f"{path}/config.pkl", "wb") as f:
            pickle.dump(self.config, f)

        print(f"Distance heuristic V5 saved to {path}")

    def load(self, path: str, state_dim: int, num_nodes: int) -> None:
        """Load distance heuristic from disk."""
        # Load config
        with open(f"{path}/config.pkl", "rb") as f:
            self.config = pickle.load(f)

        self.num_nodes = num_nodes

        # Recreate networks
        self.s_encoder = StateEncoder(
            state_dim=state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims or [64, 64],
        ).to(self.device)

        self.g_encoder = GoalEncoder(
            num_nodes=self.num_nodes,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.c_net = CostNet(num_nodes=self.num_nodes).to(self.device)

        self.d_net = MRN(
            s_encoder=self.s_encoder,
            g_encoder=self.g_encoder,
            sym_dim=64,
            asym_dim=16,
        ).to(self.device)

        # Load weights
        self.s_encoder.load_state_dict(torch.load(f"{path}/s_encoder.pt", map_location=self.device))
        self.s_encoder.eval()

        self.g_encoder.load_state_dict(torch.load(f"{path}/g_encoder.pt", map_location=self.device))
        self.g_encoder.eval()

        self.c_net.load_state_dict(torch.load(f"{path}/c_net.pt", map_location=self.device))
        self.c_net.eval()

        self.d_net.load_state_dict(torch.load(f"{path}/d_net.pt", map_location=self.device))
        self.d_net.eval()

        print(f"Distance heuristic V5 loaded from {path}")
