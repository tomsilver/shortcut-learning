"""Shared neural network modules for contrastive heuristics (CRL v2, CMD v2).

All actor, encoder, and distance-network classes live here so that
heuristic_crl_v2 and heuristic_cmd_v2 can share them without duplication.
Any heuristic that exposes ``sa_encoder`` / ``g_encoder`` attributes (with the
interfaces defined here) is automatically compatible with
``get_latent_embeddings()`` in the pipeline.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple feedforward MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Encoders (shared by CRL v2 and CMD v2)
# ---------------------------------------------------------------------------


class StateActionEncoder(nn.Module):
    """Encodes (state, action) pairs to latent embeddings.

    Takes state and action as *separate* tensors (concatenated internally),
    so downstream code can call ``sa_encoder(states, actions)``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode (state, action) pairs.

        Args:
            states:  (batch, state_dim)
            actions: (batch, action_dim)

        Returns:
            (batch, latent_dim)
        """
        return self.network(torch.cat([states, actions], dim=-1))


class GoalEncoder(nn.Module):
    """Encodes goal atom vectors (multi-hot) to latent embeddings."""

    def __init__(
        self,
        atom_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.atom_dim = atom_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32]

        layers = []
        prev_dim = atom_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, atom_vectors: torch.Tensor) -> torch.Tensor:
        """Encode atom vectors.

        Args:
            atom_vectors: (batch, atom_dim) multi-hot

        Returns:
            (batch, latent_dim)
        """
        return self.network(atom_vectors)


# ---------------------------------------------------------------------------
# CMD v2 distance / cost networks
# ---------------------------------------------------------------------------


class MRN(nn.Module):
    """Metric Residual Network for quasi-metric distances.

    Combines symmetric and asymmetric components to learn d(sa, g) that
    respects the triangle inequality.  Returns *negative* distance (energy).
    """

    def __init__(
        self,
        sa_dim: int,
        g_dim: int,
        sym_dim: int = 64,
        asym_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.sa_sym = MLP(sa_dim, sym_dim, hidden_dims)
        self.g_sym = MLP(g_dim, sym_dim, hidden_dims)
        self.sa_asym = MLP(sa_dim, asym_dim, hidden_dims)
        self.g_asym = MLP(g_dim, asym_dim, hidden_dims)

    def forward(self, sa: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Compute negative distance -d(sa, g).

        Args:
            sa: (batch, sa_dim)
            g:  (batch, g_dim)

        Returns:
            (batch, 1) negative distances
        """
        sa_enc = self.sa_sym(sa)
        g_enc = self.g_sym(g)
        dist_sym = (sa_enc - g_enc).pow(2).mean(-1, keepdim=True)

        sa_asym = self.sa_asym(sa)
        g_asym = self.g_asym(g)
        res = F.relu(sa_asym - g_asym)
        dist_asym = (F.softmax(res, -1) * res).sum(-1, keepdim=True)

        return -(dist_sym + dist_asym)


class CostNet(nn.Module):
    """Maps a goal encoding to a scalar cost c(g)."""

    def __init__(self, g_dim: int):
        super().__init__()
        self.scalar_net = nn.Sequential(
            nn.Linear(g_dim, g_dim // 2),
            nn.ReLU(),
            nn.Linear(g_dim // 2, g_dim // 4),
            nn.ReLU(),
            nn.Linear(g_dim // 4, 1),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.scalar_net(g)


# ---------------------------------------------------------------------------
# Actors (shared by CRL v2 and CMD v2)
# ---------------------------------------------------------------------------


class DiscreteActor(nn.Module):
    """Actor for discrete action spaces (categorical over actions)."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        atom_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.atom_dim = atom_dim

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = state_dim + atom_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(
        self, states: torch.Tensor, goal_atom_vectors: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([states, goal_atom_vectors], dim=-1))

    def sample(
        self, states: torch.Tensor, goal_atom_vectors: torch.Tensor
    ) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=self.forward(states, goal_atom_vectors))
        return dist.sample()

    def get_log_prob(
        self,
        states: torch.Tensor,
        goal_atom_vectors: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=self.forward(states, goal_atom_vectors))
        return dist.log_prob(actions)


class ContinuousActor(nn.Module):
    """Actor for continuous action spaces (squashed Gaussian)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_dim: int,
        hidden_dims: list[int] | None = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_dim = atom_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = state_dim + atom_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self, states: torch.Tensor, goal_atom_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(torch.cat([states, goal_atom_vectors], dim=-1))
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        states: torch.Tensor,
        goal_atom_vectors: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self.forward(states, goal_atom_vectors)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        x_t = torch.distributions.Normal(mean, std).rsample()
        action = torch.tanh(x_t)
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)


class DistributionalQNetwork(nn.Module):
    """Q(state, action, goal_atoms) → probability distribution over distance bins.

    Based on SoRB (Eysenbach et al., 2019).  Outputs ``num_bins`` logits where
    bin ``i`` represents a distance of ``i * max_bin / (num_bins - 1)`` steps.
    The final bin is a catch-all for distances ≥ ``max_bin``.

    The distributional Bellman update (γ=1, r=-1) is a right-shift:
    - Goal reached  → all probability mass in bin 0
    - Otherwise     → mass at bin i moves to bin i+1; catch-all absorbs overflow
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_dim: int,
        num_bins: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        if hidden_dims is None:
            hidden_dims = [256, 256]
        input_dim = state_dim + action_dim + atom_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, num_bins) logits."""
        return self.net(torch.cat([states, actions, goals], dim=-1))

    def get_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, num_bins) probabilities via softmax."""
        return F.softmax(self.forward(states, actions, goals), dim=-1)


class ResidualContinuousActor(nn.Module):
    """Residual actor: output = base_action + pi(state, goal, base_action).

    pi fuses a state-goal branch (MLP) with a base-action branch (MLP), concatenates
    both, passes through a combine layer, then outputs a tanh-squashed residual.
    The total action is base_action + residual where residual ∈ (-1, 1).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_dim: int,
        hidden_dims: list[int] | None = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_dim = atom_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if hidden_dims is None:
            hidden_dims = [64, 64]

        branch_dim = hidden_dims[-1]

        # Branch 1: (state, goal) → branch_dim
        self.sg_trunk = MLP(state_dim + atom_dim, branch_dim, hidden_dims[:-1])

        # Branch 2: base_action → branch_dim // 2
        self.base_trunk = nn.Sequential(
            nn.Linear(action_dim, branch_dim // 2),
            nn.ReLU(),
        )

        # Combine branches and produce residual mean/log_std
        combined_dim = branch_dim + branch_dim // 2
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, branch_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(branch_dim, action_dim)
        self.log_std_head = nn.Linear(branch_dim, action_dim)

    def _compute_params(
        self,
        states: torch.Tensor,
        goal_atom_vectors: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sg_features = self.sg_trunk(torch.cat([states, goal_atom_vectors], dim=-1))
        base_features = self.base_trunk(base_actions)
        combined = self.combine(torch.cat([sg_features, base_features], dim=-1))
        mean = self.mean_head(combined)
        log_std = torch.clamp(self.log_std_head(combined), self.log_std_min, self.log_std_max)
        return mean, log_std

    def forward(
        self, states: torch.Tensor, goal_atom_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) with zero base_actions, for diagnostics/compatibility."""
        base_actions = torch.zeros(states.shape[0], self.action_dim, device=states.device)
        return self._compute_params(states, goal_atom_vectors, base_actions)

    def sample(
        self,
        states: torch.Tensor,
        goal_atom_vectors: torch.Tensor,
        base_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if base_actions is None:
            base_actions = torch.zeros(states.shape[0], self.action_dim, device=states.device)

        mean, log_std = self._compute_params(states, goal_atom_vectors, base_actions)

        if deterministic:
            return torch.tanh(mean), None

        std = log_std.exp()
        x_t = torch.distributions.Normal(mean, std).rsample()
        residual = torch.tanh(x_t)
        # print("THE REAL RESIDUAL")
        action = base_actions + residual
        # action = residual

        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(1 - residual.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)
