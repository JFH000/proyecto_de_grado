from collections.abc import Sequence

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """hidden_sizes: width of each hidden layer of the shared trunk (Li et al. 2022, Table III,
    use 500, 250, 120); by default two layers of hidden_dim."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 128, hidden_sizes: Sequence[int] | None = None
    ):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes) if hidden_sizes is not None else (hidden_dim, hidden_dim)
        layers = []
        for n_in, n_out in zip((state_dim, *self.hidden_sizes[:-1]), self.hidden_sizes):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)
        self.advantage_head = nn.Linear(self.hidden_sizes[-1], action_dim)

    def forward_streams(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        return self.value_head(features), self.advantage_head(features)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value, advantage = self.forward_streams(state)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
