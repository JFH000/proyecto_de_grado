import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, action_dim)

    def forward_streams(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        return self.value_head(features), self.advantage_head(features)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value, advantage = self.forward_streams(state)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
