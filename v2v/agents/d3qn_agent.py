import numpy as np
import torch
import torch.nn.functional as F

from agents.network import DuelingQNetwork
from agents.replay_buffer import PrioritizedReplayBuffer


def double_dqn_targets(
    online: DuelingQNetwork,
    target: DuelingQNetwork,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=-1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions).squeeze(-1)
    return rewards + gamma * next_q


class D3QNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 10_000,
        batch_size: int = 128,
        target_update_every: int = 400,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-5,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self._learn_steps = 0
        self.device = torch.device(device)

        if seed is not None:
            action_seed, buffer_seed = np.random.SeedSequence(seed).spawn(2)
            self.rng = np.random.default_rng(action_seed)
            buffer_rng = np.random.default_rng(buffer_seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.default_rng()
            buffer_rng = np.random.default_rng()

        self.online = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_capacity, state_dim, alpha=per_alpha, eps=per_eps, rng=buffer_rng
        )

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            state = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(state)
        return int(q_values.argmax(dim=-1).item())

    def store_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray) -> None:
        self.buffer.add(obs, action, reward, next_obs)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        beta = min(
            1.0,
            self.per_beta_start
            + (1 - self.per_beta_start) * self._learn_steps / self.per_beta_frames,
        )
        states, actions, rewards, next_states, weights, indices = self.buffer.sample(self.batch_size, beta)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        targets = double_dqn_targets(self.online, self.target, rewards_t, next_states_t, self.gamma)
        current_q = self.online(states_t).gather(1, actions_t).squeeze(-1)

        td_errors = targets - current_q
        loss = (weights_t * F.smooth_l1_loss(current_q, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        self._learn_steps += 1
        if self._learn_steps % self.target_update_every == 0:
            self.update_target()

        return loss.item()

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
