import numpy as np


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        eps: float = 1e-5,
        rng: np.random.Generator | None = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.rng = rng if rng is not None else np.random.default_rng()

        self.states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.priorities = np.zeros(capacity, dtype=np.float64)

        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        idx = self._pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.priorities[idx] = self.priorities[: self._size].max() if self._size > 0 else 1.0

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        scaled_priorities = self.priorities[: self._size] ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()
        indices = self.rng.choice(self._size, size=batch_size, p=probs)

        all_weights = (self._size * probs) ** (-beta)
        weights = all_weights[indices] / all_weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            weights,
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors) + self.eps
