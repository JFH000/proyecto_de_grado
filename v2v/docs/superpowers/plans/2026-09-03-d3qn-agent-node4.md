# D3QN Agent Per V2V Pair, No Federation (Checklist Node 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each of the K=5 V2V-pair agents in `V2VEnv` its own Dueling Double DQN (D3QN) with prioritized experience replay, train them purely on their own transitions (no weight sharing, no FedAvg — DTE), and produce evidence that each one converges.

**Architecture:** A new `agents/` package (sibling of `v2v_env/`) holds three focused modules — `network.py` (the dueling Q-network), `replay_buffer.py` (prioritized experience replay), `d3qn_agent.py` (the agent: epsilon-greedy action selection, double-DQN learning step, target-network sync). A root-level script, `train_node4.py`, wires up 5 independent `D3QNAgent` instances around the existing `V2VEnv` and runs the DTE training loop.

**Tech Stack:** PyTorch (new dependency, `cp314-win_amd64` wheel confirmed available for this repo's `requires-python = ">=3.14"`), NumPy, the existing `v2v_env` package, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-03-d3qn-agent-node4-design.md`

## Global Constraints

- Python `>=3.14` (already the project's `requires-python`); install torch via `uv add torch` (confirmed: torch 2.14.0 ships a `cp314-win_amd64` wheel).
- No `done`/`(1-done)` term anywhere: this env's MDP is infinite-horizon, `max_steps` is only a training-loop cutoff, so the Bellman target always bootstraps through truncation, and the replay buffer never stores a `done` flag.
- Replay buffer: plain NumPy arrays + cumulative-distribution sampling, no sum-tree (YAGNI at capacity 10,000).
- Paper-sourced hyperparameters (Li et al. 2022, Table II) are fixed: replay buffer capacity 10,000, minibatch 128, Adam optimizer, target-network hard update every 400 steps.
- Hyperparameters the paper does not give (flagged, tunable via constructor/CLI defaults): `lr=1e-3`, `gamma=0.99`, `hidden_dim=128`, epsilon 1.0→0.05 linear over the first 60% of training steps, PER `alpha=0.6`, `beta_start=0.4`, `eps=1e-5`, default `--episodes=300`.
- Node 4 produces a CSV + a printed per-agent convergence verdict — no plot, no FedAvg, no Greedy baseline (nodes 5/6's job).
- `runs/` (training output) must be added to the monorepo `.gitignore` (root `C:\Users\juanf\proyecto_de_grado`) as `v2v/runs/` — never commit generated CSVs.
- Every new file follows this repo's existing test style: small, focused, seeded-RNG-for-determinism, plain `assert`, no comments beyond a non-obvious "why" (see `v2v_env/reward.py`, `tests/test_physics.py`).

---

### Task 1: Add PyTorch dependency and the `DuelingQNetwork`

**Files:**
- Modify: `pyproject.toml` (via `uv add torch`, not by hand)
- Create: `agents/__init__.py`
- Create: `agents/network.py`
- Test: `tests/test_network.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `agents.network.DuelingQNetwork(state_dim: int, action_dim: int, hidden_dim: int = 128)`, a `torch.nn.Module` with `forward_streams(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` (returns `(value, advantage)`, shapes `(batch, 1)` and `(batch, action_dim)`) and `forward(state: torch.Tensor) -> torch.Tensor` (returns Q-values, shape `(batch, action_dim)`).

- [ ] **Step 1: Add the torch dependency**

Run: `uv add torch`
Expected: `pyproject.toml` and `uv.lock` gain a `torch` entry; the command installs it into `.venv`.

Verify with: `.venv/Scripts/python.exe -c "import torch; print(torch.__version__)"`
Expected: prints a version string (e.g. `2.14.0`), no `ModuleNotFoundError`.

- [ ] **Step 2: Create the `agents` package**

Create `agents/__init__.py` (empty file, matching `v2v_env/__init__.py`).

- [ ] **Step 3: Write the failing tests**

Create `tests/test_network.py`:

```python
import torch

from agents.network import DuelingQNetwork


def test_forward_returns_q_values_shaped_batch_by_action_dim():
    net = DuelingQNetwork(state_dim=6, action_dim=4)
    state = torch.zeros(3, 6)
    q = net(state)
    assert q.shape == (3, 4)


def test_forward_matches_value_plus_centered_advantage():
    net = DuelingQNetwork(state_dim=6, action_dim=4)
    state = torch.randn(5, 6)
    value, advantage = net.forward_streams(state)
    q = net(state)
    expected = value + (advantage - advantage.mean(dim=-1, keepdim=True))
    assert torch.allclose(q, expected)


def test_q_mean_over_actions_equals_value():
    net = DuelingQNetwork(state_dim=6, action_dim=4)
    state = torch.randn(2, 6)
    value, _ = net.forward_streams(state)
    q = net(state)
    assert torch.allclose(q.mean(dim=-1, keepdim=True), value, atol=1e-5)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.network'` (or similar import error).

- [ ] **Step 5: Implement `DuelingQNetwork`**

Create `agents/network.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_network.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add v2v/pyproject.toml v2v/uv.lock v2v/agents/__init__.py v2v/agents/network.py v2v/tests/test_network.py
git commit -m "feat(agents): add DuelingQNetwork and torch dependency"
```

---

### Task 2: `PrioritizedReplayBuffer`

**Files:**
- Create: `agents/replay_buffer.py`
- Test: `tests/test_replay_buffer.py`

**Interfaces:**
- Consumes: nothing (only NumPy).
- Produces: `agents.replay_buffer.PrioritizedReplayBuffer(capacity: int, state_dim: int, alpha: float = 0.6, eps: float = 1e-5, rng: np.random.Generator | None = None)` with:
  - `add(state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None`
  - `sample(batch_size: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(states, actions, rewards, next_states, weights, indices)`
  - `update_priorities(indices: np.ndarray, td_errors: np.ndarray) -> None`
  - `__len__() -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replay_buffer.py`:

```python
import numpy as np

from agents.replay_buffer import PrioritizedReplayBuffer


def test_len_grows_up_to_capacity_then_stays_there():
    buf = PrioritizedReplayBuffer(capacity=3, state_dim=2)
    for i in range(5):
        buf.add(np.array([float(i), 0.0]), action=0, reward=0.0, next_state=np.zeros(2))
    assert len(buf) == 3


def test_add_past_capacity_overwrites_oldest_entry():
    buf = PrioritizedReplayBuffer(capacity=3, state_dim=2)
    for i in range(4):  # index 0's original entry (state=[0,0]) gets overwritten on the 4th add
        buf.add(np.array([float(i), 0.0]), action=i, reward=float(i), next_state=np.zeros(2))
    states, *_ = buf.sample(batch_size=3, beta=0.5)
    assert not np.any(np.all(states == np.array([0.0, 0.0]), axis=1))


def test_sample_returns_requested_batch_size_with_correct_shapes():
    buf = PrioritizedReplayBuffer(capacity=10, state_dim=4, rng=np.random.default_rng(0))
    for i in range(10):
        buf.add(np.full(4, float(i)), action=i % 2, reward=float(i), next_state=np.full(4, float(i) + 1))
    states, actions, rewards, next_states, weights, indices = buf.sample(batch_size=6, beta=0.5)
    assert states.shape == (6, 4)
    assert next_states.shape == (6, 4)
    assert actions.shape == (6,)
    assert rewards.shape == (6,)
    assert weights.shape == (6,)
    assert indices.shape == (6,)


def test_higher_priority_transition_is_sampled_more_often():
    buf = PrioritizedReplayBuffer(capacity=2, state_dim=1, alpha=1.0, rng=np.random.default_rng(0))
    buf.add(np.array([0.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.add(np.array([1.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.update_priorities(np.array([0, 1]), td_errors=np.array([0.001, 100.0]))

    counts = {0: 0, 1: 0}
    for _ in range(200):
        _, _, _, _, _, indices = buf.sample(batch_size=1, beta=0.0)
        counts[int(indices[0])] += 1
    assert counts[1] > counts[0] * 5


def test_importance_weights_match_formula_at_beta_one():
    buf = PrioritizedReplayBuffer(capacity=2, state_dim=1, alpha=1.0, eps=0.0, rng=np.random.default_rng(0))
    buf.add(np.array([0.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.add(np.array([1.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.update_priorities(np.array([0, 1]), td_errors=np.array([1.0, 3.0]))
    # priorities = [1, 3] (alpha=1), probs = [0.25, 0.75], N=2
    # raw weights (N*P)^-1 = [1/(2*0.25), 1/(2*0.75)] = [2.0, 2/3], normalized by max -> [1.0, 1/3]
    _, _, _, _, weights, indices = buf.sample(batch_size=2, beta=1.0)
    expected = {0: 1.0, 1: 1.0 / 3.0}
    for idx, w in zip(indices, weights):
        assert np.isclose(w, expected[int(idx)], atol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_replay_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.replay_buffer'`.

- [ ] **Step 3: Implement `PrioritizedReplayBuffer`**

Create `agents/replay_buffer.py`:

```python
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

        weights = (self._size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_replay_buffer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add v2v/agents/replay_buffer.py v2v/tests/test_replay_buffer.py
git commit -m "feat(agents): add PrioritizedReplayBuffer"
```

---

### Task 3: `D3QNAgent` and `double_dqn_targets`

**Files:**
- Create: `agents/d3qn_agent.py`
- Test: `tests/test_d3qn_agent.py`

**Interfaces:**
- Consumes: `agents.network.DuelingQNetwork` (Task 1: `forward`, `forward_streams`, constructor `(state_dim, action_dim, hidden_dim=128)`); `agents.replay_buffer.PrioritizedReplayBuffer` (Task 2: `add`, `sample`, `update_priorities`, `__len__`, constructor `(capacity, state_dim, alpha, eps, rng)`).
- Produces:
  - `agents.d3qn_agent.double_dqn_targets(online: DuelingQNetwork, target: DuelingQNetwork, rewards: torch.Tensor, next_states: torch.Tensor, gamma: float) -> torch.Tensor`
  - `agents.d3qn_agent.D3QNAgent(state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99, buffer_capacity=10_000, batch_size=128, target_update_every=400, per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100_000, per_eps=1e-5, seed=None)` with `select_action(obs: np.ndarray, epsilon: float) -> int`, `store_transition(obs, action, reward, next_obs) -> None`, `learn() -> float | None`, `update_target() -> None`, and public attributes `online: DuelingQNetwork`, `target: DuelingQNetwork`, `buffer: PrioritizedReplayBuffer`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_d3qn_agent.py`:

```python
import math

import numpy as np
import torch

from agents.d3qn_agent import D3QNAgent, double_dqn_targets
from agents.network import DuelingQNetwork


def _force_constant_q_values(net: DuelingQNetwork, advantage_bias, value_bias: float = 0.0) -> None:
    with torch.no_grad():
        net.value_head.weight.zero_()
        net.value_head.bias.fill_(value_bias)
        net.advantage_head.weight.zero_()
        net.advantage_head.bias.copy_(torch.tensor(advantage_bias, dtype=torch.float32))


def test_select_action_greedy_picks_argmax_when_epsilon_zero():
    agent = D3QNAgent(state_dim=2, action_dim=3, seed=0)
    _force_constant_q_values(agent.online, advantage_bias=[0.0, 5.0, -1.0])
    obs = np.zeros(2, dtype=np.float64)
    assert agent.select_action(obs, epsilon=0.0) == 1


def test_select_action_epsilon_one_explores_multiple_actions():
    agent = D3QNAgent(state_dim=2, action_dim=4, seed=1)
    obs = np.zeros(2, dtype=np.float64)
    actions = {agent.select_action(obs, epsilon=1.0) for _ in range(200)}
    assert len(actions) > 1


def test_double_dqn_targets_matches_manual_formula():
    online = DuelingQNetwork(state_dim=2, action_dim=3)
    target = DuelingQNetwork(state_dim=2, action_dim=3)
    _force_constant_q_values(online, advantage_bias=[0.0, 5.0, -1.0])  # argmax action = 1, everywhere
    _force_constant_q_values(target, advantage_bias=[2.0, 3.0, 0.0])

    rewards = torch.tensor([1.0, -2.0])
    next_states = torch.zeros(2, 2)
    result = double_dqn_targets(online, target, rewards, next_states, gamma=0.9)

    target_advantage = torch.tensor([2.0, 3.0, 0.0])
    target_q_action_1 = (target_advantage - target_advantage.mean())[1]
    expected = rewards + 0.9 * target_q_action_1
    assert torch.allclose(result, expected, atol=1e-5)


def test_update_target_copies_online_parameters_exactly():
    agent = D3QNAgent(state_dim=2, action_dim=2, seed=2)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(1.0)

    agent.update_target()

    for p_online, p_target in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(p_online, p_target)


def test_learn_returns_none_below_batch_size():
    agent = D3QNAgent(state_dim=2, action_dim=2, batch_size=4, seed=3)
    agent.store_transition(np.zeros(2), 0, 1.0, np.zeros(2))
    assert agent.learn() is None


def test_learn_returns_finite_loss_once_batch_size_reached():
    agent = D3QNAgent(state_dim=2, action_dim=2, batch_size=4, seed=4)
    for i in range(4):
        agent.store_transition(np.array([float(i), 0.0]), i % 2, 1.0, np.array([float(i) + 1, 0.0]))
    loss = agent.learn()
    assert loss is not None
    assert math.isfinite(loss)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_d3qn_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.d3qn_agent'`.

- [ ] **Step 3: Implement `D3QNAgent` and `double_dqn_targets`**

Create `agents/d3qn_agent.py`:

```python
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
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self._learn_steps = 0

        if seed is not None:
            action_seed, buffer_seed = np.random.SeedSequence(seed).spawn(2)
            self.rng = np.random.default_rng(action_seed)
            buffer_rng = np.random.default_rng(buffer_seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.default_rng()
            buffer_rng = np.random.default_rng()

        self.online = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        self.target = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_capacity, state_dim, alpha=per_alpha, eps=per_eps, rng=buffer_rng
        )

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
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

        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32)
        weights_t = torch.as_tensor(weights, dtype=torch.float32)

        targets = double_dqn_targets(self.online, self.target, rewards_t, next_states_t, self.gamma)
        current_q = self.online(states_t).gather(1, actions_t).squeeze(-1)

        td_errors = targets - current_q
        loss = (weights_t * F.smooth_l1_loss(current_q, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().numpy())

        self._learn_steps += 1
        if self._learn_steps % self.target_update_every == 0:
            self.update_target()

        return loss.item()

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_d3qn_agent.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add v2v/agents/d3qn_agent.py v2v/tests/test_d3qn_agent.py
git commit -m "feat(agents): add D3QNAgent with double-DQN target and PER"
```

---

### Task 4: `train_node4.py` training script

**Files:**
- Create: `train_node4.py`
- Test: `tests/test_train_node4.py`
- Modify: `C:\Users\juanf\proyecto_de_grado\.gitignore` (add `v2v/runs/`)

**Interfaces:**
- Consumes: `agents.d3qn_agent.D3QNAgent` (Task 3); `v2v_env.env.V2VEnv` and `v2v_env.params.V2VEnvParams` (existing, `env.py`/`params.py`).
- Produces (importable from `train_node4`, used by Task 5's manual run): `epsilon_schedule(step, total_steps, start=1.0, end=0.05, decay_fraction=0.6) -> float`; `make_agents(params: V2VEnvParams, seed: int, per_beta_frames: int) -> dict[str, D3QNAgent]`; `run_training(params: V2VEnvParams, episodes: int, seed: int, log_every: int = 10) -> list[tuple[int, str, float]]`; `write_records_csv(records, output_path: Path) -> None`; `print_convergence_verdict(records, episodes: int) -> None`; `main() -> None`.

- [ ] **Step 1: Add `v2v/runs/` to the monorepo `.gitignore`**

Append to `C:\Users\juanf\proyecto_de_grado\.gitignore`:

```
v2v/runs/
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_train_node4.py`:

```python
import csv
import math

from v2v_env.params import V2VEnvParams

from train_node4 import (
    epsilon_schedule,
    print_convergence_verdict,
    run_training,
    write_records_csv,
)


def make_tiny_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2, max_steps=5)


def test_epsilon_schedule_starts_high_and_decays_to_floor():
    assert epsilon_schedule(0, total_steps=100) == 1.0
    assert epsilon_schedule(100, total_steps=100) == 0.05


def test_run_training_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records = run_training(params, episodes=3, seed=0, log_every=100)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {ep for ep, _, _ in records} == {0, 1, 2}


def test_run_training_rewards_are_finite():
    params = make_tiny_params()
    records = run_training(params, episodes=2, seed=1, log_every=100)
    assert all(math.isfinite(r) for _, _, r in records)


def test_write_records_csv_round_trips(tmp_path):
    records = [(0, "v2v_0", 1.5), (0, "v2v_1", -2.0)]
    output_path = tmp_path / "rewards.csv"
    write_records_csv(records, output_path)
    with output_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "agent", "reward"]
    assert rows[1] == ["0", "v2v_0", "1.5"]


def test_print_convergence_verdict_mentions_every_agent(capsys):
    records = [(ep, "v2v_0", float(ep)) for ep in range(10)] + [(ep, "v2v_1", float(ep)) for ep in range(10)]
    print_convergence_verdict(records, episodes=10)
    captured = capsys.readouterr()
    assert "v2v_0" in captured.out
    assert "v2v_1" in captured.out
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_node4.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_node4'`.

- [ ] **Step 4: Implement `train_node4.py`**

Create `train_node4.py`:

```python
"""Train K independent D3QN agents on V2VEnv with no FedAvg (checklist node 4,
subtask 3). Each agent learns only from its own transitions -- DTE control
run, meant to converge before FedAvg (node 5) is layered on top.
"""

import argparse
import csv
from pathlib import Path

from agents.d3qn_agent import D3QNAgent
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def epsilon_schedule(
    step: int, total_steps: int, start: float = 1.0, end: float = 0.05, decay_fraction: float = 0.6
) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def make_agents(params: V2VEnvParams, seed: int, per_beta_frames: int) -> dict[str, D3QNAgent]:
    return {
        agent_id: D3QNAgent(
            state_dim=params.state_dim,
            action_dim=params.action_space_size,
            seed=seed + i,
            per_beta_frames=per_beta_frames,
        )
        for i, agent_id in enumerate(f"v2v_{i}" for i in range(params.num_v2v_pairs))
    }


def run_training(
    params: V2VEnvParams, episodes: int, seed: int, log_every: int = 10
) -> list[tuple[int, str, float]]:
    env = V2VEnv(params)
    total_steps = episodes * params.max_steps
    agents = make_agents(params, seed, per_beta_frames=total_steps)

    records: list[tuple[int, str, float]] = []
    recent: dict[str, list[float]] = {agent_id: [] for agent_id in env.possible_agents}
    global_step = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = {agent_id: 0.0 for agent_id in env.possible_agents}

        while env.agents:
            eps = epsilon_schedule(global_step, total_steps)
            actions = {a: agents[a].select_action(obs[a], eps) for a in env.agents}
            next_obs, rewards, _, _, _ = env.step(actions)
            for a in env.possible_agents:
                agents[a].store_transition(obs[a], actions[a], rewards[a], next_obs[a])
                agents[a].learn()
                episode_reward[a] += rewards[a]
            obs = next_obs
            global_step += 1

        for a, r in episode_reward.items():
            records.append((ep, a, r))
            recent[a].append(r)

        if (ep + 1) % log_every == 0:
            summary = ", ".join(f"{a}={sum(rs) / len(rs):.3f}" for a, rs in recent.items())
            print(f"Episode {ep + 1:>4}/{episodes} | {summary}")
            recent = {agent_id: [] for agent_id in env.possible_agents}

    return records


def write_records_csv(records: list[tuple[int, str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "agent", "reward"])
        writer.writerows(records)


def print_convergence_verdict(records: list[tuple[int, str, float]], episodes: int) -> None:
    by_agent: dict[str, list[float]] = {}
    for ep, agent_id, reward in sorted(records, key=lambda r: r[0]):
        by_agent.setdefault(agent_id, []).append(reward)

    window = max(1, episodes // 10)
    for agent_id, rewards in by_agent.items():
        first = sum(rewards[:window]) / window
        last = sum(rewards[-window:]) / window
        verdict = "improved" if last > first else "did not improve"
        print(f"agent {agent_id}: first-10%={first:.3f} -> last-10%={last:.3f} ({verdict})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/node4_no_fedavg_rewards.csv"))
    args = parser.parse_args()

    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=200)
    records = run_training(params, args.episodes, args.seed, args.log_every)

    write_records_csv(records, args.output)
    print(f"\nSaved: {args.output}")
    print_convergence_verdict(records, args.episodes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_node4.py -v`
Expected: 5 passed.

- [ ] **Step 6: Run the full test suite to confirm nothing else broke**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests passed (47 pre-existing + all new tests from Tasks 1-4).

- [ ] **Step 7: Commit**

```bash
git add v2v/train_node4.py v2v/tests/test_train_node4.py .gitignore
git commit -m "feat: add train_node4.py, the DTE-only training script for checklist node 4"
```

---

### Task 5: Run the node-4 training and confirm convergence (checklist subtask 3)

This is the actual empirical deliverable the checklist asks for — not more unit tests. No test file.

**Files:** none created or modified; runs the code from Tasks 1-4.

- [ ] **Step 1: Launch the training run in the background**

Run (from the `v2v` directory):

```bash
.venv/Scripts/python.exe train_node4.py --episodes 300 --seed 0 > runs/node4_train_log.txt 2>&1 &
```

Or, in this harness, use the Bash tool's `run_in_background: true` option on:
`.venv/Scripts/python.exe train_node4.py --episodes 300 --seed 0`

- [ ] **Step 2: Monitor until it finishes**

Poll the background job's output (or `runs/node4_train_log.txt`) until `main()` prints `Saved: runs/node4_no_fedavg_rewards.csv` followed by the 5 convergence-verdict lines (one per `v2v_0`..`v2v_4`).

- [ ] **Step 3: Inspect the result**

- Confirm `runs/node4_no_fedavg_rewards.csv` exists and has `episodes * 5 + 1` lines (header + one row per episode per agent).
- Read the 5 printed verdict lines. Report the actual `first-10%` / `last-10%` numbers to the user for every agent — do not round up to "converged" if any agent's says "did not improve"; report exactly what happened instead.
- If most or all agents show "did not improve", this is a hyperparameter-tuning problem, not a code-correctness one (Tasks 1-4's tests already pin down the formulas) — flag it to the user rather than silently re-running with different numbers.

- [ ] **Step 4: Report to the user**

Summarize: episodes run, per-agent first-10%/last-10% reward, whether each agent improved, and the CSV path. This closes checklist node 4's subtask 3 ("Entrenar sin FedAvg... y confirmar que cada uno converge").
