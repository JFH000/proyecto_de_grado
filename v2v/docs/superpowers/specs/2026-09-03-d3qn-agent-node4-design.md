# D3QN agent per V2V pair, no federation (checklist node 4)

**Status:** approved by user 2026-09-03, pending implementation plan.
**Source:** `checklists/checklist-implementacion-frl.html`, node 4 ("Agente D3QN por par V2V, sin federar todavía").
**Paper:** Li et al. (2022), `knowledge/summary_fedmarl_v2v_resource_allocation.md`.

## Goal

Each V2V pair (agent) trains a Dueling Double DQN (D3QN) with prioritized
experience replay purely on its own (state, action, reward, next-state)
tuples — no weight sharing, no FedAvg. This is DTE (Decentralized Training
and Execution). It runs inside the existing `V2VEnv` (node 3) with K=5
agents and must be shown to converge before FedAvg (node 5) is layered on
top, so that if something breaks later we know whether the bug is in the
env/agent or in the federation layer.

Out of scope (explicitly deferred to later nodes): FedAvg (node 5), the
Greedy baseline and the with/without-FedAvg comparison plot (node 6),
sweeping K.

## Repo layout

```
v2v/
  agents/                       # new package, sibling of v2v_env/
    __init__.py
    network.py                  # DuelingQNetwork
    replay_buffer.py            # PrioritizedReplayBuffer
    d3qn_agent.py                # D3QNAgent, double_dqn_targets()
  train_node4.py                 # training script for this node, root-level like main.py
  tests/
    test_network.py
    test_replay_buffer.py
    test_d3qn_agent.py
  runs/                          # git-ignored, training output
    node4_no_fedavg_rewards.csv
```

`pyproject.toml`: add `torch` to `[project.dependencies]` (confirmed a
`cp314-win_amd64` wheel exists for torch 2.14.0, matching this repo's
`requires-python = ">=3.14"`).

Monorepo `.gitignore` (root is `C:\Users\juanf\proyecto_de_grado`): add
`v2v/runs/`.

## Components

### `agents/network.py` — `DuelingQNetwork(nn.Module)`

- `__init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128)`
- Shared trunk: `Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim) -> ReLU`.
- Value head: `Linear(hidden_dim, 1)`.
- Advantage head: `Linear(hidden_dim, action_dim)`.
- `forward_streams(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`
  returns `(V, A)` (shapes `(batch, 1)` and `(batch, action_dim)`) from
  the two heads — exists so the dueling combination can be unit-tested
  directly against the streams that produced it.
- `forward(self, state: torch.Tensor) -> torch.Tensor` calls
  `forward_streams` and returns Q-values, shape `(batch, action_dim)`,
  computed as `Q = V + (A - A.mean(dim=-1, keepdim=True))`.

### `agents/replay_buffer.py` — `PrioritizedReplayBuffer`

Plain-array proportional-priority implementation (no sum-tree — at this
project's scale, capacity 10 000, O(n) sampling via cumulative
distribution is fast enough; YAGNI over the standard sum-tree
optimization used for much larger buffers).

- `__init__(self, capacity: int, state_dim: int, alpha: float = 0.6, eps: float = 1e-5)`
- `add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None`
  — circular overwrite once full; new entries get priority = current max
  priority in the buffer (or 1.0 if empty), so every new transition is
  sampled at least once before its priority is corrected. No `done`
  field: see the "no `(1-done)` term" decision below — this env's MDP
  has no true terminal state, so a transition tuple that carried a
  `done` flag nobody ever reads would just be dead data.
- `sample(self, batch_size: int, beta: float) -> tuple[states, actions, rewards, next_states, weights, indices]`
  — sampling probability `P(i) = p_i^alpha / sum(p^alpha)`; importance
  weight `w_i = (N * P(i))^(-beta)`, normalized so `max(w) == 1` in the
  batch (Schaul et al. 2016 convention).
- `update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None`
  — `p_i = |td_error_i| + eps`.
- `__len__(self) -> int`

### `agents/d3qn_agent.py`

`double_dqn_targets(online: DuelingQNetwork, target: DuelingQNetwork, rewards: torch.Tensor, next_states: torch.Tensor, gamma: float) -> torch.Tensor`
— standalone function so the target formula is unit-testable in
isolation: `next_actions = online(next_states).argmax(dim=-1)`;
`next_q = target(next_states).gather(1, next_actions)`;
`return rewards + gamma * next_q`.

Design decision (documented like `reward.py`'s deviation note): **no
`done`/`(1 - done)` term anywhere in the buffer or the target.** Li et
al.'s MDP (Sec. IV-A) is infinite-horizon; this env's `max_steps`
truncation exists only so the training loop has episode boundaries to
log against, not because the underlying process has a true terminal
state (`terminations` in `V2VEnv` is hardcoded `False`). Zeroing the
bootstrap at truncation would bias the target downward for no physical
reason, so the Bellman target always bootstraps through truncation —
and since it's never used, `done` is not stored in the buffer or passed
into `store_transition`/`learn` at all.

`D3QNAgent`:
- `__init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99, buffer_capacity=10_000, batch_size=128, target_update_every=400, per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100_000, per_eps=1e-5, seed=None)`
  — builds `online`/`target` `DuelingQNetwork`s (target starts as a copy
  of online), `Adam(online.parameters(), lr=lr)`, its own
  `PrioritizedReplayBuffer`, its own `np.random.default_rng(seed)` for
  epsilon-greedy exploration and its own `torch.manual_seed` scope for
  weight init — every agent's randomness is independent, per DTE.
- `select_action(self, obs: np.ndarray, epsilon: float) -> int` — with
  probability `epsilon`, a uniform random action from
  `self.rng.integers(action_dim)`; otherwise `argmax` of
  `online(obs)` under `torch.no_grad()`.
- `store_transition(self, obs, action, reward, next_obs) -> None`
  — forwards to the buffer.
- `learn(self) -> float | None` — returns `None` and does nothing while
  `len(buffer) < batch_size`; otherwise samples a batch, computes
  `double_dqn_targets`, current Q via
  `online(states).gather(1, actions)`, `SmoothL1Loss` (Huber) weighted
  by the batch's IS weights, backprop + `optimizer.step()`,
  `buffer.update_priorities` from the (detached) TD errors, anneals
  beta as `min(1.0, per_beta_start + (1 - per_beta_start) * learn_steps / per_beta_frames)`,
  increments an internal learn-step counter, calls `update_target()`
  every `target_update_every` learn steps, and returns the loss value.
- `update_target(self) -> None` — hard copy:
  `self.target.load_state_dict(self.online.state_dict())`.

### `train_node4.py`

CLI (argparse): `--episodes` (default 300), `--seed` (default 0),
`--log-every` (default 10), `--output` (default
`runs/node4_no_fedavg_rewards.csv`).

```
params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=200)
env = V2VEnv(params)
agents = {
    agent_id: D3QNAgent(state_dim=params.state_dim, action_dim=params.action_space_size, seed=seed + i,
                         per_beta_frames=episodes * params.max_steps)
    for i, agent_id in enumerate(env.possible_agents)
}
```

Loop, `global_step = 0`, `records = []`:

```
for ep in range(episodes):
    obs, _ = env.reset(seed=seed + ep)          # fresh topology every episode
    episode_reward = {a: 0.0 for a in env.possible_agents}
    while env.agents:
        eps = epsilon_schedule(global_step, total_steps=episodes * params.max_steps)
        actions = {a: agents[a].select_action(obs[a], eps) for a in env.agents}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        for a in env.possible_agents:
            agents[a].store_transition(obs[a], actions[a], rewards[a], next_obs[a])
            agents[a].learn()
            episode_reward[a] += rewards[a]
        obs = next_obs
        global_step += 1
    records.extend((ep, a, r) for a, r in episode_reward.items())
    if ep % log_every == 0:
        print moving-average (last `log_every` episodes) reward per agent
```

`epsilon_schedule(step, total_steps)`: linear from 1.0 to 0.05 over the
first 60% of `total_steps`, flat at 0.05 after.

After the loop: write `records` to the output CSV (`episode,agent,reward`
header). Then print a convergence verdict per agent: mean reward of the
first 10% of episodes vs. the last 10%, printed as
`agent v2v_0: first-10%=... -> last-10%=... (improved / did not improve)`.
This is a printed observation, not a `pytest`/`assert`-gated check —
empirical RL convergence isn't something to hard-assert on a fixed
sample.

## Testing (TDD, mirrors existing `tests/test_*.py` style: small, seeded,
plain assertions, no test framework beyond `pytest`)

- `tests/test_network.py`
  - forward pass on a batch returns shape `(batch, action_dim)`.
  - Q decomposition property: call `forward_streams` and `forward` on
    the same input, assert `Q.mean(dim=-1)` equals `V.squeeze(-1)`
    (checks the `A - A.mean()` term integrates to zero over actions,
    i.e. the combination formula is implemented correctly).
- `tests/test_replay_buffer.py`
  - `add` past `capacity` overwrites oldest entries (`__len__` caps at
    capacity; a known state at index 0 is gone after `capacity + 1` adds).
  - `sample` returns arrays of the requested `batch_size` with correct
    shapes/dtypes.
  - priority bias: add two transitions, give one a much larger priority
    via `update_priorities`, sample many times with `beta=0`, assert the
    high-priority index is drawn noticeably more often (statistical, seeded
    RNG, generous threshold to avoid flakiness).
  - IS weight formula: a hand-computed case with a small buffer and known
    priorities checks `sample(..., beta=1.0)` returns weights matching
    `(N * P(i))**-1` normalized to max 1.
- `tests/test_d3qn_agent.py`
  - `select_action` with `epsilon=0.0` always returns the argmax action
    for a network whose weights are set so one action's Q is clearly
    highest.
  - `select_action` with `epsilon=1.0` over many draws (seeded) covers
    more than one action (sanity check it's not silently always
    greedy).
  - `double_dqn_targets` matches a manually computed value for a tiny
    hand-built `DuelingQNetwork` pair (fixed weights) and a small batch
    of rewards/next_states — verifies the "argmax from online, value from
    target, no `(1-done)` term" formula directly.
  - `update_target` copies `online`'s parameters into `target` exactly
    (`torch.equal` on every parameter tensor) after a call, even after
    the online net's weights have since diverged via a manual
    optimizer step.
  - `learn()` returns `None` (and does not touch the optimizer/buffer
    priorities) while the buffer has fewer than `batch_size` transitions.

No test trains to convergence — that's what running `train_node4.py`
(subtask 3) demonstrates empirically via the CSV + printed verdict.

## Open parameters explicitly not given by the paper (flagged, matching
`params.py`'s convention)

`lr=1e-3`, `gamma=0.99`, epsilon schedule bounds/shape, PER
`alpha=0.6`/`beta_start=0.4`/`eps=1e-5`, `hidden_dim=128`, episode count
default (300). All exposed as constructor/CLI parameters so they can be
retuned without touching the implementation.
