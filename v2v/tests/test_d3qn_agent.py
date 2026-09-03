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
