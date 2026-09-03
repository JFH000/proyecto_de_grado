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
