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


def test_hidden_sizes_set_the_width_and_depth_of_the_trunk():
    net = DuelingQNetwork(state_dim=6, action_dim=4, hidden_sizes=(500, 250, 120))
    linears = [layer for layer in net.trunk if isinstance(layer, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(6, 500), (500, 250), (250, 120)]
    assert net.value_head.in_features == 120 and net.advantage_head.in_features == 120
    assert net(torch.zeros(2, 6)).shape == (2, 4)


def test_default_trunk_is_two_layers_of_hidden_dim():
    net = DuelingQNetwork(state_dim=6, action_dim=4, hidden_dim=32)
    linears = [layer for layer in net.trunk if isinstance(layer, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(6, 32), (32, 32)]
