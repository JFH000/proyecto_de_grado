import math

import pytest
import torch

from agents.d3qn_agent import D3QNAgent
from agents.federated import (
    fedavg_node_loads,
    fedavg_round,
    model_num_bytes,
    weight_dispersion,
    weighted_average_state_dicts,
)
from agents.network import DuelingQNetwork


def test_weighted_average_state_dicts_with_equal_weights_is_plain_mean():
    sd_a = {"w": torch.tensor([0.0, 2.0])}
    sd_b = {"w": torch.tensor([4.0, 6.0])}

    result = weighted_average_state_dicts([sd_a, sd_b], [1.0, 1.0])

    assert torch.allclose(result["w"], torch.tensor([2.0, 4.0]))


def test_weighted_average_state_dicts_weights_proportionally_to_minibatch_size():
    sd_a = {"w": torch.tensor([0.0])}
    sd_b = {"w": torch.tensor([10.0])}

    # p_a = 3/4, p_b = 1/4 -> 0.75*0 + 0.25*10 = 2.5
    result = weighted_average_state_dicts([sd_a, sd_b], [3.0, 1.0])

    assert torch.allclose(result["w"], torch.tensor([2.5]))


def test_weighted_average_state_dicts_normalizes_weights_that_dont_sum_to_one():
    sd_a = {"w": torch.tensor([1.0])}

    result = weighted_average_state_dicts([sd_a], [7.0])

    assert torch.allclose(result["w"], torch.tensor([1.0]))


def test_weighted_average_state_dicts_rejects_non_positive_total_weight():
    sd_a = {"w": torch.tensor([1.0])}

    with pytest.raises(ValueError):
        weighted_average_state_dicts([sd_a], [0.0])


def test_fedavg_round_synchronizes_all_agents_to_identical_weights():
    agent_a = D3QNAgent(state_dim=2, action_dim=2, seed=0)
    agent_b = D3QNAgent(state_dim=2, action_dim=2, seed=1)
    assert not torch.equal(agent_a.online.value_head.bias, agent_b.online.value_head.bias)

    fedavg_round({"a": agent_a, "b": agent_b}, {"a": 1.0, "b": 1.0})

    for p_a, p_b in zip(agent_a.online.parameters(), agent_b.online.parameters()):
        assert torch.equal(p_a, p_b)


def test_fedavg_round_weights_by_relative_minibatch_size():
    agent_a = D3QNAgent(state_dim=2, action_dim=2, seed=0)
    agent_b = D3QNAgent(state_dim=2, action_dim=2, seed=1)
    with torch.no_grad():
        agent_a.online.value_head.bias.fill_(0.0)
        agent_b.online.value_head.bias.fill_(10.0)

    fedavg_round({"a": agent_a, "b": agent_b}, {"a": 3.0, "b": 1.0})

    expected = torch.tensor([2.5])
    assert torch.allclose(agent_a.online.value_head.bias, expected)
    assert torch.allclose(agent_b.online.value_head.bias, expected)


def test_fedavg_round_leaves_target_networks_untouched():
    agent_a = D3QNAgent(state_dim=2, action_dim=2, seed=0)
    agent_b = D3QNAgent(state_dim=2, action_dim=2, seed=1)
    target_a_before = {k: v.clone() for k, v in agent_a.target.state_dict().items()}

    fedavg_round({"a": agent_a, "b": agent_b}, {"a": 1.0, "b": 1.0})

    for key, value in agent_a.target.state_dict().items():
        assert torch.equal(value, target_a_before[key])


def test_fedavg_node_loads_route_every_transfer_through_the_bs():
    # Each agent uploads its model and downloads the global one; the BS takes part in all 2K transfers.
    loads = fedavg_node_loads(["v2v_0", "v2v_1", "v2v_2"])
    assert loads == {"v2v_0": 2, "v2v_1": 2, "v2v_2": 2, "bs": 6}


def test_weight_dispersion_is_zero_after_fedavg_round():
    agent_a = D3QNAgent(state_dim=2, action_dim=2, seed=0)
    agent_b = D3QNAgent(state_dim=2, action_dim=2, seed=1)
    agents = {"a": agent_a, "b": agent_b}
    assert weight_dispersion(agents) > 0

    fedavg_round(agents, {"a": 1.0, "b": 1.0})

    assert weight_dispersion(agents) == 0.0


def test_weight_dispersion_is_mean_distance_to_the_plain_average():
    # All weights 0 vs all weights 1: the average is 0.5 everywhere, so each agent sits at
    # 0.5 * sqrt(P) from it.
    agent_a = D3QNAgent(state_dim=2, action_dim=2, seed=0)
    agent_b = D3QNAgent(state_dim=2, action_dim=2, seed=1)
    with torch.no_grad():
        for p in agent_a.online.parameters():
            p.fill_(0.0)
        for p in agent_b.online.parameters():
            p.fill_(1.0)
    num_params = sum(p.numel() for p in agent_a.online.parameters())

    result = weight_dispersion({"a": agent_a, "b": agent_b})

    assert math.isclose(result, 0.5 * math.sqrt(num_params), rel_tol=1e-6)


def test_model_num_bytes_counts_the_shared_online_network():
    # Default V2V network: 46 inputs, 128-128 hidden, 60 actions -> 30,397 float32 parameters.
    network = DuelingQNetwork(state_dim=46, action_dim=60, hidden_dim=128)
    assert model_num_bytes(network) == 30_397 * 4
