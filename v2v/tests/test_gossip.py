import math

import numpy as np
import pytest
import torch

from agents.d3qn_agent import D3QNAgent
from agents.federated import fedavg_round
from agents.gossip import gossip_round

IDS = ["v2v_0", "v2v_1", "v2v_2"]


def make_agents(n: int = 3) -> dict[str, D3QNAgent]:
    return {f"v2v_{i}": D3QNAgent(state_dim=2, action_dim=2, seed=i) for i in range(n)}


def set_bias(agents: dict[str, D3QNAgent], values: list[float]) -> None:
    with torch.no_grad():
        for agent, value in zip(agents.values(), values):
            agent.online.value_head.bias.fill_(value)


def biases(agents: dict[str, D3QNAgent]) -> list[float]:
    return [agent.online.value_head.bias.item() for agent in agents.values()]


def chain_mask() -> np.ndarray:
    # v2v_0 - v2v_1 - v2v_2: the ends only see the middle pair.
    return np.array([[False, True, False], [True, False, True], [False, True, False]])


def test_gossip_on_a_complete_graph_matches_fedavg():
    gossip_agents, fedavg_agents = make_agents(), make_agents()
    weights = {"v2v_0": 1.0, "v2v_1": 2.0, "v2v_2": 3.0}
    complete = ~np.eye(3, dtype=bool)

    gossip_round(gossip_agents, weights, complete, np.random.default_rng(0))
    fedavg_round(fedavg_agents, weights)

    for g, f in zip(gossip_agents.values(), fedavg_agents.values()):
        for p_g, p_f in zip(g.online.parameters(), f.online.parameters()):
            assert torch.allclose(p_g, p_f)


def test_each_agent_averages_itself_with_its_neighbors_from_the_pre_round_weights():
    # Every agent reads the weights from before the round, so the order agents are visited
    # in does not change the result: 0 -> (0+3)/2, 1 -> (0+3+6)/3, 2 -> (3+6)/2.
    agents = make_agents()
    set_bias(agents, [0.0, 3.0, 6.0])

    gossip_round(agents, dict.fromkeys(IDS, 1.0), chain_mask(), np.random.default_rng(0))

    assert biases(agents) == pytest.approx([1.5, 3.0, 4.5])


def test_gossip_weights_each_model_by_its_minibatch_samples():
    agents = make_agents(2)
    set_bias(agents, [0.0, 10.0])
    connected = np.array([[False, True], [True, False]])

    gossip_round(agents, {"v2v_0": 3.0, "v2v_1": 1.0}, connected, np.random.default_rng(0))

    assert biases(agents) == pytest.approx([2.5, 2.5])


def test_an_isolated_agent_keeps_its_model_and_moves_nothing():
    agents = make_agents()
    set_bias(agents, [0.0, 4.0, 9.0])
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = mask[1, 0] = True  # v2v_2 has nobody in range

    loads = gossip_round(agents, dict.fromkeys(IDS, 1.0), mask, np.random.default_rng(0))

    assert biases(agents) == pytest.approx([2.0, 2.0, 9.0])
    assert loads == {"v2v_0": 2, "v2v_1": 2, "v2v_2": 0}


def test_gossip_node_loads_count_models_sent_and_received():
    # Chain: v2v_1 receives from both ends and sends to both; each end exchanges once.
    loads = gossip_round(make_agents(), dict.fromkeys(IDS, 1.0), chain_mask(), np.random.default_rng(0))
    assert loads == {"v2v_0": 2, "v2v_1": 4, "v2v_2": 2}


def test_max_neighbors_limits_the_models_each_agent_receives():
    # Complete graph of 4 with S = 1: every agent pulls exactly one random neighbor's model.
    agents = make_agents(4)
    set_bias(agents, [0.0, 4.0, 8.0, 12.0])
    complete = ~np.eye(4, dtype=bool)

    loads = gossip_round(agents, {a: 1.0 for a in agents}, complete, np.random.default_rng(0), max_neighbors=1)

    assert sum(loads.values()) == 2 * 4  # 4 transfers, each with a sender and a receiver
    # Each agent ends halfway between its own model and one other agent's.
    originals = [0.0, 4.0, 8.0, 12.0]
    for own, new in zip(originals, biases(agents)):
        assert any(math.isclose(new, (own + other) / 2, rel_tol=1e-6) for other in originals if other != own)


def test_gossip_sampling_is_reproducible_given_the_rng_seed():
    complete = ~np.eye(4, dtype=bool)
    results = []
    for _ in range(2):
        agents = make_agents(4)
        set_bias(agents, [0.0, 4.0, 8.0, 12.0])
        gossip_round(agents, {a: 1.0 for a in agents}, complete, np.random.default_rng(7), max_neighbors=1)
        results.append(biases(agents))
    assert results[0] == results[1]


def test_a_group_where_nobody_trained_keeps_its_models():
    agents = make_agents()
    set_bias(agents, [0.0, 4.0, 9.0])
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = mask[1, 0] = True

    gossip_round(agents, {"v2v_0": 0.0, "v2v_1": 0.0, "v2v_2": 5.0}, mask, np.random.default_rng(0))

    assert biases(agents) == [0.0, 4.0, 9.0]


def test_gossip_leaves_target_networks_untouched():
    agents = make_agents()
    before = [{k: v.clone() for k, v in a.target.state_dict().items()} for a in agents.values()]

    gossip_round(agents, dict.fromkeys(IDS, 1.0), chain_mask(), np.random.default_rng(0))

    for agent, state in zip(agents.values(), before):
        for key, value in agent.target.state_dict().items():
            assert torch.equal(value, state[key])
