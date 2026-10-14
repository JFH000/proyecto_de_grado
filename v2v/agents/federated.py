"""FedAvg aggregation layer for D3QN agents (checklist node 5).

Every N slots, each agent's online-network weights are treated as a local
model theta_k with relative weight p_k proportional to the minibatch size it
trained on since the last round. The global model theta_g = sum_k p_k*theta_k
(Li et al. 2022, Sec. IV-B/C) is then broadcast back, replacing each agent's
local online weights.

Also measures what the aggregation costs and achieves: the models each node
transfers per round, the size of one transfer, and how far apart the agents'
weights are (communication and consensus metrics of the thesis).
"""

import torch
from torch.nn.utils import parameters_to_vector

from agents.d3qn_agent import D3QNAgent

BS = "bs"


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    averaged: dict[str, torch.Tensor] = {}
    for key, reference in state_dicts[0].items():
        weighted_sum = sum(sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights))
        averaged[key] = weighted_sum.to(reference.dtype)
    return averaged


def fedavg_round(
    agents: dict[str, D3QNAgent], weights: dict[str, float]
) -> dict[str, torch.Tensor]:
    agent_ids = list(agents)
    state_dicts = [agents[a].online.state_dict() for a in agent_ids]
    ordered_weights = [weights[a] for a in agent_ids]

    global_state = weighted_average_state_dicts(state_dicts, ordered_weights)

    for a in agent_ids:
        agents[a].online.load_state_dict(global_state)

    return global_state


def fedavg_node_loads(agent_ids: list[str]) -> dict[str, int]:
    """Models each node sends or receives in one FedAvg round.

    Every agent uploads its online network to the BS and downloads the global one, so each
    agent handles 2 transfers and the BS all 2K of them.
    """
    loads = {agent_id: 2 for agent_id in agent_ids}
    loads[BS] = 2 * len(agent_ids)
    return loads


def weight_dispersion(agents: dict[str, D3QNAgent]) -> float:
    """Mean Euclidean distance from each agent's online weights to the agents' plain average.

    Zero right after a FedAvg round; how far it grows between rounds shows how much the
    agents drift apart while they learn alone.
    """
    vectors = torch.stack([parameters_to_vector(agent.online.parameters()) for agent in agents.values()])
    return float((vectors - vectors.mean(dim=0)).norm(dim=1).mean())


def model_num_bytes(network: torch.nn.Module) -> int:
    """Size of one model transfer: the online network's parameters (the target network is never shared)."""
    return sum(p.numel() * p.element_size() for p in network.parameters())
