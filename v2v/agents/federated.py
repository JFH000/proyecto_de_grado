"""FedAvg aggregation layer for D3QN agents (checklist node 5).

Every N slots, each agent's online-network weights are treated as a local
model theta_k with relative weight p_k proportional to the minibatch size it
trained on since the last round. The global model theta_g = sum_k p_k*theta_k
(Li et al. 2022, Sec. IV-B/C) is then broadcast back, replacing each agent's
local online weights.
"""

import torch

from agents.d3qn_agent import D3QNAgent


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
