"""Decentralized aggregation by gossip between neighboring vehicles (the thesis proposal).

Replaces FedAvg's central averaging at the BS (agents/federated.py). Every N slots, each
agent pulls the online-network weights of the pairs within the neighbor radius (the same
300 m neighborhood its observation uses, V2VEnv.neighbor_mask) and replaces its own with the
average of itself and those neighbors, weighted like FedAvg by the minibatch samples each
trained on since the last round. Optionally, as in DFL-MKF (Ge et al. 2024), each agent only
pulls a random subset of at most S of those neighbors, drawn anew every round.

Rounds are synchronous: every agent averages the weights from before the round, so the
result does not depend on the order agents are visited in. An agent with no neighbor in
range keeps its model and moves nothing. As in FedAvg, target networks stay local.
"""

import numpy as np

from agents.d3qn_agent import D3QNAgent
from agents.federated import weighted_average_state_dicts


def gossip_round(
    agents: dict[str, D3QNAgent],
    weights: dict[str, float],
    neighbor_mask: np.ndarray,
    rng: np.random.Generator,
    max_neighbors: int | None = None,
) -> dict[str, int]:
    """One gossip round. `neighbor_mask[i, j]` says whether agent j (in `agents` order) is in
    range of agent i. Returns the models each agent sent or received, for the traffic metrics."""
    agent_ids = list(agents)
    before = {a: {k: v.clone() for k, v in agents[a].online.state_dict().items()} for a in agent_ids}
    loads = dict.fromkeys(agent_ids, 0)

    for i, receiver in enumerate(agent_ids):
        neighbors = np.flatnonzero(neighbor_mask[i])
        if max_neighbors is not None and neighbors.size > max_neighbors:
            neighbors = rng.choice(neighbors, size=max_neighbors, replace=False)
        senders = [agent_ids[j] for j in sorted(neighbors)]
        for sender in senders:  # each neighbor sends its model to the receiver
            loads[sender] += 1
            loads[receiver] += 1

        group = [receiver, *senders]
        group_weights = [weights[a] for a in group]
        if senders and sum(group_weights) > 0:
            agents[receiver].online.load_state_dict(
                weighted_average_state_dicts([before[a] for a in group], group_weights)
            )

    return loads
