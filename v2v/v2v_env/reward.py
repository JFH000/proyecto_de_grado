"""Shared reward (Li et al. 2022, Eq. 22-23) -- computed centrally, broadcast unchanged
to every agent.

Deviation from the literal printed Eq. 22: the CUE-rate term and the reliability term
both take the form U(satisfied_side - threshold), matching their ">=" constraints
((19c) and (15)). The queue term's constraint (19e) is "Q_k <= Q_max", a "<=" direction,
so for it to follow the same "reward the margin when the constraint holds, flat-penalize
when it doesn't" pattern it must be U(Q_max - Q_k), not U(Q_k - Q_max) as OCR'd from the
PDF -- the literal form rewards larger queue overflows, which contradicts constraint
(19e), the paper's own "guarantee the QoS requirements" framing, and this project's
checklist description of the term as a "queue/delay penalty". Implemented here as
U(Q_max - Q_k); flagged for the user/advisor to double check against their own copy.

Second deviation: every margin is divided by its own threshold before U is applied --
(R_m - R_min) / R_min, (Q_max - Q_k) / Q_max and (LHS - RHS) / RHS of Eq. 15 -- so the
three terms are dimensionless and comparable with the penalty A. In raw units (bps,
bits, linear SINR) the CUE-rate term is ~3e7 per slot, the queue term ~2.5e5 and the
reliability term ~1e2, so the reward tracks only the CUE rate and A = -10 has no effect
on any of the three constraints.

Not in the paper: compute_local_rewards splits the same terms per V2V pair, judging each pair
on its own queue, its own reliability and the CUE of the channel it picked. With
V2VEnvParams.local_reward the agents learn from it; the env still returns the shared reward.
"""

import numpy as np


def utility(x: float, penalty: float) -> float:
    return x if x > 0 else penalty


def compute_reward(
    cue_rates: np.ndarray,
    cue_min_rate_bps: float,
    v2v_queues: np.ndarray,
    max_queue_length_bits: float,
    reliability_margins: np.ndarray,
    reliability_threshold: float,
    weights: tuple[float, float, float],
    penalty: float,
) -> float:
    lambda1, lambda2, lambda3 = weights

    cue_term = sum(utility((r - cue_min_rate_bps) / cue_min_rate_bps, penalty) for r in cue_rates)
    queue_term = sum(
        utility((max_queue_length_bits - q) / max_queue_length_bits, penalty) for q in v2v_queues
    )
    reliability_term = sum(utility(m / reliability_threshold, penalty) for m in reliability_margins)

    return lambda1 * cue_term + lambda2 * queue_term + lambda3 * reliability_term


def compute_local_rewards(
    cue_rates: np.ndarray,
    channels: np.ndarray,
    cue_min_rate_bps: float,
    v2v_queues: np.ndarray,
    max_queue_length_bits: float,
    reliability_margins: np.ndarray,
    reliability_threshold: float,
    weights: tuple[float, float, float],
    penalty: float,
) -> np.ndarray:
    """Per-pair reward (not in the paper): the three terms of compute_reward restricted to what
    each pair's action touches -- the CUE of the channel it picked (every pair on that channel is
    judged on the same CUE), its own queue and its own reliability. It ignores the interference a
    pair causes to the other V2V pairs on its channel."""
    lambda1, lambda2, lambda3 = weights
    cue_utility = np.array([utility((r - cue_min_rate_bps) / cue_min_rate_bps, penalty) for r in cue_rates])
    queue_utility = np.array(
        [utility((max_queue_length_bits - q) / max_queue_length_bits, penalty) for q in v2v_queues]
    )
    reliability_utility = np.array([utility(m / reliability_threshold, penalty) for m in reliability_margins])
    return lambda1 * cue_utility[channels] + lambda2 * queue_utility + lambda3 * reliability_utility
