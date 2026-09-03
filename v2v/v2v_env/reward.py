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
    weights: tuple[float, float, float],
    penalty: float,
) -> float:
    lambda1, lambda2, lambda3 = weights

    cue_term = sum(utility(r - cue_min_rate_bps, penalty) for r in cue_rates)
    queue_term = sum(utility(max_queue_length_bits - q, penalty) for q in v2v_queues)
    reliability_term = sum(utility(m, penalty) for m in reliability_margins)

    return lambda1 * cue_term + lambda2 * queue_term + lambda3 * reliability_term
