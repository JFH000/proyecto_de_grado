"""Metrics computed from a finished run's CSV rather than during training.

Convergence speed (episodes until the reward settles) is the paper's third headline metric
(Li et al. 2022, Fig. 10) and needs the whole learning curve, so it lives here and the
results notebook applies it to each run.
"""

import math
from collections.abc import Sequence

import numpy as np


def convergence_episode(
    rewards: Sequence[float],
    window: int = 50,
    tolerance: float = 0.05,
    final_fraction: float = 0.1,
) -> float:
    """First episode from which the `window`-episode moving average of the reward stays
    within `tolerance` x |final value| of the final value, the mean of the last
    `final_fraction` of the episodes. NaN if the moving average is still outside that band
    at the end of the run.
    """
    values = np.asarray(rewards, dtype=float)
    final = values[-max(1, math.ceil(final_fraction * len(values))) :].mean()
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")  # index i ends at episode i + window - 1
    outside = np.flatnonzero(np.abs(moving_average - final) > tolerance * abs(final))
    if outside.size == 0:
        return float(window - 1)
    if outside[-1] == moving_average.size - 1:
        return math.nan
    return float(outside[-1] + 1 + window - 1)
