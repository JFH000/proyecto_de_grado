"""Non-learning threshold rule: a reference for what local information alone can achieve.

Not a method of Li et al. (2022). Each pair picks its channel like Greedy (fewest co-channel
neighbors in the last slot, ties at random) but chooses its power from two features of its own
observation:

- Own-link strength: the mean over channels of the observed own-link gains. The small-scale
  fading is unit-mean, so this mean estimates the large-scale gain, i.e. how far the receiver is.
- Queue: the fraction of Q_max waiting at the transmitter.

A strong link transmits at maximum power. A weak link stays silent, which spares the CUE on its
channel, until its queue reaches `queue_threshold` of Q_max, and then transmits at maximum power
until the queue empties. The default threshold, -91 dB, is the gain of a 25 m link under the
environment's path-loss law.
"""

import numpy as np

from v2v_env.params import V2VEnvParams

STRONG_LINK_DB = -91.0  # path_loss_coefficient_db - 10 * path_loss_exponent * log10(25 m)
QUEUE_THRESHOLD = 0.9


def own_link_gain_db(obs: np.ndarray, params: V2VEnvParams) -> float:
    """Large-scale own-link gain estimated from the observation: mean of the M decoded gains, in dB."""
    m = params.num_cues
    gains = 10 ** ((obs[:m] * params.channel_gain_scale_db + params.channel_gain_ref_db) / 10)
    return float(10 * np.log10(gains.mean()))


def threshold_action(
    obs: np.ndarray,
    params: V2VEnvParams,
    rng: np.random.Generator,
    strong_link_db: float = STRONG_LINK_DB,
    queue_threshold: float = QUEUE_THRESHOLD,
) -> int:
    m = params.num_cues
    neighbor_counts = obs[2 * m : 3 * m]
    channel = int(rng.choice(np.flatnonzero(neighbor_counts == neighbor_counts.min())))
    transmit = own_link_gain_db(obs, params) >= strong_link_db or obs[-1] >= queue_threshold
    power_level = params.num_power_levels if transmit else 0
    return channel * (params.num_power_levels + 1) + power_level
