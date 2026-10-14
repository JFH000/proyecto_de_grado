"""Non-learning Greedy baseline (Li et al. 2022, Sec. V): always picks the
channel with the fewest co-channel V2V neighbors from the last slot -- the
direct local signal for V2V interference available in the observation (see
V2VEnv._build_observations) -- and transmits at the maximum available power
level.

Ties are broken uniformly at random. Breaking them by lowest channel index made
every pair see the same all-zero counts in the first slot and pick channel 0,
then all move to channel 1 together, and back: ~98% of pair-slots shared a
channel, an artificially weak baseline.
"""

import numpy as np

from v2v_env.params import V2VEnvParams


def greedy_action(obs: np.ndarray, params: V2VEnvParams, rng: np.random.Generator) -> int:
    m = params.num_cues
    neighbor_counts = obs[2 * m : 3 * m]
    candidates = np.flatnonzero(neighbor_counts == neighbor_counts.min())
    channel = int(rng.choice(candidates))
    max_power_level = params.num_power_levels
    return channel * (params.num_power_levels + 1) + max_power_level
