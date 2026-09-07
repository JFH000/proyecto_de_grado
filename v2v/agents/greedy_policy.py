"""Non-learning Greedy baseline (Li et al. 2022, Sec. V): always picks the
channel with the fewest co-channel V2V neighbors from the last slot -- the
direct local signal for V2V interference available in the observation (see
V2VEnv._build_observations) -- and transmits at the maximum available power
level.
"""

import numpy as np

from v2v_env.params import V2VEnvParams


def greedy_action(obs: np.ndarray, params: V2VEnvParams) -> int:
    m = params.num_cues
    neighbor_counts = obs[2 * m : 3 * m]
    channel = int(np.argmin(neighbor_counts))
    max_power_level = params.num_power_levels
    return channel * (params.num_power_levels + 1) + max_power_level
