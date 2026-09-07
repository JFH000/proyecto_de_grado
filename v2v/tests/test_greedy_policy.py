import numpy as np

from agents.greedy_policy import greedy_action
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def make_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=3, max_steps=5)


def make_obs(neighbor_counts: list[float]) -> np.ndarray:
    m = len(neighbor_counts)
    return np.concatenate([np.zeros(m), np.zeros(m), neighbor_counts, [0.0]])


def test_greedy_action_picks_channel_with_fewest_neighbors():
    params = make_params()
    obs = make_obs([3, 0, 5, 2])

    action = greedy_action(obs, params)

    env = V2VEnv(params)
    channel, _ = env._decode_action(action)
    assert channel == 1


def test_greedy_action_uses_maximum_available_power_level():
    params = make_params()
    obs = make_obs([0, 0, 0, 0])

    action = greedy_action(obs, params)

    env = V2VEnv(params)
    _, power_level = env._decode_action(action)
    assert power_level == params.num_power_levels


def test_greedy_action_breaks_ties_by_lowest_channel_index():
    params = make_params()
    obs = make_obs([2, 2, 0, 0])

    action = greedy_action(obs, params)

    env = V2VEnv(params)
    channel, _ = env._decode_action(action)
    assert channel == 2
