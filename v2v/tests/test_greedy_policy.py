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

    action = greedy_action(obs, params, np.random.default_rng(0))

    env = V2VEnv(params)
    channel, _ = env._decode_action(action)
    assert channel == 1


def test_greedy_action_uses_maximum_available_power_level():
    params = make_params()
    obs = make_obs([0, 0, 0, 0])

    action = greedy_action(obs, params, np.random.default_rng(0))

    env = V2VEnv(params)
    _, power_level = env._decode_action(action)
    assert power_level == params.num_power_levels


def test_greedy_action_breaks_ties_at_random_among_fewest_neighbor_channels():
    # Channels 2 and 3 tie at zero neighbors; both must come up, and never 0 or 1.
    params = make_params()
    obs = make_obs([2, 2, 0, 0])
    env = V2VEnv(params)
    rng = np.random.default_rng(0)

    channels = {env._decode_action(greedy_action(obs, params, rng))[0] for _ in range(200)}

    assert channels == {2, 3}


def test_greedy_pairs_do_not_herd_onto_the_same_channels():
    # Breaking ties by lowest index sent every pair to channel 0, then all to channel 1, and back:
    # 98% of pair-slots shared a channel. With 15 channels for 5 pairs most should not.
    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=50)
    env = V2VEnv(params)
    rng = np.random.default_rng(0)
    shared = []
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        while env.agents:
            obs, _, _, _, infos = env.step({a: greedy_action(obs[a], params, rng) for a in env.agents})
            channels = [info["channel"] for info in infos.values()]
            shared += [channels.count(c) > 1 for c in channels]
    assert np.mean(shared) < 0.5
