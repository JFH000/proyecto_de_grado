import numpy as np

from agents.threshold_policy import threshold_action
from train_greedy import run_greedy
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def make_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=3, max_steps=5)


def make_obs(params: V2VEnvParams, own_gain_db: float, queue: float, neighbor_counts=(0, 0, 0, 0)) -> np.ndarray:
    """Observation with the same own-link gain on every channel, encoded as the env does."""
    m = params.num_cues
    own = np.full(m, (own_gain_db - params.channel_gain_ref_db) / params.channel_gain_scale_db)
    return np.concatenate([own, np.zeros(m), neighbor_counts, [queue]])


def encoded(params: V2VEnvParams, gain_db: float) -> float:
    return (gain_db - params.channel_gain_ref_db) / params.channel_gain_scale_db


def decode(params: V2VEnvParams, action: int) -> tuple[int, int]:
    return V2VEnv(params)._decode_action(action)


def test_strong_link_transmits_at_maximum_power_even_with_an_empty_queue():
    params = make_params()
    _, level = decode(params, threshold_action(make_obs(params, -85.0, 0.0), params, np.random.default_rng(0)))
    assert level == params.num_power_levels


def test_weak_link_stays_silent_while_the_queue_has_room():
    params = make_params()
    _, level = decode(params, threshold_action(make_obs(params, -95.0, 0.5), params, np.random.default_rng(0)))
    assert level == 0


def test_weak_link_transmits_at_maximum_power_once_the_queue_fills():
    params = make_params()
    _, level = decode(params, threshold_action(make_obs(params, -95.0, 0.95), params, np.random.default_rng(0)))
    assert level == params.num_power_levels


def test_link_strength_is_the_mean_gain_over_channels_not_any_single_channel():
    # The small-scale fading is unit-mean, so the mean over channels estimates the large-scale gain.
    params = make_params()
    obs = make_obs(params, -100.0, 0.0)
    obs[0] = encoded(params, -80.0)  # mean of 1e-8 and three 1e-10 -> -85.7 dB: strong
    _, level = decode(params, threshold_action(obs, params, np.random.default_rng(0)))
    assert level == params.num_power_levels
    obs[0] = encoded(params, -90.0)  # mean of 1e-9 and three 1e-10 -> -94.4 dB: weak, despite one good channel
    _, level = decode(params, threshold_action(obs, params, np.random.default_rng(0)))
    assert level == 0


def test_picks_a_channel_with_the_fewest_neighbors_like_greedy():
    params = make_params()
    rng = np.random.default_rng(0)
    obs = make_obs(params, -85.0, 0.0, (2, 0, 3, 0))
    channels = {decode(params, threshold_action(obs, params, rng))[0] for _ in range(100)}
    assert channels == {1, 3}


def test_run_greedy_with_the_threshold_policy_silences_weak_links_and_keeps_delays_under_the_maximum():
    params = V2VEnvParams(num_cues=15, num_v2v_pairs=10, num_power_levels=3, max_steps=200)
    records = run_greedy(params, episodes=2, seed=0, policy="umbral")
    assert 0.0 < np.mean([r.mean_power_ratio for r in records]) < 1.0
    # A weak link waits until its queue reaches 90 % of Q_max (90 ms of traffic), then drains it.
    assert max(r.v2v_delay_ms for r in records) < 100.0
