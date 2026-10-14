import math

import numpy as np

from v2v_env.env import V2VEnv, encode_channel_gain
from v2v_env.params import V2VEnvParams


def make_env(**overrides):
    params = V2VEnvParams(num_cues=4, num_v2v_pairs=3, num_power_levels=2, **overrides)
    return V2VEnv(params)


def test_observation_space_has_3m_plus_1_dimension():
    env = make_env()
    space = env.observation_space("v2v_0")
    assert space.shape == (3 * 4 + 1,)


def test_action_space_has_m_times_np_plus_1_size():
    env = make_env()
    space = env.action_space("v2v_0")
    assert space.n == 4 * 3


def test_reset_returns_observation_and_info_for_every_agent():
    env = make_env()
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == {"v2v_0", "v2v_1", "v2v_2"}
    assert set(infos.keys()) == {"v2v_0", "v2v_1", "v2v_2"}
    for agent_obs in obs.values():
        assert agent_obs.shape == (3 * 4 + 1,)
        assert np.all(np.isfinite(agent_obs))


def test_reset_queue_length_starts_at_zero():
    env = make_env()
    obs, _ = env.reset(seed=0)
    for agent_obs in obs.values():
        assert agent_obs[-1] == 0.0


def test_reset_is_reproducible_given_same_seed():
    env_a = make_env()
    env_b = make_env()
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    for agent in obs_a:
        assert np.array_equal(obs_a[agent], obs_b[agent])


def test_step_gives_every_agent_the_identical_shared_reward():
    env = make_env()
    env.reset(seed=0)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    values = list(rewards.values())
    assert all(v == values[0] for v in values)


def test_step_reward_is_finite():
    env = make_env()
    env.reset(seed=1)
    rng_actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    _, rewards, _, _, _ = env.step(rng_actions)
    assert all(math.isfinite(v) for v in rewards.values())


def test_step_returns_observation_for_every_agent_with_finite_values():
    env = make_env()
    env.reset(seed=2)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, _, _, _, _ = env.step(actions)
    for agent_obs in obs.values():
        assert np.all(np.isfinite(agent_obs))


def test_zero_power_action_does_not_grow_transmitting_agents_queue_from_arrivals_alone():
    # An agent choosing power level 0 sends nothing, so its queue should only grow
    # (arrivals accumulate, nothing is drained) -- never shrink.
    env = make_env()
    obs, _ = env.reset(seed=3)
    zero_power_actions = {agent: 0 for agent in env.agents}  # channel 0, power level 0
    obs2, _, _, _, _ = env.step(zero_power_actions)
    for agent in env.agents:
        assert obs2[agent][-1] >= obs[agent][-1]


def test_episode_truncates_after_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
    assert all(truncations.values())
    assert env.agents == []


def test_encode_channel_gain_maps_db_around_reference():
    # 1e-10 -> -100 dB (the reference) -> 0; 1e-9 -> -90 dB -> +1; 1e-12 -> -120 dB -> -2
    encoded = encode_channel_gain(np.array([1e-10, 1e-9, 1e-12]), ref_db=-100.0, scale_db=10.0)
    assert np.allclose(encoded, [0.0, 1.0, -2.0])


def test_encode_channel_gain_is_finite_for_a_zero_gain():
    assert np.isfinite(encode_channel_gain(np.array([0.0]), ref_db=-100.0, scale_db=10.0)).all()


def test_channel_features_are_order_one():
    # Raw coefficients (~1e-9 to 1e-11) would give a median magnitude near 1e-10.
    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3)
    env = V2VEnv(params)
    m = params.num_cues
    channel_features = []
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        channel_features += [o[: 2 * m] for o in obs.values()]
    magnitudes = np.abs(np.concatenate(channel_features))
    assert 0.1 < np.median(magnitudes) < 10
    assert magnitudes.max() <= 10


def test_queue_feature_is_fraction_of_max_queue():
    # Zero power -> rate 0, so one slot adds tau * lambda = 1e-3 s * 1e6 bps = 1000 bits = 0.01 Q_max.
    params = V2VEnvParams(num_cues=4, num_v2v_pairs=3, num_power_levels=2)
    env = V2VEnv(params)
    env.reset(seed=0)
    zero_power = {agent: 0 for agent in env.agents}  # channel 0, power level 0
    obs, _, _, _, _ = env.step(zero_power)
    for agent_obs in obs.values():
        assert np.isclose(agent_obs[-1], 0.01)


def test_observations_lie_in_observation_space():
    env = make_env()
    obs, _ = env.reset(seed=0)
    for agent, agent_obs in obs.items():
        assert env.observation_space(agent).contains(agent_obs)


def test_step_info_reports_delay_from_queue_backlog():
    # Zero power: no pair transmits. Slot 1 starts with an empty queue -> 0 ms; slot 2 starts
    # with tau * lambda = 1000 bits waiting -> 1000 bits / 1e6 bps = 1 ms (Little's law).
    env = make_env()
    env.reset(seed=0)
    zero_power = {agent: 0 for agent in env.agents}
    _, _, _, _, infos = env.step(zero_power)
    assert all(info["v2v_delay_ms"] == 0.0 for info in infos.values())
    _, _, _, _, infos = env.step(zero_power)
    assert all(np.isclose(info["v2v_delay_ms"], 1.0) for info in infos.values())


def test_step_info_flags_outage_only_where_reliability_constraint_fails():
    # gamma_o = -50 dB and p_o = 0.5 give a threshold of ~1.4e-5 that any transmitting pair
    # clears; a pair at zero power has expected SINR 0 and always fails it.
    env = make_env(sinr_threshold_db=-50.0, outage_probability_threshold=0.5)
    env.reset(seed=0)
    levels = env.params.num_power_levels + 1
    actions = {"v2v_0": 0 * levels + 0, "v2v_1": 1 * levels + 2, "v2v_2": 2 * levels + 2}
    _, _, _, _, infos = env.step(actions)
    assert infos["v2v_0"]["outage"] is True
    assert infos["v2v_1"]["outage"] is False
    assert infos["v2v_2"]["outage"] is False


def test_step_info_reports_one_shared_cue_rate_in_mbps():
    # Silent V2V pairs leave the CUEs with noise only: a few Mbps on a 1 MHz channel (bps would be ~1e6+).
    env = V2VEnv(V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3))
    env.reset(seed=0)
    _, _, _, _, infos = env.step({agent: 0 for agent in env.agents})
    rates = {info["cue_rate_mbps"] for info in infos.values()}
    assert len(rates) == 1
    assert 1.0 < rates.pop() < 50.0


def test_v2v_interference_lowers_reported_cue_rate():
    # Same seed -> same fading draws; only the V2V actions differ.
    params = V2VEnvParams(num_cues=4, num_v2v_pairs=3, num_power_levels=2)
    silent, loud = V2VEnv(params), V2VEnv(params)
    silent.reset(seed=5)
    loud.reset(seed=5)
    _, _, _, _, info_silent = silent.step({agent: 0 for agent in silent.agents})
    _, _, _, _, info_loud = loud.step({agent: 2 for agent in loud.agents})  # all on channel 0, max power
    assert info_loud["v2v_0"]["cue_rate_mbps"] < info_silent["v2v_0"]["cue_rate_mbps"]


def test_reward_stays_on_the_scale_of_the_penalty():
    # With dimensionless (threshold-relative) terms the reward per slot is tens, not the
    # ~3e7 it reaches when the CUE-rate margin is summed in raw bps.
    env = V2VEnv(V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=50))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    while env.agents:
        actions = {agent: int(rng.integers(env.action_space(agent).n)) for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)
        assert abs(rewards["v2v_0"]) < 1e3


def test_random_action_episode_runs_without_nan_or_inf_rewards():
    env = make_env(max_steps=20)
    env.reset(seed=123)
    step = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for value in rewards.values():
            assert math.isfinite(value)
        for agent_obs in obs.values():
            assert np.all(np.isfinite(agent_obs))
        step += 1
    assert step == 20
