import math

import numpy as np

from v2v_env import channel
from v2v_env.env import V2VEnv, encode_channel_gain
from v2v_env.params import V2VEnvParams

P_23_DBM_W = 10 ** (23 / 10) * 1e-3  # CUE and maximum V2V transmit power
NOISE_1_MHZ_W = 10 ** (-174 / 10) * 1e-3 * 1e6  # -174 dBm/Hz over W = 1 MHz


def decode_channel_gain(feature, params: V2VEnvParams):
    """Inverse of the observation encoding (10 log10 h - ref) / scale."""
    return 10 ** ((feature * params.channel_gain_scale_db + params.channel_gain_ref_db) / 10)


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


def test_step_info_reports_neighbors_within_the_neighbor_radius():
    # A 1000 km radius makes every other pair a neighbor; a zero radius leaves every pair alone.
    for radius_m, expected in ((1e6, 2), (0.0, 0)):
        env = make_env(neighbor_radius_m=radius_m)
        env.reset(seed=0)
        _, _, _, _, infos = env.step({agent: 0 for agent in env.agents})
        assert [info["neighbors"] for info in infos.values()] == [expected] * 3


def test_neighbor_mask_is_symmetric_and_agrees_with_the_reported_neighbors():
    env = V2VEnv(V2VEnvParams(num_cues=4, num_v2v_pairs=6, num_power_levels=2))
    env.reset(seed=3)
    mask = env.neighbor_mask
    assert mask.shape == (6, 6)
    assert (mask == mask.T).all() and not mask.diagonal().any()
    _, _, _, _, infos = env.step({agent: 0 for agent in env.agents})
    assert [info["neighbors"] for info in infos.values()] == mask.sum(axis=1).tolist()


def test_step_info_flags_co_channel_only_between_transmitting_pairs():
    env = make_env()
    env.reset(seed=0)
    levels = env.params.num_power_levels + 1
    # Pairs 0 and 1 transmit on channel 0, pair 2 alone on channel 1.
    _, _, _, _, infos = env.step({"v2v_0": 0 * levels + 2, "v2v_1": 0 * levels + 2, "v2v_2": 1 * levels + 2})
    assert [infos[a]["co_channel"] for a in ("v2v_0", "v2v_1", "v2v_2")] == [True, True, False]
    assert all(info["channels_used"] == 2 for info in infos.values())
    # Pair 0 picks channel 0 at zero power: it does not transmit, so pair 1 is alone there.
    _, _, _, _, infos = env.step({"v2v_0": 0 * levels + 0, "v2v_1": 0 * levels + 2, "v2v_2": 1 * levels + 2})
    assert [infos[a]["co_channel"] for a in ("v2v_0", "v2v_1", "v2v_2")] == [False, False, False]
    assert all(info["channels_used"] == 2 for info in infos.values())


def test_step_info_reports_power_as_fraction_of_maximum():
    # N_p = 2 -> levels 0, P_max/2 and P_max.
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, infos = env.step({"v2v_0": 0, "v2v_1": 1, "v2v_2": 2})
    assert [infos[a]["power_ratio"] for a in ("v2v_0", "v2v_1", "v2v_2")] == [0.0, 0.5, 1.0]
    assert [infos[a]["max_power"] for a in ("v2v_0", "v2v_1", "v2v_2")] == [False, False, True]


def test_v2v_interference_lowers_reported_cue_rate():
    # Same seed -> same fading draws; only the V2V actions differ.
    params = V2VEnvParams(num_cues=4, num_v2v_pairs=3, num_power_levels=2)
    silent, loud = V2VEnv(params), V2VEnv(params)
    silent.reset(seed=5)
    loud.reset(seed=5)
    _, _, _, _, info_silent = silent.step({agent: 0 for agent in silent.agents})
    _, _, _, _, info_loud = loud.step({agent: 2 for agent in loud.agents})  # all on channel 0, max power
    assert info_loud["v2v_0"]["cue_rate_mbps"] < info_silent["v2v_0"]["cue_rate_mbps"]


def test_cue_rates_use_the_cue_channel_the_agents_observe():
    # With every V2V pair silent the CUEs only see noise, so CUE m transmits at
    # W log2(1 + P_m h_{m,B}[m] / sigma^2) with the coefficient observed for this slot.
    params = V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2)
    env = V2VEnv(params)
    obs, _ = env.reset(seed=0)
    h_cue = decode_channel_gain(obs["v2v_0"][4:8], params)
    _, _, _, _, infos = env.step({agent: 0 for agent in env.agents})
    expected_mbps = np.mean(np.log2(1 + P_23_DBM_W * h_cue / NOISE_1_MHZ_W))  # W = 1 MHz
    assert np.isclose(infos["v2v_0"]["cue_rate_mbps"], expected_mbps, rtol=1e-9)


def test_v2v_rate_uses_the_own_channel_the_agent_observes():
    # Pair 0 transmits alone at maximum power and the CUEs are effectively silent, so its SINR is
    # P_max h_0[m] / sigma^2 with the coefficient it observed for channel m. Arrivals of 1 Gbit/s
    # exceed any feasible rate, so the queue after one slot is tau * (lambda - R) and reveals R.
    params = V2VEnvParams(
        num_cues=4, num_v2v_pairs=2, num_power_levels=2, cue_tx_power_dbm=-200.0, v2v_arrival_rate_bps=1e9
    )
    env = V2VEnv(params)
    obs, _ = env.reset(seed=0)
    chosen = 2
    h_own = decode_channel_gain(obs["v2v_0"][chosen], params)
    levels = params.num_power_levels + 1
    obs, _, _, _, _ = env.step({"v2v_0": chosen * levels + params.num_power_levels, "v2v_1": 0})
    queue_bits = obs["v2v_0"][-1] * params.max_queue_length_bits
    rate = params.v2v_arrival_rate_bps - queue_bits / params.slot_duration_s
    assert np.isclose(rate, 1e6 * np.log2(1 + P_23_DBM_W * h_own / NOISE_1_MHZ_W), rtol=1e-6)


def test_reliability_uses_large_scale_gain_of_v2v_interferers(monkeypatch):
    # Eq. 15 puts only the large-scale gain of co-channel V2V interferers in the denominator, so
    # with a fixed layout and fixed actions the outage flag cannot change from slot to slot.
    # Pair 0 has a 10 m link and pair 1 transmits 90 m from pair 0's receiver on the same channel:
    # large-scale SINR = (90 / 10)^1.61 ~ 34.4 above the threshold of ~30.0, so pair 0 is never
    # in outage. With the interferer's fading in the denominator it would fail in ~1/3 of the slots.
    layout = channel.Positions(
        bs=np.zeros(2),
        cue=np.array([[400.0, 0.0], [0.0, 400.0]]),
        v2v_tx=np.array([[100.0, 0.0], [200.0, 0.0]]),
        v2v_rx=np.array([[110.0, 0.0], [210.0, 0.0]]),
    )
    monkeypatch.setattr(channel, "sample_positions", lambda rng, params: layout)
    env = V2VEnv(V2VEnvParams(num_cues=2, num_v2v_pairs=2, num_power_levels=1, cue_tx_power_dbm=-200.0))
    env.reset(seed=0)
    both_on_channel_0_at_max_power = {"v2v_0": 1, "v2v_1": 1}
    outages = []
    while env.agents:
        _, _, _, _, infos = env.step(both_on_channel_0_at_max_power)
        outages.append(infos["v2v_0"]["outage"])
    assert not any(outages)


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
