import math

import numpy as np

from v2v_env.env import V2VEnv
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
