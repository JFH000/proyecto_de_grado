import math

from v2v_env.params import V2VEnvParams

from train_greedy import run_greedy


def make_tiny_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2, max_steps=5)


def test_run_greedy_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records = run_greedy(params, episodes=3, seed=0)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {ep for ep, _, _ in records} == {0, 1, 2}


def test_run_greedy_rewards_are_finite():
    params = make_tiny_params()
    records = run_greedy(params, episodes=2, seed=1)
    assert all(math.isfinite(r) for _, _, r in records)


def test_run_greedy_is_deterministic_given_same_seed():
    params = make_tiny_params()
    first = run_greedy(params, episodes=2, seed=7)
    second = run_greedy(params, episodes=2, seed=7)
    assert first == second
