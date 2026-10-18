import math

from v2v_env.params import V2VEnvParams

from train_greedy import run_greedy


def make_tiny_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2, max_steps=5)


def test_run_greedy_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records = run_greedy(params, episodes=3, seed=0)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {r.episode for r in records} == {0, 1, 2}


def test_run_greedy_rewards_are_finite():
    params = make_tiny_params()
    records = run_greedy(params, episodes=2, seed=1)
    assert all(math.isfinite(r.reward) for r in records)


def test_run_greedy_records_carry_qos_metrics():
    params = make_tiny_params()
    records = run_greedy(params, episodes=2, seed=1)
    for r in records:
        assert r.cue_rate_mbps > 0
        assert r.v2v_delay_ms >= 0
        assert 0.0 <= r.outage_rate <= 1.0


def test_consecutive_seeds_do_not_replay_the_same_episodes():
    # With env seed = seed + episode, run seed=0 episode 1 and run seed=1 episode 0 would share
    # a layout and, Greedy being deterministic, the same reward.
    params = make_tiny_params()
    run0 = run_greedy(params, episodes=2, seed=0)
    run1 = run_greedy(params, episodes=1, seed=1)
    assert [r.reward for r in run0 if r.episode == 1] != [r.reward for r in run1 if r.episode == 0]


def test_run_greedy_is_deterministic_given_same_seed():
    params = make_tiny_params()
    first = run_greedy(params, episodes=2, seed=7)
    second = run_greedy(params, episodes=2, seed=7)
    assert first == second


def test_run_greedy_always_transmits_at_maximum_power():
    params = make_tiny_params()
    records = run_greedy(params, episodes=2, seed=0)
    assert all(r.max_power_rate == 1.0 and r.mean_power_ratio == 1.0 for r in records)


def test_run_greedy_neither_trains_nor_shares_models():
    params = make_tiny_params()
    records = run_greedy(params, episodes=2, seed=0)
    for r in records:
        assert math.isnan(r.train_loss)
        assert math.isnan(r.weight_dispersion)
        assert (r.comm_models, r.comm_bytes, r.comm_max_load) == (0, 0, 0)


def test_cli_policy_option_names_the_method_in_the_metadata(tmp_path, monkeypatch):
    import json

    import train_greedy

    output = tmp_path / "k2" / "umbral_seed0.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["train_greedy.py", "--episodes", "1", "--num-v2v-pairs", "2", "--policy", "umbral", "--output", str(output)],
    )
    train_greedy.main()
    assert json.loads(output.with_suffix(".json").read_text())["method"] == "umbral"
