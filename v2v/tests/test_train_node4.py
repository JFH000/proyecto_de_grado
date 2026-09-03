import csv
import math

from v2v_env.params import V2VEnvParams

from train_node4 import (
    epsilon_schedule,
    print_convergence_verdict,
    run_training,
    write_records_csv,
)


def make_tiny_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2, max_steps=5)


def test_epsilon_schedule_starts_high_and_decays_to_floor():
    assert epsilon_schedule(0, total_steps=100) == 1.0
    assert epsilon_schedule(100, total_steps=100) == 0.05


def test_run_training_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records = run_training(params, episodes=3, seed=0, log_every=100)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {ep for ep, _, _ in records} == {0, 1, 2}


def test_run_training_rewards_are_finite():
    params = make_tiny_params()
    records = run_training(params, episodes=2, seed=1, log_every=100)
    assert all(math.isfinite(r) for _, _, r in records)


def test_write_records_csv_round_trips(tmp_path):
    records = [(0, "v2v_0", 1.5), (0, "v2v_1", -2.0)]
    output_path = tmp_path / "rewards.csv"
    write_records_csv(records, output_path)
    with output_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "agent", "reward"]
    assert rows[1] == ["0", "v2v_0", "1.5"]


def test_print_convergence_verdict_mentions_every_agent(capsys):
    records = [(ep, "v2v_0", float(ep)) for ep in range(10)] + [(ep, "v2v_1", float(ep)) for ep in range(10)]
    print_convergence_verdict(records, episodes=10)
    captured = capsys.readouterr()
    assert "v2v_0" in captured.out
    assert "v2v_1" in captured.out
