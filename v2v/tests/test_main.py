import csv
import math

import torch

from v2v_env.params import V2VEnvParams

from main import (
    epsilon_schedule,
    make_agents,
    print_convergence_verdict,
    run_training,
    write_records_csv,
)


def make_tiny_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2, max_steps=5)


def test_epsilon_schedule_starts_high_and_decays_to_floor():
    assert epsilon_schedule(0, total_steps=100) == 1.0
    assert epsilon_schedule(100, total_steps=100) == 0.05


def test_make_agents_forwards_agent_kwargs_overrides():
    params = make_tiny_params()
    agents = make_agents(params, seed=0, per_beta_frames=100, agent_kwargs={"batch_size": 4})
    assert all(agent.batch_size == 4 for agent in agents.values())


def test_run_training_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records, _ = run_training(params, episodes=3, seed=0, fedavg_every=2, log_every=100)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {ep for ep, _, _ in records} == {0, 1, 2}


def test_run_training_rewards_are_finite():
    params = make_tiny_params()
    records, _ = run_training(params, episodes=2, seed=1, fedavg_every=2, log_every=100)
    assert all(math.isfinite(r) for _, _, r in records)


def test_run_training_synchronizes_agents_online_weights():
    params = make_tiny_params()
    _, agents = run_training(
        params,
        episodes=4,
        seed=0,
        fedavg_every=1,
        log_every=100,
        agent_kwargs={"batch_size": 2},
    )
    ids = list(agents)
    for p_a, p_b in zip(agents[ids[0]].online.parameters(), agents[ids[1]].online.parameters()):
        assert torch.equal(p_a, p_b)


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
