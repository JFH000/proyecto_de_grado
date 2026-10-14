import csv
import math

import pytest
import torch

from v2v_env.params import V2VEnvParams

from main import (
    EpisodeAccumulator,
    EpisodeRecord,
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


def test_agents_of_consecutive_seeds_start_from_different_weights():
    # With agent seed = seed + index, agent v2v_1 of seed 0 and agent v2v_0 of seed 1 would be clones.
    params = make_tiny_params()
    a = make_agents(params, seed=0, per_beta_frames=100)["v2v_1"]
    b = make_agents(params, seed=1, per_beta_frames=100)["v2v_0"]
    assert any(not torch.equal(p, q) for p, q in zip(a.online.parameters(), b.online.parameters()))


def test_run_training_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records, _ = run_training(params, episodes=3, seed=0, fedavg_every=2, log_every=100)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {r.episode for r in records} == {0, 1, 2}


def test_run_training_rewards_are_finite():
    params = make_tiny_params()
    records, _ = run_training(params, episodes=2, seed=1, fedavg_every=2, log_every=100)
    assert all(math.isfinite(r.reward) for r in records)


def test_run_training_records_carry_qos_metrics():
    params = make_tiny_params()
    records, _ = run_training(params, episodes=2, seed=1, fedavg_every=2, log_every=100)
    for r in records:
        assert r.cue_rate_mbps > 0
        assert r.v2v_delay_ms >= 0
        assert 0.0 <= r.outage_rate <= 1.0


def slot_info(cue_rate_mbps: float, v2v_delay_ms: float, outage: bool) -> dict:
    # Same keys V2VEnv.step puts in each agent's info.
    return {
        "channel": 0,
        "power_w": 0.2,
        "cue_rate_mbps": cue_rate_mbps,
        "v2v_delay_ms": v2v_delay_ms,
        "outage": outage,
    }


def test_episode_accumulator_sums_reward_and_averages_metrics():
    acc = EpisodeAccumulator(["v2v_0", "v2v_1"])
    acc.add(
        {"v2v_0": 1.0, "v2v_1": 1.0},
        {"v2v_0": slot_info(8.0, 0.0, True), "v2v_1": slot_info(8.0, 4.0, False)},
    )
    acc.add(
        {"v2v_0": 3.0, "v2v_1": 3.0},
        {"v2v_0": slot_info(6.0, 2.0, False), "v2v_1": slot_info(6.0, 0.0, False)},
    )
    assert acc.records(episode=7) == [
        EpisodeRecord(7, "v2v_0", 4.0, 7.0, 1.0, 0.5),
        EpisodeRecord(7, "v2v_1", 4.0, 7.0, 2.0, 0.0),
    ]


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


def test_run_training_without_aggregation_keeps_agents_independent():
    # Same setup that synchronizes the weights under FedAvg (aggregation every slot),
    # but with aggregation="none" the agents must never be averaged: DTE control.
    params = make_tiny_params()
    _, agents = run_training(
        params,
        episodes=4,
        seed=0,
        fedavg_every=1,
        log_every=100,
        agent_kwargs={"batch_size": 2},
        aggregation="none",
    )
    ids = list(agents)
    pairs = zip(agents[ids[0]].online.parameters(), agents[ids[1]].online.parameters())
    assert any(not torch.equal(p_a, p_b) for p_a, p_b in pairs)


def test_run_training_rejects_unknown_aggregation():
    params = make_tiny_params()
    with pytest.raises(ValueError):
        run_training(params, episodes=1, seed=0, fedavg_every=1, log_every=100, aggregation="fedav")


def test_write_records_csv_round_trips(tmp_path):
    records = [EpisodeRecord(0, "v2v_0", 1.5, 7.25, 0.5, 0.25), EpisodeRecord(0, "v2v_1", -2.0, 7.25, 0.0, 1.0)]
    output_path = tmp_path / "rewards.csv"
    write_records_csv(records, output_path)
    with output_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "agent", "reward", "cue_rate_mbps", "v2v_delay_ms", "outage_rate"]
    assert rows[1] == ["0", "v2v_0", "1.5", "7.25", "0.5", "0.25"]


def test_print_convergence_verdict_mentions_every_agent(capsys):
    records = [EpisodeRecord(ep, agent, float(ep), 7.0, 0.0, 0.0) for agent in ("v2v_0", "v2v_1") for ep in range(10)]
    print_convergence_verdict(records, episodes=10)
    captured = capsys.readouterr()
    assert "v2v_0" in captured.out
    assert "v2v_1" in captured.out
