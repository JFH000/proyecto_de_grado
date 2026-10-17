import csv
import json
import math

import pytest
import torch

from agents.federated import fedavg_node_loads, model_num_bytes
from v2v_env.params import V2VEnvParams

import main
from main import (
    EpisodeAccumulator,
    EpisodeRecord,
    epsilon_schedule,
    make_agents,
    print_convergence_verdict,
    run_metadata,
    run_training,
    write_records_csv,
    write_run_metadata,
)


def make_tiny_params(**overrides) -> V2VEnvParams:
    return V2VEnvParams(**{"num_cues": 4, "num_v2v_pairs": 2, "num_power_levels": 2, "max_steps": 5, **overrides})


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


def slot_info(
    cue_rate_mbps: float,
    v2v_delay_ms: float,
    outage: bool,
    co_channel: bool = False,
    channels_used: int = 1,
    power_ratio: float = 1.0,
    neighbors: int = 1,
) -> dict:
    # Same keys V2VEnv.step puts in each agent's info.
    return {
        "channel": 0,
        "power_w": 0.2,
        "cue_rate_mbps": cue_rate_mbps,
        "v2v_delay_ms": v2v_delay_ms,
        "outage": outage,
        "power_ratio": power_ratio,
        "max_power": power_ratio == 1.0,
        "co_channel": co_channel,
        "channels_used": channels_used,
        "neighbors": neighbors,
    }


def make_record(**overrides) -> EpisodeRecord:
    fields = dict(
        episode=0,
        agent="v2v_0",
        reward=1.0,
        cue_rate_mbps=7.0,
        v2v_delay_ms=0.0,
        outage_rate=0.0,
        neighbors=1.0,
        co_channel_rate=0.0,
        channels_used=1.0,
        mean_power_ratio=1.0,
        max_power_rate=1.0,
        train_loss=0.5,
        weight_dispersion=0.0,
        comm_models=0,
        comm_bytes=0,
        comm_max_load=0,
    )
    fields.update(overrides)
    return EpisodeRecord(**fields)


def test_episode_accumulator_sums_reward_and_averages_slot_metrics():
    acc = EpisodeAccumulator(["v2v_0", "v2v_1"])
    acc.add(
        {"v2v_0": 1.0, "v2v_1": 1.0},
        {
            "v2v_0": slot_info(8.0, 0.0, True, co_channel=True, channels_used=1),
            "v2v_1": slot_info(8.0, 4.0, False, co_channel=True, channels_used=1),
        },
    )
    acc.add(
        {"v2v_0": 3.0, "v2v_1": 3.0},
        {
            "v2v_0": slot_info(6.0, 2.0, False, channels_used=2, power_ratio=0.5),
            "v2v_1": slot_info(6.0, 0.0, False, channels_used=2),
        },
    )
    first, second = acc.records(episode=7)
    # episode, agent, reward, cue rate, delay, outage, neighbors, co-channel, channels used, power, max power
    assert first[:11] == (7, "v2v_0", 4.0, 7.0, 1.0, 0.5, 1.0, 0.5, 1.5, 0.75, 0.5)
    assert second[:11] == (7, "v2v_1", 4.0, 7.0, 2.0, 0.0, 1.0, 0.5, 1.5, 1.0, 1.0)


def test_episode_accumulator_averages_each_agents_training_loss():
    acc = EpisodeAccumulator(["v2v_0", "v2v_1"])
    acc.add({"v2v_0": 0.0, "v2v_1": 0.0}, {"v2v_0": slot_info(1.0, 0.0, False), "v2v_1": slot_info(1.0, 0.0, False)})
    acc.add_loss("v2v_0", 2.0)
    acc.add_loss("v2v_0", 4.0)
    first, second = acc.records(episode=0)
    assert first.train_loss == 3.0
    assert math.isnan(second.train_loss)  # never learned this episode


def test_episode_accumulator_counts_model_transfers_and_busiest_node():
    acc = EpisodeAccumulator(["v2v_0", "v2v_1"], model_bytes=10)
    acc.add({"v2v_0": 0.0, "v2v_1": 0.0}, {"v2v_0": slot_info(1.0, 0.0, False), "v2v_1": slot_info(1.0, 0.0, False)})
    acc.add_aggregation(fedavg_node_loads(["v2v_0", "v2v_1"]))
    acc.add_aggregation(fedavg_node_loads(["v2v_0", "v2v_1"]))
    for record in acc.records(episode=0):
        assert record.comm_models == 8  # 2 rounds x 2K transfers
        assert record.comm_bytes == 80
        assert record.comm_max_load == 8  # the BS takes part in every transfer


def test_episode_accumulator_averages_weight_dispersion_over_measurements():
    acc = EpisodeAccumulator(["v2v_0"])
    acc.add({"v2v_0": 0.0}, {"v2v_0": slot_info(1.0, 0.0, False)})
    acc.add_dispersion(1.0)
    acc.add_dispersion(3.0)
    assert acc.records(episode=0)[0].weight_dispersion == 2.0


def test_episode_accumulator_without_aggregation_reports_no_traffic_and_no_dispersion():
    acc = EpisodeAccumulator(["v2v_0"], model_bytes=10)
    acc.add({"v2v_0": 0.0}, {"v2v_0": slot_info(1.0, 0.0, False)})
    (record,) = acc.records(episode=0)
    assert (record.comm_models, record.comm_bytes, record.comm_max_load) == (0, 0, 0)
    assert math.isnan(record.weight_dispersion)


def test_run_training_fedavg_moves_two_models_per_agent_per_round():
    # Aggregation every slot; with batch_size 2 every agent learns from the second slot of
    # episode 0 on, so episode 1 has 5 rounds of 2K = 4 transfers, all through the BS.
    params = make_tiny_params()
    records, agents = run_training(
        params, episodes=2, seed=0, fedavg_every=1, log_every=100, agent_kwargs={"batch_size": 2}
    )
    model_bytes = model_num_bytes(agents["v2v_0"].online)
    for record in (r for r in records if r.episode == 1):
        assert record.comm_models == 20
        assert record.comm_max_load == 20
        assert record.comm_bytes == 20 * model_bytes


def test_run_training_without_aggregation_moves_no_models():
    params = make_tiny_params()
    records, _ = run_training(
        params,
        episodes=2,
        seed=0,
        fedavg_every=1,
        log_every=100,
        agent_kwargs={"batch_size": 2},
        aggregation="none",
    )
    assert all((r.comm_models, r.comm_bytes, r.comm_max_load) == (0, 0, 0) for r in records)


def test_gossip_with_everyone_in_range_moves_one_model_per_ordered_pair_and_syncs_agents():
    # A 1000 km radius makes the 2 pairs neighbors: each round each sends its model to the other
    # (2 transfers, 2 per node), and averaging both like FedAvg leaves them identical.
    params = make_tiny_params(neighbor_radius_m=1e6)
    records, agents = run_training(
        params, episodes=2, seed=0, fedavg_every=1, log_every=100, agent_kwargs={"batch_size": 2}, aggregation="gossip"
    )
    for record in (r for r in records if r.episode == 1):
        assert record.comm_models == 10  # 5 rounds x 2 transfers
        assert record.comm_max_load == 10
        assert record.neighbors == 1.0
    for p_a, p_b in zip(agents["v2v_0"].online.parameters(), agents["v2v_1"].online.parameters()):
        assert torch.equal(p_a, p_b)


def test_gossip_without_neighbors_moves_nothing_and_behaves_like_dte():
    params = make_tiny_params(neighbor_radius_m=0.0)
    common = dict(episodes=2, seed=0, fedavg_every=1, log_every=100, agent_kwargs={"batch_size": 2})
    gossip, gossip_agents = run_training(params, aggregation="gossip", **common)
    dte, dte_agents = run_training(params, aggregation="none", **common)
    assert all(r.comm_models == 0 for r in gossip)
    assert [r.reward for r in gossip] == [r.reward for r in dte]
    for p_g, p_d in zip(gossip_agents["v2v_0"].online.parameters(), dte_agents["v2v_0"].online.parameters()):
        assert torch.equal(p_g, p_d)


def test_gossip_neighbors_caps_the_models_each_agent_pulls():
    # 3 pairs all in range, S = 1: each agent pulls one model per round -> 3 transfers per round.
    params = make_tiny_params(num_v2v_pairs=3, neighbor_radius_m=1e6)
    records, _ = run_training(
        params,
        episodes=2,
        seed=0,
        fedavg_every=1,
        log_every=100,
        agent_kwargs={"batch_size": 2},
        aggregation="gossip",
        gossip_neighbors=1,
    )
    assert all(r.comm_models == 15 for r in records if r.episode == 1)  # 5 rounds x 3


def test_dte_agents_drift_further_apart_than_fedavg_agents():
    # Dispersion is read just before each aggregation: under FedAvg the agents only had one slot
    # to drift since the last round, under DTE they keep their different initializations.
    params = make_tiny_params()
    common = dict(episodes=3, seed=0, fedavg_every=1, log_every=100, agent_kwargs={"batch_size": 2})
    fedavg, _ = run_training(params, **common)
    dte, _ = run_training(params, aggregation="none", **common)
    fedavg_dispersion = [r.weight_dispersion for r in fedavg if r.episode >= 1]
    dte_dispersion = [r.weight_dispersion for r in dte if r.episode >= 1]
    assert max(fedavg_dispersion) < min(dte_dispersion)


def test_run_training_records_training_loss_once_agents_learn():
    params = make_tiny_params()
    records, _ = run_training(
        params, episodes=2, seed=0, fedavg_every=2, log_every=100, agent_kwargs={"batch_size": 2}
    )
    for record in (r for r in records if r.episode == 1):
        assert math.isfinite(record.train_loss)
        assert record.train_loss >= 0


def test_run_training_records_behaviour_metrics_in_range():
    params = make_tiny_params()
    records, _ = run_training(params, episodes=2, seed=1, fedavg_every=2, log_every=100)
    for r in records:
        assert 0.0 <= r.co_channel_rate <= 1.0
        assert 0.0 <= r.channels_used <= params.num_cues
        assert 0.0 <= r.mean_power_ratio <= 1.0
        assert 0.0 <= r.max_power_rate <= 1.0
        assert 0 <= r.neighbors <= params.num_v2v_pairs - 1


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
    records = [make_record(reward=1.5, comm_models=4), make_record(agent="v2v_1", reward=-2.0)]
    output_path = tmp_path / "rewards.csv"
    write_records_csv(records, output_path)
    with output_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "episode",
        "agent",
        "reward",
        "cue_rate_mbps",
        "v2v_delay_ms",
        "outage_rate",
        "neighbors",
        "co_channel_rate",
        "channels_used",
        "mean_power_ratio",
        "max_power_rate",
        "train_loss",
        "weight_dispersion",
        "comm_models",
        "comm_bytes",
        "comm_max_load",
    ]
    assert rows[1] == ["0", "v2v_0", "1.5", "7.0", "0.0", "0.0", "1.0", "0.0", "1.0", "1.0", "1.0", "0.5", "0.0", "4", "0", "0"]


def test_run_metadata_records_wall_time_and_where_it_ran():
    meta = run_metadata(12.5, device="cpu", method="fedavg", seed=0)
    assert meta["method"] == "fedavg"
    assert meta["seed"] == 0
    assert meta["wall_time_s"] == 12.5
    assert meta["device"] == "cpu"
    assert meta["host"]
    assert meta["torch"] == torch.__version__
    assert "gpu" not in meta  # only reported for runs on cuda


def test_write_run_metadata_lands_next_to_the_csv(tmp_path):
    output_path = tmp_path / "k5" / "fedavg_seed0.csv"
    write_run_metadata(output_path, {"method": "fedavg", "wall_time_s": 3.0})
    with (tmp_path / "k5" / "fedavg_seed0.json").open() as f:
        assert json.load(f) == {"method": "fedavg", "wall_time_s": 3.0}


def test_print_convergence_verdict_mentions_every_agent(capsys):
    records = [make_record(episode=ep, agent=agent, reward=float(ep)) for agent in ("v2v_0", "v2v_1") for ep in range(10)]
    print_convergence_verdict(records, episodes=10)
    captured = capsys.readouterr()
    assert "v2v_0" in captured.out
    assert "v2v_1" in captured.out


def test_reward_scale_scales_what_agents_learn_from_but_not_the_logged_reward():
    # 10 slots never fill a 128-sample minibatch, so nobody learns and both runs act the same.
    params = make_tiny_params()
    plain_records, plain_agents = run_training(params, episodes=2, seed=0, fedavg_every=2, log_every=100)
    scaled_records, scaled_agents = run_training(
        params, episodes=2, seed=0, fedavg_every=2, log_every=100, reward_scale=0.01
    )
    assert [r.reward for r in scaled_records] == [r.reward for r in plain_records]
    for agent_id, agent in scaled_agents.items():
        stored = agent.buffer.rewards[: len(agent.buffer)]
        assert len(agent.buffer) == 10
        assert stored == pytest.approx(0.01 * plain_agents[agent_id].buffer.rewards[: len(agent.buffer)])


def test_cli_passes_buffer_capacity_and_reward_scale_and_records_them(tmp_path, monkeypatch):
    output = tmp_path / "k2" / "fedavg_seed0.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py", "--episodes", "1", "--num-v2v-pairs", "2", "--device", "cpu", "--log-every", "1",
            "--buffer-capacity", "64", "--reward-scale", "0.5", "--output", str(output),
        ],
    )
    main.main()
    meta = json.loads(output.with_suffix(".json").read_text())
    assert meta["buffer_capacity"] == 64
    assert meta["reward_scale"] == 0.5


def test_cli_defaults_keep_the_agents_buffer_and_the_raw_reward(tmp_path, monkeypatch):
    output = tmp_path / "k2" / "fedavg_seed0.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--episodes", "1", "--num-v2v-pairs", "2", "--device", "cpu", "--log-every", "1", "--output", str(output)],
    )
    main.main()
    meta = json.loads(output.with_suffix(".json").read_text())
    assert meta["buffer_capacity"] == 10_000
    assert meta["reward_scale"] == 1.0
