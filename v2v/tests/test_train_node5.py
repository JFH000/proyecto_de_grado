import math

import torch

from v2v_env.params import V2VEnvParams

from train_node5 import run_training_fedavg


def make_tiny_params() -> V2VEnvParams:
    return V2VEnvParams(num_cues=4, num_v2v_pairs=2, num_power_levels=2, max_steps=5)


def test_run_training_fedavg_returns_one_record_per_episode_per_agent():
    params = make_tiny_params()
    records, _ = run_training_fedavg(params, episodes=3, seed=0, fedavg_every=2, log_every=100)
    assert len(records) == 3 * params.num_v2v_pairs
    assert {ep for ep, _, _ in records} == {0, 1, 2}


def test_run_training_fedavg_rewards_are_finite():
    params = make_tiny_params()
    records, _ = run_training_fedavg(params, episodes=2, seed=1, fedavg_every=2, log_every=100)
    assert all(math.isfinite(r) for _, _, r in records)


def test_run_training_fedavg_synchronizes_agents_online_weights():
    params = make_tiny_params()
    _, agents = run_training_fedavg(
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
