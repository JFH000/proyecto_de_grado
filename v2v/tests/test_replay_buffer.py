import numpy as np

from agents.replay_buffer import PrioritizedReplayBuffer


def test_len_grows_up_to_capacity_then_stays_there():
    buf = PrioritizedReplayBuffer(capacity=3, state_dim=2)
    for i in range(5):
        buf.add(np.array([float(i), 0.0]), action=0, reward=0.0, next_state=np.zeros(2))
    assert len(buf) == 3


def test_add_past_capacity_overwrites_oldest_entry():
    buf = PrioritizedReplayBuffer(capacity=3, state_dim=2)
    for i in range(4):  # index 0's original entry (state=[0,0]) gets overwritten on the 4th add
        buf.add(np.array([float(i), 0.0]), action=i, reward=float(i), next_state=np.zeros(2))
    states, *_ = buf.sample(batch_size=3, beta=0.5)
    assert not np.any(np.all(states == np.array([0.0, 0.0]), axis=1))


def test_sample_returns_requested_batch_size_with_correct_shapes():
    buf = PrioritizedReplayBuffer(capacity=10, state_dim=4, rng=np.random.default_rng(0))
    for i in range(10):
        buf.add(np.full(4, float(i)), action=i % 2, reward=float(i), next_state=np.full(4, float(i) + 1))
    states, actions, rewards, next_states, weights, indices = buf.sample(batch_size=6, beta=0.5)
    assert states.shape == (6, 4)
    assert next_states.shape == (6, 4)
    assert actions.shape == (6,)
    assert rewards.shape == (6,)
    assert weights.shape == (6,)
    assert indices.shape == (6,)


def test_higher_priority_transition_is_sampled_more_often():
    buf = PrioritizedReplayBuffer(capacity=2, state_dim=1, alpha=1.0, rng=np.random.default_rng(0))
    buf.add(np.array([0.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.add(np.array([1.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.update_priorities(np.array([0, 1]), td_errors=np.array([0.001, 100.0]))

    counts = {0: 0, 1: 0}
    for _ in range(200):
        _, _, _, _, _, indices = buf.sample(batch_size=1, beta=0.0)
        counts[int(indices[0])] += 1
    assert counts[1] > counts[0] * 5


def test_importance_weights_match_formula_at_beta_one():
    buf = PrioritizedReplayBuffer(capacity=2, state_dim=1, alpha=1.0, eps=0.0, rng=np.random.default_rng(0))
    buf.add(np.array([0.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.add(np.array([1.0]), action=0, reward=0.0, next_state=np.zeros(1))
    buf.update_priorities(np.array([0, 1]), td_errors=np.array([1.0, 3.0]))
    # priorities = [1, 3] (alpha=1), probs = [0.25, 0.75], N=2
    # raw weights (N*P)^-1 = [1/(2*0.25), 1/(2*0.75)] = [2.0, 2/3], normalized by max -> [1.0, 1/3]
    _, _, _, _, weights, indices = buf.sample(batch_size=2, beta=1.0)
    expected = {0: 1.0, 1: 1.0 / 3.0}
    for idx, w in zip(indices, weights):
        assert np.isclose(w, expected[int(idx)], atol=1e-6)
