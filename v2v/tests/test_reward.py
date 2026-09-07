import numpy as np

from v2v_env.reward import utility, compute_reward


def test_utility_returns_x_when_positive():
    assert utility(5.0, penalty=-10.0) == 5.0


def test_utility_returns_penalty_when_zero_or_negative():
    assert utility(0.0, penalty=-10.0) == -10.0
    assert utility(-3.0, penalty=-10.0) == -10.0


def test_reward_rewards_cue_rate_margin_above_minimum():
    weights = (1.0, 0.0, 0.0)
    high_margin = compute_reward(
        cue_rates=np.array([2.0e6]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([]),
        max_queue_length_bits=1.0,
        reliability_margins=np.array([]),
        weights=weights,
        penalty=-10.0,
    )
    low_margin = compute_reward(
        cue_rates=np.array([1.1e6]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([]),
        max_queue_length_bits=1.0,
        reliability_margins=np.array([]),
        weights=weights,
        penalty=-10.0,
    )
    assert high_margin > low_margin > 0


def test_reward_penalizes_cue_rate_below_minimum():
    weights = (1.0, 0.0, 0.0)
    r = compute_reward(
        cue_rates=np.array([0.5e6]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([]),
        max_queue_length_bits=1.0,
        reliability_margins=np.array([]),
        weights=weights,
        penalty=-10.0,
    )
    assert r == -10.0


def test_reward_rewards_queue_headroom_below_max():
    # Node 3's own framing is a "queue/delay penalty": a queue comfortably under
    # Q_max should score better than one that has overflowed it. The margin used
    # here is (Q_max - Q_k), the mirror image of the CUE-rate term (R_m - R_min) --
    # deliberately NOT the literal "Q_k - Q_max" as OCR'd from Eq. 22, since that
    # form would reward larger queue overflows, which contradicts both the
    # checklist's own description of this term and constraint (19e) (Q_k <= Q_max).
    weights = (0.0, 1.0, 0.0)
    small_queue = compute_reward(
        cue_rates=np.array([]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([10.0]),
        max_queue_length_bits=100.0,
        reliability_margins=np.array([]),
        weights=weights,
        penalty=-10.0,
    )
    big_queue_within_max = compute_reward(
        cue_rates=np.array([]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([90.0]),
        max_queue_length_bits=100.0,
        reliability_margins=np.array([]),
        weights=weights,
        penalty=-10.0,
    )
    assert small_queue > big_queue_within_max > 0


def test_reward_penalizes_queue_overflow():
    weights = (0.0, 1.0, 0.0)
    r = compute_reward(
        cue_rates=np.array([]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([150.0]),
        max_queue_length_bits=100.0,
        reliability_margins=np.array([]),
        weights=weights,
        penalty=-10.0,
    )
    assert r == -10.0


def test_reward_rewards_positive_reliability_margin_and_penalizes_negative():
    weights = (0.0, 0.0, 1.0)
    reliable = compute_reward(
        cue_rates=np.array([]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([]),
        max_queue_length_bits=1.0,
        reliability_margins=np.array([2.0]),
        weights=weights,
        penalty=-10.0,
    )
    unreliable = compute_reward(
        cue_rates=np.array([]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([]),
        max_queue_length_bits=1.0,
        reliability_margins=np.array([-2.0]),
        weights=weights,
        penalty=-10.0,
    )
    assert reliable == 2.0
    assert unreliable == -10.0


def test_reward_combines_all_three_weighted_terms():
    r = compute_reward(
        cue_rates=np.array([2.0e6]),
        cue_min_rate_bps=1.0e6,
        v2v_queues=np.array([10.0]),
        max_queue_length_bits=100.0,
        reliability_margins=np.array([1.0]),
        weights=(0.3, 0.5, 0.2),
        penalty=-10.0,
    )
    expected = 0.3 * 1.0e6 + 0.5 * 90.0 + 0.2 * 1.0
    assert np.isclose(r, expected)
