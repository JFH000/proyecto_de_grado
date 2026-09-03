import numpy as np

from v2v_env.channel import (
    large_scale_fading,
    small_scale_fading,
    sample_positions,
    distance,
    pairwise_distance,
    neighbor_mask,
    neighbor_channel_counts,
)
from v2v_env.params import V2VEnvParams


def test_large_scale_fading_decreases_with_distance():
    params = V2VEnvParams()
    near = large_scale_fading(np.array([10.0]), params)[0]
    far = large_scale_fading(np.array([200.0]), params)[0]
    assert near > far > 0


def test_large_scale_fading_matches_path_loss_formula():
    params = V2VEnvParams(path_loss_coefficient_db=-68.5, path_loss_exponent=1.61)
    d = np.array([50.0])
    expected = (10 ** (-68.5 / 10)) * 50.0 ** (-1.61)
    assert np.isclose(large_scale_fading(d, params)[0], expected)


def test_large_scale_fading_floors_distance_to_avoid_blowup():
    params = V2VEnvParams(min_distance_m=1.0)
    at_zero = large_scale_fading(np.array([0.0]), params)[0]
    at_floor = large_scale_fading(np.array([1.0]), params)[0]
    assert at_zero == at_floor


def test_small_scale_fading_is_nonnegative():
    rng = np.random.default_rng(0)
    g = small_scale_fading(rng, size=(5, 15))
    assert np.all(g >= 0.0)
    assert g.shape == (5, 15)


def test_small_scale_fading_has_unit_mean_over_many_samples():
    rng = np.random.default_rng(0)
    g = small_scale_fading(rng, size=(20000,))
    assert np.isclose(g.mean(), 1.0, atol=0.05)


def test_sample_positions_returns_expected_shapes():
    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5)
    rng = np.random.default_rng(0)
    positions = sample_positions(rng, params)
    assert positions.bs.shape == (2,)
    assert positions.cue.shape == (15, 2)
    assert positions.v2v_tx.shape == (5, 2)
    assert positions.v2v_rx.shape == (5, 2)


def test_sample_positions_is_reproducible_with_seeded_rng():
    params = V2VEnvParams()
    positions_a = sample_positions(np.random.default_rng(42), params)
    positions_b = sample_positions(np.random.default_rng(42), params)
    assert np.array_equal(positions_a.cue, positions_b.cue)


def test_v2v_link_distance_within_configured_range():
    params = V2VEnvParams(v2v_link_distance_min_m=10.0, v2v_link_distance_max_m=50.0)
    rng = np.random.default_rng(0)
    positions = sample_positions(rng, params)
    d = distance(positions.v2v_tx, positions.v2v_rx)
    assert np.all(d >= 10.0 - 1e-6)
    assert np.all(d <= 50.0 + 1e-6)


def test_pairwise_distance_returns_matrix_of_shape_len_a_by_len_b():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 10.0], [3.0, 4.0]])
    d = pairwise_distance(a, b)
    assert d.shape == (2, 3)
    assert np.isclose(d[0, 0], 0.0)
    assert np.isclose(d[0, 2], 5.0)
    assert np.isclose(d[1, 1], np.hypot(10.0, 10.0))


def test_neighbor_mask_excludes_self():
    tx = np.array([[0.0, 0.0], [1.0, 0.0], [1000.0, 0.0]])
    mask = neighbor_mask(tx, radius_m=300.0)
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert not mask[2, 2]


def test_neighbor_mask_respects_radius():
    tx = np.array([[0.0, 0.0], [1.0, 0.0], [1000.0, 0.0]])
    mask = neighbor_mask(tx, radius_m=300.0)
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2] and not mask[2, 0]


def test_neighbor_channel_counts_counts_matching_channel_neighbors():
    mask = np.array(
        [
            [False, True, True],
            [True, False, False],
            [True, False, False],
        ]
    )
    prev_channels = np.array([0, 0, 2])
    counts = neighbor_channel_counts(mask, prev_channels, num_channels=3)
    # agent 0's neighbors are 1 and 2, who picked channels 0 and 2 respectively
    assert counts[0, 0] == 1
    assert counts[0, 2] == 1
    assert counts[0, 1] == 0
    # agent 1's only neighbor is 0, who picked channel 0
    assert counts[1, 0] == 1
    assert counts[1].sum() == 1
