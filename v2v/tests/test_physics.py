import numpy as np

from v2v_env.physics import (
    dbm_to_watts,
    noise_power_watts,
    power_levels_watts,
    sinr,
    rate_bps,
    update_queue,
    reliability_margin,
)


def test_dbm_to_watts_known_value():
    # 30 dBm == 1 W
    assert np.isclose(dbm_to_watts(30.0), 1.0)


def test_noise_power_scales_with_bandwidth():
    small_bw = noise_power_watts(bandwidth_hz=1.0e6, noise_psd_dbm_per_hz=-174.0)
    big_bw = noise_power_watts(bandwidth_hz=2.0e6, noise_psd_dbm_per_hz=-174.0)
    assert np.isclose(big_bw, 2 * small_bw)


def test_power_levels_watts_includes_zero_and_max():
    levels = power_levels_watts(max_power_dbm=23.0, num_power_levels=3)
    assert len(levels) == 4
    assert levels[0] == 0.0
    assert np.isclose(levels[-1], dbm_to_watts(23.0))


def test_power_levels_watts_are_evenly_spaced():
    levels = power_levels_watts(max_power_dbm=23.0, num_power_levels=2)
    assert np.isclose(levels[1], levels[2] / 2)


def test_sinr_matches_definition():
    result = sinr(rx_power_w=4.0, noise_w=1.0, interference_w=1.0)
    assert np.isclose(result, 4.0 / (1.0 + 1.0))


def test_sinr_zero_when_rx_power_zero():
    assert sinr(rx_power_w=0.0, noise_w=1.0, interference_w=0.0) == 0.0


def test_rate_bps_matches_shannon_formula():
    result = rate_bps(bandwidth_hz=1.0e6, sinr_linear=3.0)
    assert np.isclose(result, 1.0e6 * np.log2(4.0))


def test_update_queue_grows_when_no_data_sent():
    q = update_queue(prev_queue=0.0, arrival_rate_bps=1000.0, slot_duration_s=0.01, rate_bps=0.0)
    assert np.isclose(q, 10.0)


def test_update_queue_floors_at_zero():
    q = update_queue(prev_queue=0.0, arrival_rate_bps=0.0, slot_duration_s=0.01, rate_bps=1000.0)
    assert q == 0.0


def test_update_queue_shrinks_when_rate_exceeds_arrivals():
    q = update_queue(prev_queue=100.0, arrival_rate_bps=1000.0, slot_duration_s=0.01, rate_bps=5000.0)
    assert np.isclose(q, 100.0 + 10.0 - 50.0)


def test_reliability_margin_positive_when_constraint_satisfied():
    # LHS of Eq. 15 comfortably above the threshold
    margin = reliability_margin(
        tx_power_w=1.0,
        alpha=1.0,
        noise_w=0.01,
        interference_w=0.0,
        sinr_threshold_db=0.0,  # gamma_o = 1 (linear)
        outage_probability_threshold=0.5,  # ln(1/0.5) = ln 2
    )
    lhs = 1.0 * 1.0 / 0.01
    rhs = 1.0 / np.log(2.0)
    assert np.isclose(margin, lhs - rhs)


def test_reliability_margin_negative_when_signal_weak():
    margin = reliability_margin(
        tx_power_w=1e-6,
        alpha=1e-6,
        noise_w=1.0,
        interference_w=1.0,
        sinr_threshold_db=10.0,
        outage_probability_threshold=0.1,
    )
    assert margin < 0
