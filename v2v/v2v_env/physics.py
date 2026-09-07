"""Per-slot link physics: SINR (Eq. 5-6), rate (Eq. 8), queue recursion (Eq. 16),
and the reliability margin behind Eq. 22's third term (Eq. 15).
"""

import numpy as np


def dbm_to_watts(dbm: float) -> float:
    return 10 ** (dbm / 10) * 1e-3


def noise_power_watts(bandwidth_hz: float, noise_psd_dbm_per_hz: float) -> float:
    return dbm_to_watts(noise_psd_dbm_per_hz) * bandwidth_hz


def power_levels_watts(max_power_dbm: float, num_power_levels: int) -> np.ndarray:
    max_power_w = dbm_to_watts(max_power_dbm)
    return np.linspace(0.0, max_power_w, num_power_levels + 1)


def sinr(rx_power_w: float, noise_w: float, interference_w: float) -> float:
    if rx_power_w <= 0.0:
        return 0.0
    return rx_power_w / (noise_w + interference_w)


def rate_bps(bandwidth_hz: float, sinr_linear: float) -> float:
    return bandwidth_hz * np.log2(1.0 + sinr_linear)


def update_queue(
    prev_queue: float, arrival_rate_bps: float, slot_duration_s: float, rate_bps: float
) -> float:
    return max(0.0, prev_queue + slot_duration_s * arrival_rate_bps - slot_duration_s * rate_bps)


def reliability_margin(
    tx_power_w: float,
    alpha: float,
    noise_w: float,
    interference_w: float,
    sinr_threshold_db: float,
    outage_probability_threshold: float,
) -> float:
    lhs = tx_power_w * alpha / (noise_w + interference_w)
    gamma_o_linear = 10 ** (sinr_threshold_db / 10)
    rhs = gamma_o_linear / np.log(1.0 / (1.0 - outage_probability_threshold))
    return lhs - rhs
