"""Channel/geometry model backing the per-agent state (Li et al. 2022, Sec. IV-A/B).

Node 1 of the project checklist deliberately simplifies the paper's LOS/WLOS/NLOS
lane-geometry path-loss model (Eq. 4) down to a single power-law path-loss coefficient,
while keeping the paper's core channel structure: coefficient = large-scale fading
(distance-dependent, ~constant within an episode) x small-scale fading (frequency-
dependent, unit-mean exponential, resampled every slot).
"""

from dataclasses import dataclass

import numpy as np

from v2v_env.params import V2VEnvParams


@dataclass
class Positions:
    bs: np.ndarray  # (2,)
    cue: np.ndarray  # (M, 2)
    v2v_tx: np.ndarray  # (K, 2)
    v2v_rx: np.ndarray  # (K, 2)


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between every point in `a` (N,2) and every point in `b` (M,2) -> (N, M)."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def large_scale_fading(distance_m: np.ndarray, params: V2VEnvParams) -> np.ndarray:
    d = np.maximum(distance_m, params.min_distance_m)
    phi_linear = 10 ** (params.path_loss_coefficient_db / 10)
    return phi_linear * d ** (-params.path_loss_exponent)


def small_scale_fading(rng: np.random.Generator, size) -> np.ndarray:
    return rng.exponential(scale=1.0, size=size)


def sample_positions(rng: np.random.Generator, params: V2VEnvParams) -> Positions:
    bs = np.zeros(2)

    def uniform_in_disc(n: int) -> np.ndarray:
        radius = params.cell_radius_m * np.sqrt(rng.uniform(0.0, 1.0, size=n))
        angle = rng.uniform(0.0, 2 * np.pi, size=n)
        return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)

    cue = uniform_in_disc(params.num_cues)
    v2v_tx = uniform_in_disc(params.num_v2v_pairs)

    link_distance = rng.uniform(
        params.v2v_link_distance_min_m,
        params.v2v_link_distance_max_m,
        size=params.num_v2v_pairs,
    )
    link_angle = rng.uniform(0.0, 2 * np.pi, size=params.num_v2v_pairs)
    offset = np.stack(
        [link_distance * np.cos(link_angle), link_distance * np.sin(link_angle)], axis=-1
    )
    v2v_rx = v2v_tx + offset

    return Positions(bs=bs, cue=cue, v2v_tx=v2v_tx, v2v_rx=v2v_rx)


def neighbor_mask(v2v_tx: np.ndarray, radius_m: float) -> np.ndarray:
    k = v2v_tx.shape[0]
    diffs = v2v_tx[:, None, :] - v2v_tx[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    mask = dists <= radius_m
    np.fill_diagonal(mask, False)
    return mask


def neighbor_channel_counts(
    mask: np.ndarray, prev_channels: np.ndarray, num_channels: int
) -> np.ndarray:
    k = mask.shape[0]
    counts = np.zeros((k, num_channels), dtype=int)
    for k_prime in range(k):
        channel = prev_channels[k_prime]
        if channel < 0:
            continue
        counts[:, channel] += mask[:, k_prime]
    return counts
