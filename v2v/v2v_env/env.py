"""V2V communication environment (Li et al. 2022, Sec. IV-A) as a PettingZoo ParallelEnv.

Each agent is a V2V pair. Per-agent observation is local only (own channel, CUE
channels, one-slot-delayed neighbor selections within 300 m, own queue length) --
no agent ever sees another agent's private state or the true global state. Channel
coefficients are observed in normalized dB (encode_channel_gain) and the queue as a
fraction of Q_max, so the network's inputs stay near order 1. The
reward is a single value computed centrally at each step from the joint action and
broadcast identically to every agent (Eq. 22); see reward.py for the one deliberate
deviation from the literal printed formula (the queue term's sign).
"""

import functools

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from v2v_env import channel, physics
from v2v_env.params import V2VEnvParams
from v2v_env.reward import compute_reward

MIN_CHANNEL_GAIN = 1e-20  # -200 dB floor, so a zero fading draw still has a finite log


def encode_channel_gain(gain: np.ndarray, ref_db: float, scale_db: float) -> np.ndarray:
    """Channel coefficient as an observation feature: dB, centered on ref_db, in units of scale_db."""
    gain_db = 10.0 * np.log10(np.maximum(gain, MIN_CHANNEL_GAIN))
    return (gain_db - ref_db) / scale_db


class V2VEnv(ParallelEnv):
    metadata = {"name": "v2v_env_v0"}

    def __init__(self, params: V2VEnvParams | None = None):
        self.params = params or V2VEnvParams()
        self.possible_agents = [f"v2v_{i}" for i in range(self.params.num_v2v_pairs)]
        self._power_levels_w = physics.power_levels_watts(
            self.params.v2v_max_tx_power_dbm, self.params.num_power_levels
        )
        self._cue_power_w = physics.dbm_to_watts(self.params.cue_tx_power_dbm)
        self._noise_w = physics.noise_power_watts(
            self.params.channel_bandwidth_hz, self.params.noise_psd_dbm_per_hz
        )
        self.agents = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        m = self.params.num_cues
        # Encoded channel gains (first 2M entries) are signed; neighbor counts and queue are not.
        low = np.concatenate([np.full(2 * m, -np.inf), np.zeros(m + 1)])
        return Box(low=low, high=np.inf, shape=(self.params.state_dim,), dtype=np.float64)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.params.action_space_size)

    @property
    def neighbor_mask(self) -> np.ndarray:
        """mask[i, j]: pair j's transmitter is within the neighbor radius of pair i's (K x K,
        symmetric, False on the diagonal). Fixed within an episode; gossip aggregates over it."""
        return self._neighbor_mask

    def _decode_action(self, action_id: int) -> tuple[int, int]:
        num_power_levels = self.params.num_power_levels + 1
        channel_idx = action_id // num_power_levels
        power_level_idx = action_id % num_power_levels
        return channel_idx, power_level_idx

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        positions = channel.sample_positions(self._rng, self.params)
        self._positions = positions

        self._alpha_v2v_own = channel.large_scale_fading(
            channel.distance(positions.v2v_tx, positions.v2v_rx), self.params
        )
        self._alpha_cue_bs = channel.large_scale_fading(
            channel.distance(positions.cue, positions.bs[None, :]), self.params
        )
        self._alpha_v2v_to_bs = channel.large_scale_fading(
            channel.distance(positions.v2v_tx, positions.bs[None, :]), self.params
        )
        self._alpha_cue_to_v2v = channel.large_scale_fading(
            channel.pairwise_distance(positions.cue, positions.v2v_rx), self.params
        )
        self._alpha_v2v_to_v2v = channel.large_scale_fading(
            channel.pairwise_distance(positions.v2v_tx, positions.v2v_rx), self.params
        )
        self._neighbor_mask = channel.neighbor_mask(positions.v2v_tx, self.params.neighbor_radius_m)

        self._queue = np.zeros(self.params.num_v2v_pairs)
        self._prev_channels = np.full(self.params.num_v2v_pairs, -1, dtype=int)

        self._h_own, self._h_cue = self._draw_observation_fading()
        obs = self._build_observations()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def _draw_observation_fading(self):
        g_own = channel.small_scale_fading(
            self._rng, size=(self.params.num_v2v_pairs, self.params.num_cues)
        )
        g_cue = channel.small_scale_fading(self._rng, size=(self.params.num_cues,))
        h_own = self._alpha_v2v_own[:, None] * g_own
        h_cue = self._alpha_cue_bs * g_cue
        return h_own, h_cue

    def _build_observations(self):
        neighbor_counts = channel.neighbor_channel_counts(
            self._neighbor_mask, self._prev_channels, self.params.num_cues
        )
        ref_db, scale_db = self.params.channel_gain_ref_db, self.params.channel_gain_scale_db
        h_own = encode_channel_gain(self._h_own, ref_db, scale_db)
        h_cue = encode_channel_gain(self._h_cue, ref_db, scale_db)
        queue = self._queue / self.params.max_queue_length_bits
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            obs[agent] = np.concatenate(
                [
                    h_own[i],
                    h_cue,
                    neighbor_counts[i].astype(np.float64),
                    [queue[i]],
                ]
            )
        return obs

    def step(self, actions: dict):
        self._step_count += 1
        k = self.params.num_v2v_pairs
        m = self.params.num_cues

        channels = np.zeros(k, dtype=int)
        powers_w = np.zeros(k)
        for i, agent in enumerate(self.possible_agents):
            ch, level = self._decode_action(actions[agent])
            channels[i] = ch
            powers_w[i] = self._power_levels_w[level]

        # The own-link and CUE-to-BS coefficients are the ones in the agents' state for this slot
        # (Li et al. 2022: decisions use instantaneous CSI). The links no agent observes get
        # fresh fading draws.
        g_v2v_to_bs = channel.small_scale_fading(self._rng, size=(k,))
        g_cue_to_v2v = channel.small_scale_fading(self._rng, size=(k,))
        g_v2v_to_v2v = channel.small_scale_fading(self._rng, size=(k, k))

        h_cue_bs = self._h_cue  # (M,)
        h_v2v_to_bs = self._alpha_v2v_to_bs * g_v2v_to_bs  # (K,), at each k's chosen channel

        cue_interference_w = np.zeros(m)
        for i in range(k):
            cue_interference_w[channels[i]] += powers_w[i] * h_v2v_to_bs[i]

        cue_rates = np.zeros(m)
        for mm in range(m):
            s = physics.sinr(self._cue_power_w * h_cue_bs[mm], self._noise_w, cue_interference_w[mm])
            cue_rates[mm] = physics.rate_bps(self.params.channel_bandwidth_hz, s)

        same_channel = channels[:, None] == channels[None, :]
        np.fill_diagonal(same_channel, False)
        co_channel_power_w = same_channel * powers_w[None, :]
        v2v_interference_w = (co_channel_power_w * self._alpha_v2v_to_v2v.T * g_v2v_to_v2v).sum(axis=1)
        # Eq. 15 uses only the large-scale gain of the co-channel V2V interferers.
        v2v_interference_large_scale_w = (co_channel_power_w * self._alpha_v2v_to_v2v.T).sum(axis=1)

        v2v_rates = np.zeros(k)
        reliability_margins = np.zeros(k)
        for i in range(k):
            ch = channels[i]
            h_own_i = self._h_own[i, ch]
            interference_cue_w = self._cue_power_w * self._alpha_cue_to_v2v[ch, i] * g_cue_to_v2v[i]
            s = physics.sinr(powers_w[i] * h_own_i, self._noise_w, interference_cue_w + v2v_interference_w[i])
            v2v_rates[i] = physics.rate_bps(self.params.channel_bandwidth_hz, s) if powers_w[i] > 0 else 0.0
            reliability_margins[i] = physics.reliability_margin(
                powers_w[i],
                self._alpha_v2v_own[i],
                self._noise_w,
                interference_cue_w + v2v_interference_large_scale_w[i],
                self.params.sinr_threshold_db,
                self.params.outage_probability_threshold,
            )

        shared_reward = compute_reward(
            cue_rates=cue_rates,
            cue_min_rate_bps=self.params.cue_min_rate_bps,
            v2v_queues=self._queue,
            max_queue_length_bits=self.params.max_queue_length_bits,
            reliability_margins=reliability_margins,
            reliability_threshold=physics.reliability_threshold(
                self.params.sinr_threshold_db, self.params.outage_probability_threshold
            ),
            weights=self.params.reward_weights,
            penalty=self.params.penalty_constant,
        )

        # QoS metrics for logging, taken from the same quantities the reward judged:
        # delay via Little's law on Q_k^t (so delay > D_max <=> Q_k^t > Q_max), outage where Eq. 15 fails.
        cue_rate_mbps = float(cue_rates.mean() / 1e6)
        v2v_delay_ms = self._queue / self.params.v2v_arrival_rate_bps * 1e3

        # Behaviour metrics: a pair at zero power picks a channel but does not occupy it, so channel
        # sharing only counts pairs that actually transmit.
        transmitting = powers_w > 0
        co_channel = transmitting & (same_channel & transmitting[None, :]).any(axis=1)
        channels_used = int(np.unique(channels[transmitting]).size)
        max_power_w = self._power_levels_w[-1]
        neighbor_counts = self._neighbor_mask.sum(axis=1)

        self._queue = np.array(
            [
                physics.update_queue(
                    self._queue[i], self.params.v2v_arrival_rate_bps, self.params.slot_duration_s, v2v_rates[i]
                )
                for i in range(k)
            ]
        )
        self._prev_channels = channels
        self._h_own, self._h_cue = self._draw_observation_fading()

        observations = self._build_observations()
        rewards = {agent: shared_reward for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncated = self._step_count >= self.params.max_steps
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {
            agent: {
                "channel": int(channels[i]),
                "power_w": float(powers_w[i]),
                "cue_rate_mbps": cue_rate_mbps,
                "v2v_delay_ms": float(v2v_delay_ms[i]),
                "outage": bool(reliability_margins[i] < 0),
                "power_ratio": float(powers_w[i] / max_power_w),
                "max_power": bool(powers_w[i] == max_power_w),
                "co_channel": bool(co_channel[i]),
                "channels_used": channels_used,
                "neighbors": int(neighbor_counts[i]),
            }
            for i, agent in enumerate(self.possible_agents)
        }

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
