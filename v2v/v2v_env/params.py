"""Environment parameters for the V2V resource-allocation MDP (Li et al. 2022, Sec. IV-A).

Values annotated "Table II" / "Table III" come directly from the paper's simulation
setup. Values annotated "not given by the paper" are free parameters this simplified,
first-pass environment needs but that Sec. IV-A/V never tabulate numerically -- they are
exposed here so they can be tuned without touching the environment code.
"""

from dataclasses import dataclass, field


@dataclass
class V2VEnvParams:
    # --- Scenario scale (Node 1 decision: M=15, K=5, not the full 5-30 sweep) ---
    num_cues: int = 15  # M, Table II
    num_v2v_pairs: int = 5  # K, first point of Table II's {5,10,15,20,25,30} sweep
    num_power_levels: int = 3  # N_p; action offers N_p+1 discrete levels including 0
    max_steps: int = 200  # episode truncation length; not given by the paper (infinite-horizon MDP)

    # --- Radio parameters (Table II) ---
    channel_bandwidth_hz: float = 1.0e6  # W
    cue_tx_power_dbm: float = 23.0  # P_m^t
    v2v_max_tx_power_dbm: float = 23.0  # P_max^v
    noise_psd_dbm_per_hz: float = -174.0  # thermal noise spectral density
    neighbor_radius_m: float = 300.0  # 3GPP TR 36.885 max communication radius
    path_loss_coefficient_db: float = -68.5  # phi; single-law simplification of Eq. 4
    path_loss_exponent: float = 1.61  # e
    sinr_threshold_db: float = 5.0  # gamma_o, Table II

    # --- Geometry (not given by the paper; Node 1 simplifies away LOS/WLOS/NLOS) ---
    cell_radius_m: float = 500.0
    v2v_link_distance_min_m: float = 10.0
    v2v_link_distance_max_m: float = 50.0
    min_distance_m: float = 1.0  # distance floor, avoids blow-up as distance -> 0

    # --- QoS / reward parameters ---
    reward_weights: tuple[float, float, float] = (0.3, 0.5, 0.2)  # (lambda1,2,3), Table III
    penalty_constant: float = -10.0  # A in Eq. 23; not given numerically by the paper
    outage_probability_threshold: float = 0.1  # p_o; not given numerically by the paper
    cue_min_rate_bps: float = 1.0e6  # R_m^min; not given numerically by the paper
    max_tolerable_delay_s: float = 0.1  # D_max = 100 ms, 3GPP TR 36.885
    v2v_arrival_rate_bps: float = 1.0e6  # lambda_k traffic arrival rate; not given numerically
    slot_duration_s: float = 1.0e-3  # tau; not given numerically by the paper

    # --- Observation encoding (not part of the paper's MDP; keeps network inputs near order 1) ---
    channel_gain_ref_db: float = -100.0  # channel coefficients are observed as (dB - ref) / scale
    channel_gain_scale_db: float = 10.0

    # --- Extension of the paper's state (off by default, i.e. the paper's 3M + 1 features) ---
    # One-hot of the channel this pair picked in the last slot, between the neighbor counts and the
    # queue. The paper's counts exclude the pair itself, so without it an agent cannot tell the
    # channel it is using from one nobody uses, and cannot learn to stay on a channel no neighbor took.
    observe_own_channel: bool = False

    # --- Deviation from the paper's reward timing (off by default) ---
    # Eq. 22 judges Q_k^t, the queue at the start of the slot, which this slot's action cannot
    # change: the cost of staying silent shows up one slot later. With this on, the queue term
    # judges the queue the slot leaves, so the slot whose silence overflows the queue pays for it.
    reward_end_of_slot_queue: bool = False

    max_queue_length_bits: float = field(init=False)  # Q_max, derived via Little's law (Eq. 18)

    def __post_init__(self) -> None:
        self.max_queue_length_bits = self.v2v_arrival_rate_bps * self.max_tolerable_delay_s

    @property
    def action_space_size(self) -> int:
        return self.num_cues * (self.num_power_levels + 1)

    @property
    def state_dim(self) -> int:
        return (4 if self.observe_own_channel else 3) * self.num_cues + 1
