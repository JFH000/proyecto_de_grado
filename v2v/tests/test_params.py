from v2v_env.params import V2VEnvParams


def test_default_scope_matches_node1_decision():
    params = V2VEnvParams()
    assert params.num_cues == 15
    assert params.num_v2v_pairs == 5


def test_action_space_size_matches_channels_times_power_levels():
    params = V2VEnvParams(num_cues=15, num_power_levels=3)
    assert params.action_space_size == 15 * 4


def test_state_dim_matches_3m_plus_1():
    params = V2VEnvParams(num_cues=15)
    assert params.state_dim == 3 * 15 + 1


def test_max_queue_length_derived_from_arrival_rate_and_max_delay():
    params = V2VEnvParams(v2v_arrival_rate_bps=2.0e6, max_tolerable_delay_s=0.1)
    assert params.max_queue_length_bits == 2.0e6 * 0.1
