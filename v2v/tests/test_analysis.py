import math

from analysis import convergence_episode


def test_convergence_episode_is_where_the_moving_average_enters_the_final_band_for_good():
    # 100 episodes at 0, then 100 at 10: final value 10, band +-0.5. A 10-episode moving
    # average is within 0.5 of 10 only once its whole window is past the jump: episode 109.
    rewards = [0.0] * 100 + [10.0] * 100
    assert convergence_episode(rewards, window=10) == 109


def test_convergence_episode_of_a_flat_curve_is_its_first_full_window():
    assert convergence_episode([5.0] * 100, window=10) == 9


def test_convergence_episode_is_nan_when_the_curve_is_still_moving_at_the_end():
    # A jump in the last 5 episodes: the final 10 % averages 25, the last window 50.
    rewards = [0.0] * 195 + [100.0] * 5
    assert math.isnan(convergence_episode(rewards, window=10))


def test_convergence_episode_handles_negative_rewards():
    rewards = [-100.0] * 50 + [-10.0] * 150
    assert convergence_episode(rewards, window=10) == 59
