import json
import math

import pandas as pd

from analysis import (
    convergence_episode,
    final_values,
    load_metadata,
    load_runs,
    per_episode_means,
    summarize_by_k,
)


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


def write_run(root, k, method, seed, rows, metadata=None):
    path = root / f"k{k}" / f"{method}_seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata))


def test_load_runs_tags_each_row_with_method_k_and_seed(tmp_path):
    write_run(tmp_path, 5, "greedy", 0, [{"episode": 0, "agent": "v2v_0", "reward": 1.0}])
    write_run(tmp_path, 10, "greedy", 2, [{"episode": 0, "agent": "v2v_0", "reward": 2.0}])
    write_run(tmp_path, 10, "fedavg", 0, [{"episode": 0, "agent": "v2v_0", "reward": 3.0}])
    (tmp_path / "k10" / "notas.csv").write_text("x\n1\n")
    runs = load_runs(tmp_path, methods=["greedy"])
    assert sorted(zip(runs["method"], runs["k"], runs["seed"], runs["reward"])) == [
        ("greedy", 5, 0, 1.0),
        ("greedy", 10, 2, 2.0),
    ]


def test_per_episode_means_averages_over_agents_and_counts_isolated_pairs():
    runs = pd.DataFrame(
        {
            "method": ["greedy"] * 2,
            "k": [2, 2],
            "seed": [0, 0],
            "episode": [0, 0],
            "agent": ["v2v_0", "v2v_1"],
            "outage_rate": [1.0, 0.0],
            "neighbors": [0.0, 2.0],
        }
    )
    (row,) = per_episode_means(runs).to_dict("records")
    assert row["outage_rate"] == 0.5
    assert row["neighbors"] == 1.0
    assert row["isolated_rate"] == 0.5


def test_final_values_average_the_last_fraction_of_each_runs_episodes():
    per_episode = pd.DataFrame(
        {"method": "greedy", "k": 5, "seed": 0, "episode": range(10), "reward": [float(e) for e in range(10)]}
    )
    final = final_values(per_episode, final_fraction=0.2)
    assert final.loc[("greedy", 5, 0), "reward"] == 8.5  # episodes 8 and 9


def test_summarize_by_k_gives_mean_and_std_across_seeds():
    final = pd.DataFrame(
        {"method": "greedy", "k": 5, "seed": [0, 1], "reward": [1.0, 3.0]}
    ).set_index(["method", "k", "seed"])
    summary = summarize_by_k(final)
    assert summary.loc[("greedy", 5), ("reward", "mean")] == 2.0
    assert math.isclose(summary.loc[("greedy", 5), ("reward", "std")], math.sqrt(2))
    assert summary.loc[("greedy", 5), ("seeds", "")] == 2


def test_load_metadata_reads_the_json_next_to_each_csv(tmp_path):
    write_run(
        tmp_path, 30, "greedy", 1, [{"episode": 0}], metadata={"method": "greedy", "seed": 1, "wall_time_s": 60.0}
    )
    (meta,) = load_metadata(tmp_path, methods=["greedy"]).to_dict("records")
    assert (meta["method"], meta["k"], meta["seed"], meta["wall_time_s"]) == ("greedy", 30, 1, 60.0)
