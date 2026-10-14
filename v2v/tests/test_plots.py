import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots import fedavg_models_per_episode, log_y, plot_vs_k


def make_summary(rows: dict[str, list[float]]) -> pd.DataFrame:
    """Summary shaped like analysis.summarize_by_k for one metric, K = 5 and 10."""
    index = pd.MultiIndex.from_tuples([(m, k) for m in rows for k in (5, 10)], names=["method", "k"])
    means = [v for values in rows.values() for v in values]
    return pd.DataFrame({("cue_rate_mbps", "mean"): means, ("cue_rate_mbps", "std"): [0.1] * len(means)}, index=index)


def test_plot_vs_k_draws_one_errorbar_series_per_method():
    summary = make_summary({"fedavg": [7.0, 6.0], "greedy": [5.0, 4.0]})
    fig, ax = plt.subplots()
    plot_vs_k(ax, summary, "cue_rate_mbps", ["fedavg", "greedy"])
    assert [c.get_label() for c in ax.containers] == ["FedAvg", "Greedy"]
    plt.close(fig)


def test_plot_vs_k_skips_a_method_without_values():
    summary = make_summary({"fedavg": [7.0, 6.0], "greedy": [math.nan, math.nan]})
    fig, ax = plt.subplots()
    plot_vs_k(ax, summary, "cue_rate_mbps", ["fedavg", "greedy"])
    assert [c.get_label() for c in ax.containers] == ["FedAvg"]
    plt.close(fig)


def test_fedavg_models_per_episode_is_2k_per_round_times_rounds():
    # 200-slot episodes aggregated every 50 slots: 4 rounds of 2K transfers.
    assert fedavg_models_per_episode(5) == 40
    assert fedavg_models_per_episode(30, slots_per_episode=200, aggregate_every=50) == 240


def test_log_y_uses_a_log_axis_with_plain_labels_when_data_span_decades():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.3, 40.0])
    log_y(ax)
    assert ax.get_yscale() == "log"
    assert ax.yaxis.get_major_formatter()(0.5, 0) == "0.5"
    plt.close(fig)


def test_log_y_keeps_a_linear_axis_within_one_decade():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.25, 0.3])
    log_y(ax)
    assert ax.get_yscale() == "linear"
    plt.close(fig)
