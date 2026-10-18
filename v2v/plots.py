"""Style and plot helpers shared by the analysis notebooks, so every figure of the thesis
uses the same colors, labels and reference values.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Fixed color per method (validated categorical palette, positions 1-4). Greedy is dashed
# because it does not learn.
METHODS = {
    "fedavg": {"label": "FedAvg", "color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "dte": {"label": "DTE (sin federar)", "color": "#eb6834", "linestyle": "-", "marker": "s"},
    "greedy": {"label": "Greedy", "color": "#1baf7a", "linestyle": "--", "marker": "^"},
    "gossip": {"label": "Gossip", "color": "#eda100", "linestyle": "-", "marker": "D"},
    # Not a method of the paper: the threshold rule of agents/threshold_policy.py, dashed like Greedy.
    "umbral": {"label": "Regla de umbral", "color": "#8a4fd6", "linestyle": "--", "marker": "v"},
}
LABELS = {
    "reward": "Recompensa por episodio",
    "cue_rate_mbps": "Tasa CUE (Mbps)",
    "v2v_delay_ms": "Delay V2V (ms)",
    "outage_rate": "Outage (%)",
    "co_channel_rate": "Canal compartido (%)",
    "channels_used": "Canales en uso",
    "mean_power_ratio": "Potencia media / Pmax",
    "max_power_rate": "Slots a Pmax (%)",
    "neighbors": "Vecinos a 300 m o menos",
    "isolated_rate": "Pares sin vecinos (%)",
    "train_loss": "Perdida D3QN",
    "weight_dispersion": "Dispersion de pesos",
    "comm_models": "Modelos transmitidos por episodio",
    "comm_mb": "MB transmitidos por episodio",
    "comm_max_load": "Carga del nodo mas cargado",
    "convergence_episode": "Episodios hasta converger",
}
PERCENT = ["outage_rate", "co_channel_rate", "max_power_rate", "isolated_rate"]  # fractions of slots, shown in %
EXPECTED_KS = [5, 10, 15, 20, 25, 30]
EXPECTED_SEEDS = [0, 1, 2]

# CUE rate in Li et al. (2022), Fig. 3, read off the figure (approximate): for trends only.
LI_FIG3_MBPS = {
    "greedy": {5: 5.75, 10: 5.25, 15: 4.75, 20: 4.0, 25: 3.0, 30: 1.75},
    "fedavg": {5: 7.95, 10: 7.45, 15: 6.95, 20: 6.45, 25: 5.7, 30: 4.7},
}

TEXT, TEXT_2, GRID, REFERENCE = "#0b0b0b", "#52514e", "#e6e6e3", "#8a8985"


def use_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.edgecolor": TEXT_2,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 9.5,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def method_label(method: str) -> str:
    return METHODS.get(method, {}).get("label", method)


def k_colors(ks: Sequence[int]) -> dict[int, tuple]:
    """One color per K, in order along viridis, so consecutive values stay distinguishable."""
    return dict(zip(ks, plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))))


def plot_vs_k(ax, summary: pd.DataFrame, metric: str, methods: Sequence[str]) -> None:
    """Mean ± 1 std across seeds against K, one series per method (from analysis.summarize_by_k).
    Methods without values for the metric (e.g. Greedy has no dispersion) are skipped."""
    for method in methods:
        if (metric, "mean") not in summary or method not in summary.index.get_level_values("method"):
            continue
        s = summary.loc[method]
        if s[(metric, "mean")].isna().all():
            continue
        style = METHODS[method]
        ax.errorbar(
            s.index,
            s[(metric, "mean")],
            yerr=s[(metric, "std")].fillna(0.0),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.5,
            capsize=3,
            label=style["label"],
        )
    ax.set_xticks(EXPECTED_KS)
    ax.set_xlabel("K (pares V2V)")
    ax.set_title(LABELS.get(metric, metric))


def log_y(ax) -> None:
    """Log y axis, only when the panel's data span more than one order of magnitude (dispersion
    and loss do across methods and episodes), labeled 0.2, 0.5, 1, 2, 5... instead of powers of
    ten. Call it after plotting: it reads the range of the plotted data."""
    low, high = ax.dataLim.intervaly
    if low <= 0 or high / low < 10:
        return
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fedavg_models_per_episode(k: int, slots_per_episode: int = 200, aggregate_every: int = 50) -> int:
    """Model transfers FedAvg should log per episode once every agent learns: 2K per round."""
    return 2 * k * (slots_per_episode // aggregate_every)


def gossip_models_per_episode(
    k: int, mean_neighbors: float, slots_per_episode: int = 200, aggregate_every: int = 50
) -> float:
    """Model transfers gossip should log per episode when every agent pulls from all its neighbors:
    one per neighbor per round, K x (mean neighbors) per round."""
    return k * mean_neighbors * (slots_per_episode // aggregate_every)
