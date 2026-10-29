"""Generates the result figures (figures/res_*.pdf) and tables (tables/res_*.tex) of the thesis from the
runs in v2v/runs/, with the same analysis functions that the notebooks use (v2v/analysis.py). This way,
every number in the document comes from the CSV files and is never copied by hand.

Usage, from the root of the repository:
    v2v/.venv/Scripts/python.exe thesis-template/scripts/generate_results.py

Sweeps it reads (2000 episodes, seeds 0, 1 and 2, K = 5, 10, ..., 30):
- runs/propia: each agent learns from its individual reward (the main configuration).
- runs/paper: all agents learn from the shared reward of Li et al. (2022), Eq. 22.
- runs/propia-vecinos2 and runs/propia-vecinos3: gossip with at most 2 or 3 neighbors per round.
- runs/diag/<variant>: diagnosis with K = 10 and seed 0 (FedAvg).
In all of them, the `reward` column is the shared reward of Eq. 22, so every method is measured the same way.

The episode means are cached in scripts/_cache/; delete that folder when the runs change.
"""

import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2V = ROOT / "v2v"
RUNS = V2V / "runs"
THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"
TAB = THESIS / "tables"
CACHE = THESIS / "scripts" / "_cache"
sys.path.insert(0, str(V2V))

from analysis import (  # noqa: E402
    convergence_episode,
    episodes_to_final_gain,
    final_values,
    load_runs,
    per_episode_means,
)
from plots import LI_FIG3_MBPS, METHODS, REFERENCE, use_thesis_style  # noqa: E402

KS = [5, 10, 15, 20, 25, 30]
LEARNERS = ["dte", "fedavg", "gossip"]
ORDER = ["umbral", "greedy", "dte", "fedavg", "gossip"]
FLOOR_EPISODE = 1200  # epsilon reaches its floor at 60 % of the 2000 episodes
WIDTH = 5.33  # text width of the thesis in inches (385 pt)

STYLE = {m: dict(v) for m, v in METHODS.items()}
STYLE["umbral"]["label"] = "Threshold rule"
STYLE["dte"]["label"] = "DTE (no sharing)"
STYLE["gossip"]["label"] = "Gossip"
STYLE["gossip3"] = {"label": "Gossip, $S=3$", "color": "#b87400", "linestyle": "-.", "marker": "h"}
STYLE["gossip2"] = {"label": "Gossip, $S=2$", "color": "#7a4d00", "linestyle": ":", "marker": "p"}
LABEL_TEX = {
    "umbral": "Threshold",
    "greedy": "Greedy",
    "dte": "DTE",
    "fedavg": "FedAvg",
    "gossip": "Gossip",
    "gossip3": "Gossip $S{=}3$",
    "gossip2": "Gossip $S{=}2$",
}
K_LABEL = "$K$ (V2V pairs)"
REWARD_LABEL = "Reward per episode"


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def load_sweep(root: Path, methods, rename=None) -> pd.DataFrame:
    frames = []
    for m in methods:
        per_ep = per_episode_means(load_runs(root, methods=[m]))
        if rename:
            per_ep["method"] = per_ep["method"].replace(rename)
        frames.append(per_ep)
    return pd.concat(frames, ignore_index=True)


def load_all() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "episodes.pkl"
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())
    data = {
        "propia": load_sweep(RUNS / "propia", ORDER),
        "paper": load_sweep(RUNS / "paper", ORDER),
        "vecinos2": load_sweep(RUNS / "propia-vecinos2", ["gossip"], {"gossip": "gossip2"}),
        "vecinos3": load_sweep(RUNS / "propia-vecinos3", ["gossip"], {"gossip": "gossip3"}),
        "diag": {
            d.name: per_episode_means(load_runs(d))
            for d in sorted((RUNS / "diag").iterdir())
            if d.is_dir()
        },
    }
    cache_file.write_bytes(pickle.dumps(data))
    return data


# ----------------------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------------------
def num(x: float, decimals: int = 0) -> str:
    """Number for LaTeX: typographic minus sign and a thousands separator only with 5 digits or more."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) >= 5:
        groups = []
        while whole:
            groups.insert(0, whole[-3:])
            whole = whole[:-3]
        whole = ",".join(groups)
    s = whole + ("." + frac if frac else "")
    if round(x, decimals) < 0:
        s = "$-$" + s
    return s


def pm(mean: float, std: float, decimals: int = 0) -> str:
    return f"{num(mean, decimals)} $\\pm$ {num(std, decimals)}"


def signed(x: float, decimals: int = 0) -> str:
    return ("$+$" if round(x, decimals) > 0 else "") + num(x, decimals)


def write_table(name: str, body: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"res_{name}.tex").write_text(
        "% Generated by scripts/generate_results.py from v2v/runs/. Do not edit by hand.\n" + body,
        encoding="utf-8",
    )


# ----------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------
def finals(per_episode: pd.DataFrame) -> pd.DataFrame:
    return final_values(per_episode)  # index (method, k, seed): mean of the last 10 %


def mean_std(final: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = final[metric].groupby(["k", "method"])
    return g.mean().unstack(), g.std().unstack()


def forgetting(per_episode: pd.DataFrame) -> pd.Series:
    """Best block of 200 episodes minus the final block (the last 200), per run."""
    by_run = per_episode.set_index("episode").groupby(["method", "k", "seed"])["reward"]
    return by_run.apply(lambda r: r.groupby(r.index // 200).mean().max() - r.iloc[-200:].mean())


def per_run_metric(per_episode: pd.DataFrame, fn) -> pd.Series:
    runs = per_episode.sort_values("episode").groupby(["method", "k", "seed"])["reward"]
    return runs.apply(lambda r: fn(r.to_numpy()))


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def style_axes(ax, xlabel=K_LABEL, ks=True):
    if ks:
        ax.set_xticks(KS)
    ax.set_xlabel(xlabel)


def errorbar(ax, x, mean, std, method, **kw):
    st = STYLE[method]
    opts = dict(color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=4,
                capsize=2.5, linewidth=1.4, label=st["label"])
    opts.update(kw)
    ax.errorbar(x, mean, yerr=std, **opts)


def fig_reward_vs_k(final, path, methods=ORDER, title=None):
    mean, std = mean_std(final, "reward")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    for m in methods:
        errorbar(ax, mean.index, mean[m], std[m], m)
    ax.axhline(0, color=REFERENCE, linewidth=0.6)
    style_axes(ax)
    ax.set_ylabel(REWARD_LABEL)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_learning_curves(per_episode, final, path, ks=(10, 15, 20, 25)):
    curve = per_episode.groupby(["method", "k", "episode"])["reward"].mean()
    greedy = mean_std(final, "reward")[0]["greedy"]
    fig, axes = plt.subplots(1, len(ks), figsize=(WIDTH, 2.25), sharex=True)
    for ax, k in zip(axes, ks):
        for m in LEARNERS:
            s = curve.loc[(m, k)].rolling(50, min_periods=1).mean()
            ax.plot(s.index, s.values, color=STYLE[m]["color"], linewidth=1.0, label=STYLE[m]["label"])
        ax.axhline(greedy.loc[k], color=STYLE["greedy"]["color"], linestyle="--", linewidth=1.0,
                   label="Greedy")
        ax.axvline(FLOOR_EPISODE, color=REFERENCE, linestyle=":", linewidth=0.9)
        ax.set_title(f"$K = {k}$")
        ax.set_xlabel("Episode")
        ax.set_xticks([0, 1000, 2000])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(REWARD_LABEL)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.09), fontsize=7.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def fig_individual_vs_shared(final_p, final_s, path):
    mp, sp = mean_std(final_p, "reward")
    ms, ss = mean_std(final_s, "reward")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True)
    for ax, m in zip(axes, LEARNERS):
        errorbar(ax, mp.index, mp[m], sp[m], m, label="Individual reward")
        errorbar(ax, ms.index, ms[m], ss[m], m, linestyle=":", fillstyle="none", label="Shared reward")
        ax.plot(mp.index, mp["greedy"], color=REFERENCE, linestyle="--", marker="^", markersize=3,
                linewidth=1.0, label="Greedy")
        ax.axhline(0, color=REFERENCE, linewidth=0.5)
        ax.set_title(STYLE[m]["label"])
        style_axes(ax)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(REWARD_LABEL)
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(path)
    plt.close(fig)


def fig_qos(final, path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
    metrics = [("cue_rate_mbps", "CUE rate (Mbps)", 1.0),
               ("outage_rate", "Outage (% of slots)", 100.0),
               ("v2v_delay_ms", "V2V delay (ms)", 1.0)]
    for ax, (metric, title, scale) in zip(axes, metrics):
        mean, std = mean_std(final, metric)
        for m in ORDER:
            errorbar(ax, mean.index, mean[m] * scale, std[m] * scale, m, markersize=3, capsize=2,
                     linewidth=1.1)
        if metric == "cue_rate_mbps":
            for ref, marker in (("fedavg", "x"), ("greedy", "+")):
                pts = LI_FIG3_MBPS[ref]
                ax.plot(list(pts), list(pts.values()), linestyle="none", marker=marker, color="#555555",
                        markersize=5, label=f"Li et al., {STYLE[ref]['label']}")
        ax.set_title(title)
        style_axes(ax)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.17), fontsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def fig_behavior(final, path):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    for ax, (metric, title, scale) in zip(
        axes, [("channels_used", "Channels in use per slot", 1.0),
               ("mean_power_ratio", "Mean power / maximum power", 1.0)]):
        mean, std = mean_std(final, metric)
        for m in ORDER:
            errorbar(ax, mean.index, mean[m] * scale, std[m] * scale, m, markersize=3, capsize=2,
                     linewidth=1.1)
        ax.set_title(title)
        style_axes(ax)
        ax.tick_params(labelsize=7)
    axes[0].plot(KS, [min(k, 15) for k in KS], color=REFERENCE, linewidth=0.8, linestyle="-",
                 label="Largest possible, min$(K, 15)$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(path)
    plt.close(fig)


def fig_comm(final_all, path):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    methods = ["fedavg", "gossip", "gossip3", "gossip2"]
    for ax, (metric, title) in zip(
        axes, [("comm_models", "Models transferred per episode"),
               ("comm_max_load", "Transfers of the busiest node")]):
        mean, std = mean_std(final_all, metric)
        for m in methods:
            errorbar(ax, mean.index, mean[m], std[m], m, markersize=3.5, capsize=2, linewidth=1.2)
        ax.set_title(title)
        style_axes(ax)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(path)
    plt.close(fig)


def fig_gossip_neighbors_reward(final_all, path):
    mean, std = mean_std(final_all, "reward")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    for m in ["dte", "fedavg", "gossip", "gossip3", "gossip2"]:
        errorbar(ax, mean.index, mean[m], std[m], m)
    ax.axhline(0, color=REFERENCE, linewidth=0.6)
    style_axes(ax)
    ax.set_ylabel(REWARD_LABEL)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_topology(final, path):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1))
    g = final.loc["greedy"]
    for ax, (metric, title, scale) in zip(
        axes, [("neighbors", "Neighbors within 300 m (mean)", 1.0),
               ("isolated_rate", "Pairs with no neighbor (%)", 100.0)]):
        stats = (g[metric] * scale).groupby("k").agg(["mean", "std"])
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], color="#444444", marker="o",
                    markersize=3.5, capsize=2, linewidth=1.2)
        ax.set_title(title)
        style_axes(ax)
        ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path)
    plt.close(fig)


def fig_dispersion(final_all, path):
    mean, std = mean_std(final_all, "weight_dispersion")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    for m in ["dte", "fedavg", "gossip", "gossip3", "gossip2"]:
        errorbar(ax, mean.index, mean[m], std[m], m, markersize=3.5)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axes(ax)
    ax.set_ylabel("Weight dispersion")
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(speed, path):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    for m in LEARNERS:
        st = STYLE[m]
        ax.plot(speed.index, speed[m], color=st["color"], marker=st["marker"], markersize=4,
                linewidth=1.4, label=st["label"])
    ax.axhline(FLOOR_EPISODE, color=REFERENCE, linestyle=":", linewidth=1.0,
               label="End of exploration (episode 1200)")
    style_axes(ax)
    ax.set_ylabel("Episode at which it reaches\n90 % of its improvement")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


DIAG_SHOWN = [
    ("base", "Shared, $\\gamma=0.99$ (initial configuration)", "#9a9a96", "-"),
    ("gamma0.9", "Shared, $\\gamma=0.9$", "#2a78d6", "-"),
    ("cola-fin-gamma0.9", "Shared, $\\gamma=0.9$, queue at the end of the slot", "#6aa7ec", "--"),
    ("local-gamma0.9", "Individual, $\\gamma=0.9$", "#c47d00", "--"),
    ("local-gamma0.9-cola-fin", "Individual, $\\gamma=0.9$, queue at the end of the slot (chosen)", "#eda100", "-"),
]


def fig_diagnosis(diag, ref_greedy, ref_umbral, path):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    for name, label, color, ls in DIAG_SHOWN:
        s = diag[name].sort_values("episode").set_index("episode")["reward"].rolling(50, min_periods=1).mean()
        ax.plot(s.index, s.values, color=color, linestyle=ls, linewidth=1.2, label=label)
    ax.axhline(ref_greedy, color=STYLE["greedy"]["color"], linestyle="--", linewidth=1.0, label="Greedy")
    ax.axhline(ref_umbral, color=STYLE["umbral"]["color"], linestyle="--", linewidth=1.0,
               label="Threshold rule")
    ax.axvline(FLOOR_EPISODE, color=REFERENCE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel(REWARD_LABEL)
    ax.legend(fontsize=6.8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------------------------
def main() -> None:
    use_thesis_style()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    FIG.mkdir(parents=True, exist_ok=True)
    data = load_all()
    ep_p, ep_s = data["propia"], data["paper"]
    fin_p, fin_s = finals(ep_p), finals(ep_s)
    fin_v = pd.concat([fin_p, finals(data["vecinos2"]), finals(data["vecinos3"])])
    numbers = {}

    seeds = fin_p.groupby(["method", "k"]).size()
    assert (seeds == 3).all(), seeds
    assert (fin_s.groupby(["method", "k"]).size() == 3).all()

    # ---- Figures --------------------------------------------------------------------------------
    fig_reward_vs_k(fin_p, FIG / "res_reward_vs_k.pdf")
    fig_learning_curves(ep_p, fin_p, FIG / "res_curves_individual.pdf")
    fig_learning_curves(ep_s, fin_s, FIG / "res_curves_shared.pdf")
    fig_individual_vs_shared(fin_p, fin_s, FIG / "res_individual_vs_shared.pdf")
    fig_qos(fin_p, FIG / "res_qos.pdf")
    fig_behavior(fin_p, FIG / "res_behavior.pdf")
    fig_comm(fin_v, FIG / "res_communication.pdf")
    fig_gossip_neighbors_reward(fin_v, FIG / "res_gossip_neighbors.pdf")
    fig_topology(fin_p, FIG / "res_topology.pdf")
    fig_dispersion(fin_v, FIG / "res_dispersion.pdf")

    # ---- Table: reward per method and K (individual) --------------------------------------------
    mean, std = mean_std(fin_p, "reward")
    rows = []
    for k in KS:
        best = mean.loc[k, LEARNERS].idxmax()
        cells = []
        for m in ORDER:
            c = pm(mean.loc[k, m], std.loc[k, m])
            if m == best:
                c = r"\textbf{" + c + "}"
            cells.append(c)
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    write_table("reward_individual", "\n".join(rows) + "\n")
    numbers["reward_propia_mean"] = mean.round(1).to_dict()
    numbers["reward_propia_std"] = std.round(1).to_dict()

    # ---- Table: gossip against each method, seed by seed (individual and shared) ----------------
    def wins(final, others):
        rewards = final["reward"].unstack("method")
        out = {}
        for o in others:
            d = (rewards["gossip"] - rewards[o]).groupby("k")
            out[o] = (d.mean(), d.apply(lambda x: int((x > 0).sum())))
        return out

    w = wins(fin_p, ["fedavg", "dte", "greedy"])
    ws = wins(fin_s, ["fedavg", "dte", "greedy"])
    rows = []
    for k in KS:
        cells = []
        for o in ["fedavg", "dte", "greedy"]:
            cells += [signed(w[o][0].loc[k]), f"{w[o][1].loc[k]}/3"]
        cells += [signed(ws["fedavg"][0].loc[k]), f"{ws['fedavg'][1].loc[k]}/3"]
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    write_table("gossip_wins", "\n".join(rows) + "\n")
    numbers["wins_propia"] = {o: {"diff": w[o][0].round(0).to_dict(), "wins": w[o][1].to_dict()} for o in w}
    numbers["wins_paper"] = {o: {"diff": ws[o][0].round(0).to_dict(), "wins": ws[o][1].to_dict()} for o in ws}

    # ---- Table: shared reward -> individual reward ----------------------------------------------
    ms_, ss_ = mean_std(fin_s, "reward")
    gain_rows = []
    rp = fin_p["reward"].unstack("method")
    rs = fin_s["reward"].unstack("method")
    for k in KS:
        cells = []
        for m in LEARNERS:
            gain = mean.loc[k, m] - ms_.loc[k, m]
            seeds_up = int((rp.loc[k][m] > rs.loc[k][m]).sum())  # seed by seed (same geometry)
            cells += [num(ms_.loc[k, m]), num(mean.loc[k, m]), signed(gain), f"{seeds_up}/3"]
        cells.append(num(mean.loc[k, "greedy"]))
        gain_rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    write_table("shared_vs_individual", "\n".join(gain_rows) + "\n")
    numbers["reward_paper_mean"] = ms_.round(1).to_dict()
    numbers["reward_paper_std"] = ss_.round(1).to_dict()
    numbers["gain_individual"] = (mean[LEARNERS] - ms_[LEARNERS]).round(0).to_dict()

    # ---- Table: the complete proposal against the original scheme (FedAvg, shared) --------------
    rv_all = fin_v["reward"].unstack("method")
    original = rs["fedavg"]
    rows = []
    for k in KS:
        cells = [num(original.loc[k].mean())]
        for m in ["gossip", "gossip2"]:
            d = rv_all.loc[k][m] - original.loc[k]  # seed by seed (same geometry)
            cells += [num(rv_all.loc[k][m].mean()), signed(d.mean()), f"{int((d > 0).sum())}/3"]
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    write_table("combined", "\n".join(rows) + "\n")

    # FedAvg against DTE, both with the individual reward: does sharing with everyone help?
    d = (rp["fedavg"] - rp["dte"]).groupby("k")
    numbers["fedavg_vs_dte_propia"] = {"diff": d.mean().round(0).to_dict(),
                                       "wins": d.apply(lambda x: int((x > 0).sum())).to_dict()}

    # ---- Table and numbers: what is lost after the best block -----------------------------------
    drop_p = forgetting(ep_p[ep_p["method"].isin(LEARNERS)]).groupby(["k", "method"]).mean().unstack()
    drop_s = forgetting(ep_s[ep_s["method"].isin(LEARNERS)]).groupby(["k", "method"]).mean().unstack()
    rows = []
    for k in KS:
        rows.append(f"{k} & " + " & ".join(num(drop_s.loc[k, m]) for m in LEARNERS) + " & "
                    + " & ".join(num(drop_p.loc[k, m]) for m in LEARNERS) + r" \\")
    write_table("forgetting", "\n".join(rows) + "\n")
    numbers["drop_propia"] = drop_p.round(0).to_dict()
    numbers["drop_paper"] = drop_s.round(0).to_dict()

    # ---- Convergence ----------------------------------------------------------------------------
    learn_p = ep_p[ep_p["method"].isin(LEARNERS)]
    to_gain = per_run_metric(learn_p, episodes_to_final_gain)
    settled = per_run_metric(learn_p, convergence_episode)
    speed = to_gain.groupby(["k", "method"]).mean().unstack()[LEARNERS]
    fig_convergence(speed, FIG / "res_convergence.pdf")
    rows = []
    for k in KS:
        rows.append(f"{k} & " + " & ".join(num(speed.loc[k, m]) for m in LEARNERS) + " & "
                    + " & ".join(num(mean.loc[k, m] - ep_p[(ep_p.method == m) & (ep_p.k == k) & (ep_p.episode < 50)]
                                     .groupby("seed")["reward"].mean().mean()) for m in LEARNERS)
                    + r" \\")
    write_table("convergence", "\n".join(rows) + "\n")
    numbers["speed"] = speed.round(0).to_dict()
    numbers["settled_count"] = int(settled.notna().sum())
    numbers["settled_total"] = int(settled.size)
    numbers["settled_min"] = float(settled.min())
    numbers["to_gain_nan"] = int(to_gain.isna().sum())

    # ---- Quality of service and behavior (individual) -------------------------------------------
    for metric, scale, dec, name in [("cue_rate_mbps", 1, 2, "cue_rate"), ("outage_rate", 100, 0, "outage"),
                                     ("v2v_delay_ms", 1, 1, "delay"), ("channels_used", 1, 1, "channels"),
                                     ("mean_power_ratio", 1, 2, "power")]:
        mm, sd = mean_std(fin_p, metric)
        rows = [f"{k} & " + " & ".join(pm(mm.loc[k, m] * scale, sd.loc[k, m] * scale, dec) for m in ORDER)
                + r" \\" for k in KS]
        write_table(name, "\n".join(rows) + "\n")
        numbers[f"{metric}_propia"] = (mm * scale).round(3).to_dict()
        mm_s, _ = mean_std(fin_s, metric)
        numbers[f"{metric}_paper"] = (mm_s * scale).round(3).to_dict()

    # Replication: CUE rate against Fig. 3 of Li et al. (values read from the figure)
    cue_p, _ = mean_std(fin_p, "cue_rate_mbps")
    cue_s, _ = mean_std(fin_s, "cue_rate_mbps")
    rows = []
    for k in KS:
        rows.append(f"{k} & {LI_FIG3_MBPS['fedavg'][k]:.2f} & {num(cue_s.loc[k, 'fedavg'], 2)} & "
                    f"{num(cue_p.loc[k, 'fedavg'], 2)} & {LI_FIG3_MBPS['greedy'][k]:.2f} & "
                    f"{num(cue_p.loc[k, 'greedy'], 2)}" + r" \\")
    li_f, li_g = LI_FIG3_MBPS["fedavg"], LI_FIG3_MBPS["greedy"]
    rows.append(r"\midrule")
    rows.append(r"Change 5$\to$30 & " + f"{100 * (li_f[30] / li_f[5] - 1):.0f}\\% & "
                f"{100 * (cue_s.loc[30, 'fedavg'] / cue_s.loc[5, 'fedavg'] - 1):.0f}\\% & "
                f"{100 * (cue_p.loc[30, 'fedavg'] / cue_p.loc[5, 'fedavg'] - 1):.0f}\\% & "
                f"{100 * (li_g[30] / li_g[5] - 1):.0f}\\% & "
                f"{100 * (cue_p.loc[30, 'greedy'] / cue_p.loc[5, 'greedy'] - 1):.0f}\\%" + r" \\")
    write_table("replication_cue", "\n".join(rows).replace("-", "$-$").replace("$$-$$", "$-$") + "\n")

    # ---- Communication --------------------------------------------------------------------------
    models, _ = mean_std(fin_v, "comm_models")
    load, _ = mean_std(fin_v, "comm_max_load")
    model_bytes = 121588
    rows = []
    for k in KS:
        f = models.loc[k, "fedavg"]
        cells = [num(f), num(models.loc[k, "gossip"]), num(models.loc[k, "gossip3"]), num(models.loc[k, "gossip2"]),
                 f"{models.loc[k, 'gossip'] / f:.2f}", f"{models.loc[k, 'gossip2'] / f:.2f}",
                 num(load.loc[k, "fedavg"]), num(load.loc[k, "gossip"], 1), num(load.loc[k, "gossip2"], 1),
                 f"{load.loc[k, 'gossip'] / load.loc[k, 'fedavg']:.2f}",
                 f"{load.loc[k, 'gossip2'] / load.loc[k, 'fedavg']:.2f}"]
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    write_table("communication", "\n".join(rows) + "\n")
    numbers["comm_models"] = models.round(2).to_dict()
    numbers["comm_max_load"] = load.round(2).to_dict()
    numbers["comm_mb_fedavg"] = {k: round(models.loc[k, "fedavg"] * model_bytes / 1e6, 2) for k in KS}
    numbers["comm_mb_gossip"] = {k: round(models.loc[k, "gossip"] * model_bytes / 1e6, 2) for k in KS}
    numbers["comm_mb_gossip2"] = {k: round(models.loc[k, "gossip2"] * model_bytes / 1e6, 2) for k in KS}

    # ---- Gossip with S neighbors: reward and paired comparison with FedAvg ----------------------
    mv, sv = mean_std(fin_v, "reward")
    rv = fin_v["reward"].unstack("method")
    rows = []
    for k in KS:
        cells = [pm(mv.loc[k, m], sv.loc[k, m]) for m in ["fedavg", "gossip", "gossip3", "gossip2"]]
        for m in ["gossip", "gossip3", "gossip2"]:
            cells.append(f"{int((rv.loc[k][m] > rv.loc[k]['fedavg']).sum())}/3")
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    write_table("gossip_neighbors", "\n".join(rows) + "\n")
    numbers["reward_vecinos_mean"] = mv[["gossip", "gossip3", "gossip2"]].round(1).to_dict()
    numbers["wins_vs_fedavg"] = {m: {k: int((rv.loc[k][m] > rv.loc[k]["fedavg"]).sum()) for k in KS}
                                 for m in ["gossip", "gossip3", "gossip2"]}

    # ---- Dispersion and topology ----------------------------------------------------------------
    disp, _ = mean_std(fin_v, "weight_dispersion")
    numbers["dispersion"] = disp.round(2).to_dict()
    neigh = fin_p.loc["greedy"]["neighbors"].groupby("k").mean()
    iso = fin_p.loc["greedy"]["isolated_rate"].groupby("k").mean()
    numbers["neighbors"] = neigh.round(2).to_dict()
    numbers["isolated"] = (iso * 100).round(1).to_dict()

    # ---- Diagnosis (K = 10, seed 0) -------------------------------------------------------------
    diag = data["diag"]
    g10 = fin_p.loc[("greedy", 10, 0), "reward"]
    u10 = fin_p.loc[("umbral", 10, 0), "reward"]
    fig_diagnosis(diag, g10, u10, FIG / "res_diagnosis.pdf")
    diag_rows = []
    info = {}
    for name, df in diag.items():
        r = df.sort_values("episode").set_index("episode")["reward"]
        n = len(r)
        final = r.iloc[-max(1, math.ceil(0.1 * n)):].mean()
        peak = r.groupby(r.index // 200).mean().max()
        info[name] = {"final": round(final), "peak": round(peak), "change_pct": round(100 * (final - peak) / abs(peak), 1)}
    numbers["diag"] = info
    numbers["diag_greedy_k10_s0"] = round(float(g10))
    numbers["diag_umbral_k10_s0"] = round(float(u10))
    DIAG_TABLE = [
        ("base", "Shared", "0.99", "Initial configuration"),
        ("buffer100k", "Shared", "0.99", "Buffer of 100,000 transitions"),
        ("reward0.01", "Shared", "0.99", "Reward multiplied by 0.01"),
        ("ambos", "Shared", "0.99", "Both of the above"),
        ("red-paper", "Shared", "0.99", "Network of Li et al.\\ (500, 250, 120)"),
        ("eps0.1", "Shared", "0.99", "$\\varepsilon$ floor of 0.1"),
        ("paper-todo", "Shared", "0.9", "Network and $\\varepsilon$ floor of Li et al."),
        ("gamma0.95", "Shared", "0.95", "---"),
        ("gamma0.9", "Shared", "0.9", "---"),
        ("gamma0.85", "Shared", "0.85", "---"),
        ("gamma0.8", "Shared", "0.8", "---"),
        ("gamma0.7", "Shared", "0.7", "---"),
        ("cola-fin", "Shared", "0.99", "Queue at the end of the slot"),
        ("cola-fin-gamma0.9", "Shared", "0.9", "Queue at the end of the slot"),
        ("cola-fin-gamma0.5", "Shared", "0.5", "Queue at the end of the slot"),
        ("cola-fin-gamma0.5-canal", "Shared", "0.5", "Queue at the end; own channel in the observation"),
        ("local", "Individual", "0.99", "Queue at the start of the slot"),
        ("local-gamma0.9", "Individual", "0.9", "Queue at the start of the slot"),
        ("local-gamma0.9-canal", "Individual", "0.9", "Queue at the start; own channel in the observation"),
        ("local-gamma0.9-cola-fin", "Individual", "0.9", "Queue at the end of the slot (chosen)"),
        ("mejor-lr3e-4", "Individual", "0.9", "Queue at the end; learning rate $3\\times10^{-4}$"),
        ("mejor-lr1e-4", "Individual", "0.9", "Queue at the end; learning rate $10^{-4}$"),
        ("mejor-eps0.1", "Individual", "0.9", "Queue at the end; $\\varepsilon$ floor of 0.1"),
        ("mejor-buffer100k", "Individual", "0.9", "Queue at the end; buffer of 100,000"),
    ]
    groups = {"base": "Drop", "red-paper": "Learning", "gamma0.95": "$\\gamma$", "cola-fin": "Queue",
              "local": "Reward", "mejor-lr3e-4": "Stability"}
    for name, reward_kind, gamma, other in DIAG_TABLE:
        if name in groups and name != "base":
            diag_rows.append(r"\midrule")
        i = info[name]
        best = name == "local-gamma0.9-cola-fin"
        cells = [reward_kind, gamma, other, num(i["peak"]), num(i["final"]), signed(i["change_pct"], 1) + r"\%"]
        if best:
            cells = [r"\textbf{" + c + "}" for c in cells]
        diag_rows.append(" & ".join(cells) + r" \\")
    write_table("diagnosis", "\n".join(diag_rows) + "\n")

    # ---- Appendix: complete tables per sweep ----------------------------------------------------
    for sweep, fin in [("individual", fin_p), ("shared", fin_s)]:
        rows = []
        for m in ORDER:
            for k in KS:
                sub = fin.loc[m].loc[k]
                cells = [LABEL_TEX[m] if k == 5 else "", str(k),
                         pm(sub["reward"].mean(), sub["reward"].std()),
                         num(sub["cue_rate_mbps"].mean(), 2), num(100 * sub["outage_rate"].mean(), 1),
                         num(sub["v2v_delay_ms"].mean(), 1), num(sub["channels_used"].mean(), 1),
                         num(sub["mean_power_ratio"].mean(), 2), num(100 * sub["co_channel_rate"].mean(), 1)]
                rows.append(" & ".join(cells) + r" \\")
            if m != ORDER[-1]:
                rows.append(r"\midrule")
        write_table(f"appendix_{sweep}", "\n".join(rows) + "\n")

    # ---- Computation time -----------------------------------------------------------------------
    walls = []
    for sweep in ["propia", "paper", "propia-vecinos2", "propia-vecinos3"]:
        for f in (RUNS / sweep).glob("k*/*.json"):
            m = json.loads(f.read_text())
            walls.append({"sweep": sweep, "method": m["method"], "k": m["num_v2v_pairs"], "h": m["wall_time_s"] / 3600})
    walls = pd.DataFrame(walls)
    wt = walls[walls.sweep.isin(["propia", "paper"])].groupby(["k", "method"])["h"].mean().unstack()
    rows = [f"{k} & " + " & ".join(num(wt.loc[k, m], 2) for m in ["greedy", "umbral", "dte", "fedavg", "gossip"])
            + r" \\" for k in KS]
    write_table("time", "\n".join(rows) + "\n")
    numbers["wall_total_h"] = round(float(walls["h"].sum()), 1)
    numbers["n_runs"] = int(len(walls))

    (CACHE / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")
    print(json.dumps(numbers, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
