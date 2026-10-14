"""Metrics computed from finished runs rather than during training, and the loaders the
notebooks share.

Convergence speed (episodes until the reward settles) is the paper's third headline metric
(Li et al. 2022, Fig. 10) and needs the whole learning curve, so it lives here. The loaders
read the K sweep as `make run-greedy` (and the other methods' jobs) leave it:
runs/k<K>/<method>_seed<seed>.csv, with a .json of settings and wall time next to each CSV.
"""

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

RUN_FILE = re.compile(r"(?P<method>[a-z]+)_seed(?P<seed>\d+)")
K_DIR = re.compile(r"k(?P<k>\d+)")
RUN_KEYS = ["method", "k", "seed"]


def convergence_episode(
    rewards: Sequence[float],
    window: int = 50,
    tolerance: float = 0.05,
    final_fraction: float = 0.1,
) -> float:
    """First episode from which the `window`-episode moving average of the reward stays
    within `tolerance` x |final value| of the final value, the mean of the last
    `final_fraction` of the episodes. NaN if the moving average is still outside that band
    at the end of the run.
    """
    values = np.asarray(rewards, dtype=float)
    final = values[-max(1, math.ceil(final_fraction * len(values))) :].mean()
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")  # index i ends at episode i + window - 1
    outside = np.flatnonzero(np.abs(moving_average - final) > tolerance * abs(final))
    if outside.size == 0:
        return float(window - 1)
    if outside[-1] == moving_average.size - 1:
        return math.nan
    return float(outside[-1] + 1 + window - 1)


def _run_files(root: Path, suffix: str, methods: Sequence[str] | None):
    """(method, k, seed, path) for every runs/k<K>/<method>_seed<seed><suffix> under root."""
    for path in sorted(Path(root).glob(f"k*/*{suffix}")):
        k_match = K_DIR.fullmatch(path.parent.name)
        run_match = RUN_FILE.fullmatch(path.stem)
        if not k_match or not run_match:
            continue
        if methods is not None and run_match["method"] not in methods:
            continue
        yield run_match["method"], int(k_match["k"]), int(run_match["seed"]), path


def load_runs(root: Path, methods: Sequence[str] | None = None) -> pd.DataFrame:
    """Every run's CSV rows, tagged with its method, K and seed."""
    frames = [
        pd.read_csv(path).assign(method=method, k=k, seed=seed)
        for method, k, seed, path in _run_files(root, ".csv", methods)
    ]
    if not frames:
        raise FileNotFoundError(f"No run CSVs under {Path(root).resolve()} (expected k<K>/<method>_seed<seed>.csv)")
    return pd.concat(frames, ignore_index=True)


def load_metadata(root: Path, methods: Sequence[str] | None = None) -> pd.DataFrame:
    """The JSON next to each run's CSV (settings, host, wall time), one row per run."""
    rows = [
        {**json.loads(path.read_text()), "method": method, "k": k, "seed": seed}
        for method, k, seed, path in _run_files(root, ".json", methods)
    ]
    return pd.DataFrame(rows)


def per_episode_means(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run and episode: per-pair metrics averaged over the K agents (cell metrics
    are the same in every row, so their mean is the value), plus the share of isolated pairs."""
    runs = runs.drop(columns="agent", errors="ignore")
    if "neighbors" in runs:
        runs = runs.assign(isolated_rate=(runs["neighbors"] == 0).astype(float))
    return runs.groupby([*RUN_KEYS, "episode"], as_index=False).mean(numeric_only=True)


def final_values(per_episode: pd.DataFrame, final_fraction: float = 0.1) -> pd.DataFrame:
    """Each run's value of every metric: the mean over its last `final_fraction` of episodes."""

    def tail_mean(run: pd.DataFrame) -> pd.Series:
        last = run.sort_values("episode").tail(max(1, math.ceil(final_fraction * len(run))))
        return last.drop(columns="episode").mean(numeric_only=True)

    return per_episode.groupby(RUN_KEYS)[[c for c in per_episode if c not in RUN_KEYS]].apply(tail_mean)


def summarize_by_k(final: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across seeds, per method and K, plus the number of seeds."""
    summary = final.groupby(level=["method", "k"]).agg(["mean", "std"])
    summary[("seeds", "")] = final.groupby(level=["method", "k"]).size()
    return summary
