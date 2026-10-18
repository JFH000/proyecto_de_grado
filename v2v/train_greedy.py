"""Run a non-learning baseline on the V2V resource-allocation environment.

`--policy greedy` (default): the Greedy baseline of Li et al. (2022, Sec. V), which always
picks the channel with the fewest co-channel V2V neighbors (ties broken at random) and
transmits at maximum power (agents/greedy_policy.py).
`--policy umbral`: the threshold rule of agents/threshold_policy.py, not in the paper: same
channel choice, maximum power on strong links, and weak links silent until their queue fills.
It shows what local information alone can achieve.
No training happens here -- these are the comparison points for main.py's runs.
The CSV has the same columns as main.py's (EpisodeRecord), with no training loss,
weight dispersion or model traffic; a JSON with the run's settings and wall time goes
next to it.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from agents.greedy_policy import greedy_action
from agents.threshold_policy import threshold_action
from main import (
    EpisodeAccumulator,
    EpisodeRecord,
    episode_seed,
    run_metadata,
    write_records_csv,
    write_run_metadata,
)
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


POLICIES = {"greedy": greedy_action, "umbral": threshold_action}


def run_greedy(params: V2VEnvParams, episodes: int, seed: int, policy: str = "greedy") -> list[EpisodeRecord]:
    act = POLICIES[policy]
    env = V2VEnv(params)
    # Tie-breaking stream, spawned so it never coincides with the env's layout seeds.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    records: list[EpisodeRecord] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        episode = EpisodeAccumulator(env.possible_agents)

        while env.agents:
            actions = {a: act(obs[a], params, rng) for a in env.agents}
            obs, rewards, _, _, infos = env.step(actions)
            episode.add(rewards, infos)

        records.extend(episode.records(ep))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-v2v-pairs", type=int, default=5)
    parser.add_argument("--policy", default="greedy", choices=list(POLICIES))
    parser.add_argument("--output", type=Path, default=Path("runs/greedy_rewards.csv"))
    args = parser.parse_args()

    params = V2VEnvParams(
        num_cues=15, num_v2v_pairs=args.num_v2v_pairs, num_power_levels=3, max_steps=200
    )
    start = time.perf_counter()
    records = run_greedy(params, args.episodes, args.seed, args.policy)
    wall_time_s = time.perf_counter() - start

    write_records_csv(records, args.output)
    write_run_metadata(
        args.output,
        run_metadata(
            wall_time_s,
            "cpu",
            method=args.policy,
            seed=args.seed,
            episodes=args.episodes,
            num_v2v_pairs=args.num_v2v_pairs,
        ),
    )
    print(f"Saved: {args.output} ({wall_time_s / 60:.1f} min)")


if __name__ == "__main__":
    main()
