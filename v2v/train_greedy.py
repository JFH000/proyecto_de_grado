"""Run the non-learning Greedy baseline (Li et al. 2022, Sec. V) on the V2V
resource-allocation environment: always picks the channel with the fewest
co-channel V2V neighbors (ties broken at random) and transmits at maximum power
(agents/greedy_policy.py).
No training happens here -- this is the comparison point for main.py's FedAvg run.
The CSV has the same columns as main.py's (EpisodeRecord).
"""

import argparse
from pathlib import Path

import numpy as np

from agents.greedy_policy import greedy_action
from main import EpisodeAccumulator, EpisodeRecord, episode_seed, write_records_csv
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def run_greedy(params: V2VEnvParams, episodes: int, seed: int) -> list[EpisodeRecord]:
    env = V2VEnv(params)
    # Tie-breaking stream, spawned so it never coincides with the env's layout seeds.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    records: list[EpisodeRecord] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        episode = EpisodeAccumulator(env.possible_agents)

        while env.agents:
            actions = {a: greedy_action(obs[a], params, rng) for a in env.agents}
            obs, rewards, _, _, infos = env.step(actions)
            episode.add(rewards, infos)

        records.extend(episode.records(ep))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-v2v-pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/greedy_rewards.csv"))
    args = parser.parse_args()

    params = V2VEnvParams(
        num_cues=15, num_v2v_pairs=args.num_v2v_pairs, num_power_levels=3, max_steps=200
    )
    records = run_greedy(params, args.episodes, args.seed)

    write_records_csv(records, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
