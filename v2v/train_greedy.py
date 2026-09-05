"""Run the non-learning Greedy baseline (Li et al. 2022, Sec. V) on the V2V
resource-allocation environment: always picks the channel with the fewest
co-channel V2V neighbors and transmits at maximum power (agents/greedy_policy.py).
No training happens here -- this is the comparison point for main.py's FedAvg run.
"""

import argparse
from pathlib import Path

from agents.greedy_policy import greedy_action
from main import write_records_csv
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def run_greedy(params: V2VEnvParams, episodes: int, seed: int) -> list[tuple[int, str, float]]:
    env = V2VEnv(params)
    records: list[tuple[int, str, float]] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = {agent_id: 0.0 for agent_id in env.possible_agents}

        while env.agents:
            actions = {a: greedy_action(obs[a], params) for a in env.agents}
            obs, rewards, _, _, _ = env.step(actions)
            for a in env.possible_agents:
                episode_reward[a] += rewards[a]

        for a, r in episode_reward.items():
            records.append((ep, a, r))

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
