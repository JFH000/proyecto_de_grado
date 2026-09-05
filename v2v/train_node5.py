"""Train K D3QN agents on V2VEnv with FedAvg aggregation (checklist node 5):
every `fedavg_every` slots, each agent's online weights are uploaded, averaged
weighted by the minibatch size it trained on since the last round, and
broadcast back. Builds on node 4's DTE-only run (train_node4.py) as the
control this is meant to be compared against (node 6).
"""

import argparse
from pathlib import Path

from agents.d3qn_agent import D3QNAgent
from agents.federated import fedavg_round
from train_node4 import (
    epsilon_schedule,
    make_agents,
    print_convergence_verdict,
    resolve_device,
    write_records_csv,
)
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def run_training_fedavg(
    params: V2VEnvParams,
    episodes: int,
    seed: int,
    fedavg_every: int,
    log_every: int = 10,
    device: str = "cpu",
    agent_kwargs: dict | None = None,
) -> tuple[list[tuple[int, str, float]], dict[str, D3QNAgent]]:
    env = V2VEnv(params)
    total_steps = episodes * params.max_steps
    agents = make_agents(
        params, seed, per_beta_frames=total_steps, device=device, agent_kwargs=agent_kwargs
    )

    records: list[tuple[int, str, float]] = []
    recent: dict[str, list[float]] = {agent_id: [] for agent_id in env.possible_agents}
    sample_counts = {agent_id: 0.0 for agent_id in env.possible_agents}
    global_step = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = {agent_id: 0.0 for agent_id in env.possible_agents}

        while env.agents:
            eps = epsilon_schedule(global_step, total_steps)
            actions = {a: agents[a].select_action(obs[a], eps) for a in env.agents}
            next_obs, rewards, _, _, _ = env.step(actions)
            for a in env.possible_agents:
                agents[a].store_transition(obs[a], actions[a], rewards[a], next_obs[a])
                if agents[a].learn() is not None:
                    sample_counts[a] += agents[a].batch_size
                episode_reward[a] += rewards[a]
            obs = next_obs
            global_step += 1

            if global_step % fedavg_every == 0 and any(sample_counts.values()):
                fedavg_round(agents, sample_counts)
                sample_counts = {agent_id: 0.0 for agent_id in env.possible_agents}

        for a, r in episode_reward.items():
            records.append((ep, a, r))
            recent[a].append(r)

        if (ep + 1) % log_every == 0:
            summary = ", ".join(f"{a}={sum(rs) / len(rs):.3f}" for a, rs in recent.items())
            print(f"Episode {ep + 1:>4}/{episodes} | {summary}")
            recent = {agent_id: [] for agent_id in env.possible_agents}

    return records, agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--fedavg-every", type=int, default=50, help="Aggregate weights every N slots"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/node5_fedavg_rewards.csv"))
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="'auto' picks cuda if available, else cpu",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=200)
    records, _ = run_training_fedavg(
        params, args.episodes, args.seed, args.fedavg_every, args.log_every, device=device
    )

    write_records_csv(records, args.output)
    print(f"\nSaved: {args.output}")
    print_convergence_verdict(records, args.episodes)


if __name__ == "__main__":
    main()
