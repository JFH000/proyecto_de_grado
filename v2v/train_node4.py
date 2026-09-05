"""Train K independent D3QN agents on V2VEnv with no FedAvg (checklist node 4,
subtask 3). Each agent learns only from its own transitions -- DTE control
run, meant to converge before FedAvg (node 5) is layered on top.
"""

import argparse
import csv
from pathlib import Path

import torch

from agents.d3qn_agent import D3QNAgent
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def epsilon_schedule(
    step: int, total_steps: int, start: float = 1.0, end: float = 0.05, decay_fraction: float = 0.6
) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def make_agents(
    params: V2VEnvParams, seed: int, per_beta_frames: int, device: str = "cpu"
) -> dict[str, D3QNAgent]:
    return {
        agent_id: D3QNAgent(
            state_dim=params.state_dim,
            action_dim=params.action_space_size,
            seed=seed + i,
            per_beta_frames=per_beta_frames,
            device=device,
        )
        for i, agent_id in enumerate(f"v2v_{i}" for i in range(params.num_v2v_pairs))
    }


def run_training(
    params: V2VEnvParams, episodes: int, seed: int, log_every: int = 10, device: str = "cpu"
) -> list[tuple[int, str, float]]:
    env = V2VEnv(params)
    total_steps = episodes * params.max_steps
    agents = make_agents(params, seed, per_beta_frames=total_steps, device=device)

    records: list[tuple[int, str, float]] = []
    recent: dict[str, list[float]] = {agent_id: [] for agent_id in env.possible_agents}
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
                agents[a].learn()
                episode_reward[a] += rewards[a]
            obs = next_obs
            global_step += 1

        for a, r in episode_reward.items():
            records.append((ep, a, r))
            recent[a].append(r)

        if (ep + 1) % log_every == 0:
            summary = ", ".join(f"{a}={sum(rs) / len(rs):.3f}" for a, rs in recent.items())
            print(f"Episode {ep + 1:>4}/{episodes} | {summary}")
            recent = {agent_id: [] for agent_id in env.possible_agents}

    return records


def write_records_csv(records: list[tuple[int, str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "agent", "reward"])
        writer.writerows(records)


def print_convergence_verdict(records: list[tuple[int, str, float]], episodes: int) -> None:
    by_agent: dict[str, list[float]] = {}
    for ep, agent_id, reward in sorted(records, key=lambda r: r[0]):
        by_agent.setdefault(agent_id, []).append(reward)

    window = max(1, episodes // 10)
    for agent_id, rewards in by_agent.items():
        first = sum(rewards[:window]) / window
        last = sum(rewards[-window:]) / window
        verdict = "improved" if last > first else "did not improve"
        print(f"agent {agent_id}: first-10%={first:.3f} -> last-10%={last:.3f} ({verdict})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/node4_no_fedavg_rewards.csv"))
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="'auto' picks cuda if available, else cpu",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=200)
    records = run_training(params, args.episodes, args.seed, args.log_every, device=device)

    write_records_csv(records, args.output)
    print(f"\nSaved: {args.output}")
    print_convergence_verdict(records, args.episodes)


if __name__ == "__main__":
    main()
