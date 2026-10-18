"""Train K D3QN agents on the V2V resource-allocation environment (Li et al.
2022, "Federated Multi-Agent Deep RL for Resource Allocation of V2V
Communications").

`--aggregation fedavg` (default): every `--aggregate-every` slots, each agent's
online-network weights are uploaded, averaged weighted by the minibatch size
it trained on since the last round, and broadcast back (agents/federated.py).
`--aggregation gossip`: the thesis proposal. At the same instants, each agent
averages its weights with those of the pairs within the 300 m neighbor radius,
without the BS (agents/gossip.py); `--gossip-neighbors S` pulls at most S random
ones per round.
`--aggregation none`: the agents never share weights -- pure DTE, the control
that shows what federation adds.

Writes one CSV row per episode and agent: the summed reward, the per-slot QoS and
behaviour metrics V2VEnv.step reports averaged over the episode, and the training-side
metrics (loss, weight dispersion, model traffic) -- see EpisodeRecord. Next to the CSV, a
JSON file records the run's settings, where it ran and how long it took.
"""

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from agents.d3qn_agent import D3QNAgent
from agents.federated import fedavg_node_loads, fedavg_round, model_num_bytes, weight_dispersion
from agents.gossip import gossip_round
from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams

AGGREGATIONS = ("none", "fedavg", "gossip")

# Runs with different --seed must not share randomness. With seed + episode and seed + index,
# run s+1 would replay run s shifted by one episode, and its agent i would be run s's agent i+1.
SEED_STRIDE = 1_000_000


def episode_seed(seed: int, episode: int) -> int:
    """Env layout seed of one episode. Every method run with the same seed sees the same layouts."""
    return seed * SEED_STRIDE + episode


def agent_seed(seed: int, index: int) -> int:
    return seed * SEED_STRIDE + index


class EpisodeRecord(NamedTuple):
    episode: int
    agent: str
    reward: float  # summed over the episode's slots
    cue_rate_mbps: float  # mean over slots of the cell's mean CUE rate
    v2v_delay_ms: float  # mean over slots of this pair's queueing delay
    outage_rate: float  # fraction of slots in which this pair failed the reliability constraint
    neighbors: float  # pairs within the neighbor radius (fixed within an episode)
    co_channel_rate: float  # fraction of slots this pair transmitted on a channel another transmitting pair used
    channels_used: float  # mean over slots of the distinct channels carrying V2V transmissions (cell)
    mean_power_ratio: float  # mean transmit power as a fraction of P_max
    max_power_rate: float  # fraction of slots at P_max
    train_loss: float  # mean D3QN loss over this agent's updates; NaN if it never learned
    weight_dispersion: float  # mean over aggregation instants of the agents' weight dispersion (cell); NaN if none
    comm_models: int  # model transfers between any two nodes during the episode (cell)
    comm_bytes: int  # comm_models x size of one model
    comm_max_load: int  # most transfers a single node (vehicle or BS) took part in (cell)


# EpisodeRecord field <- env.step info key, averaged over the episode's slots.
SLOT_AVERAGES = {
    "cue_rate_mbps": "cue_rate_mbps",
    "v2v_delay_ms": "v2v_delay_ms",
    "outage_rate": "outage",
    "neighbors": "neighbors",
    "co_channel_rate": "co_channel",
    "channels_used": "channels_used",
    "mean_power_ratio": "power_ratio",
    "max_power_rate": "max_power",
}


def mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


class EpisodeAccumulator:
    """Collects one episode into one EpisodeRecord per agent: the per-slot rewards and
    env.step infos, plus the training events (losses, weight dispersion readings and the
    node loads of each aggregation round)."""

    def __init__(self, agent_ids: list[str], model_bytes: int = 0):
        self.slots = 0
        self.model_bytes = model_bytes
        self.totals = {agent_id: dict.fromkeys(["reward", *SLOT_AVERAGES], 0.0) for agent_id in agent_ids}
        self.losses: dict[str, list[float]] = {agent_id: [] for agent_id in agent_ids}
        self.dispersions: list[float] = []
        self.node_loads: dict[str, int] = {}

    def add(self, rewards: dict[str, float], infos: dict[str, dict]) -> None:
        self.slots += 1
        for agent_id, total in self.totals.items():
            info = infos[agent_id]
            total["reward"] += rewards[agent_id]
            for field, key in SLOT_AVERAGES.items():
                total[field] += info[key]

    def add_loss(self, agent_id: str, loss: float) -> None:
        self.losses[agent_id].append(loss)

    def add_dispersion(self, value: float) -> None:
        self.dispersions.append(value)

    def add_aggregation(self, node_loads: dict[str, int]) -> None:
        """One aggregation round: models each node (agents, and the BS if any) sent or received."""
        for node, load in node_loads.items():
            self.node_loads[node] = self.node_loads.get(node, 0) + load

    def records(self, episode: int) -> list[EpisodeRecord]:
        n = max(1, self.slots)
        comm_models = sum(self.node_loads.values()) // 2  # every transfer has a sender and a receiver
        return [
            EpisodeRecord(
                episode=episode,
                agent=agent_id,
                reward=total["reward"],
                **{field: total[field] / n for field in SLOT_AVERAGES},
                train_loss=mean_or_nan(self.losses[agent_id]),
                weight_dispersion=mean_or_nan(self.dispersions),
                comm_models=comm_models,
                comm_bytes=comm_models * self.model_bytes,
                comm_max_load=max(self.node_loads.values(), default=0),
            )
            for agent_id, total in self.totals.items()
        ]


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
    params: V2VEnvParams,
    seed: int,
    per_beta_frames: int,
    device: str = "cpu",
    agent_kwargs: dict | None = None,
) -> dict[str, D3QNAgent]:
    agent_kwargs = agent_kwargs or {}
    return {
        agent_id: D3QNAgent(
            state_dim=params.state_dim,
            action_dim=params.action_space_size,
            seed=agent_seed(seed, i),
            per_beta_frames=per_beta_frames,
            device=device,
            **agent_kwargs,
        )
        for i, agent_id in enumerate(f"v2v_{i}" for i in range(params.num_v2v_pairs))
    }


def run_training(
    params: V2VEnvParams,
    episodes: int,
    seed: int,
    fedavg_every: int,
    log_every: int = 10,
    device: str = "cpu",
    agent_kwargs: dict | None = None,
    aggregation: str = "fedavg",
    gossip_neighbors: int | None = None,
    reward_scale: float = 1.0,
    epsilon_end: float = 0.05,
) -> tuple[list[EpisodeRecord], dict[str, D3QNAgent]]:
    """reward_scale multiplies the reward the agents store and learn from; the CSV keeps the
    environment's reward, so runs with different scales stay comparable. epsilon_end is the
    exploration floor, reached at 60 % of the training (epsilon_schedule)."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    env = V2VEnv(params)
    # Draws the random neighbor subsets of --gossip-neighbors; spawned so it never coincides
    # with the env's layout seeds or the agents' own streams.
    gossip_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    total_steps = episodes * params.max_steps
    agents = make_agents(
        params, seed, per_beta_frames=total_steps, device=device, agent_kwargs=agent_kwargs
    )

    records: list[EpisodeRecord] = []
    recent: dict[str, list[float]] = {agent_id: [] for agent_id in env.possible_agents}
    sample_counts = {agent_id: 0.0 for agent_id in env.possible_agents}
    model_bytes = model_num_bytes(next(iter(agents.values())).online)
    global_step = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        episode = EpisodeAccumulator(env.possible_agents, model_bytes=model_bytes)

        while env.agents:
            eps = epsilon_schedule(global_step, total_steps, end=epsilon_end)
            actions = {a: agents[a].select_action(obs[a], eps) for a in env.agents}
            next_obs, rewards, _, _, infos = env.step(actions)
            for a in env.possible_agents:
                agents[a].store_transition(obs[a], actions[a], rewards[a] * reward_scale, next_obs[a])
                loss = agents[a].learn()
                if loss is not None:
                    sample_counts[a] += agents[a].batch_size
                    episode.add_loss(a, loss)
            episode.add(rewards, infos)
            obs = next_obs
            global_step += 1

            if global_step % fedavg_every == 0:
                # Read before aggregating (right after a FedAvg round it is always 0), and at the
                # same instants under DTE, so every method is measured alike.
                episode.add_dispersion(weight_dispersion(agents))
                if aggregation != "none" and any(sample_counts.values()):
                    if aggregation == "fedavg":
                        fedavg_round(agents, sample_counts)
                        episode.add_aggregation(fedavg_node_loads(list(agents)))
                    else:
                        episode.add_aggregation(
                            gossip_round(agents, sample_counts, env.neighbor_mask, gossip_rng, gossip_neighbors)
                        )
                    sample_counts = {agent_id: 0.0 for agent_id in env.possible_agents}

        for record in episode.records(ep):
            records.append(record)
            recent[record.agent].append(record.reward)

        if (ep + 1) % log_every == 0:
            summary = ", ".join(f"{a}={sum(rs) / len(rs):.3f}" for a, rs in recent.items())
            print(f"Episode {ep + 1:>4}/{episodes} | {summary}")
            recent = {agent_id: [] for agent_id in env.possible_agents}

    return records, agents


def write_records_csv(records: list[EpisodeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EpisodeRecord._fields)
        writer.writerows(records)


def run_metadata(wall_time_s: float, device: str, **settings) -> dict:
    """A run's settings plus where and how long it ran, for the experimental-setup table."""
    meta = {
        **settings,
        "wall_time_s": wall_time_s,
        "device": device,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def write_run_metadata(output_path: Path, metadata: dict) -> None:
    """Writes the metadata next to the run's CSV, same name with a .json suffix."""
    path = output_path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def print_convergence_verdict(records: list[EpisodeRecord], episodes: int) -> None:
    by_agent: dict[str, list[float]] = {}
    for record in sorted(records, key=lambda r: r.episode):
        by_agent.setdefault(record.agent, []).append(record.reward)

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
    parser.add_argument("--num-v2v-pairs", type=int, default=5)
    parser.add_argument(
        "--aggregation", default="fedavg", choices=AGGREGATIONS,
        help="'fedavg' averages all weights at the BS every --aggregate-every slots; 'gossip' "
        "averages each agent with its neighbors within 300 m; 'none' is the DTE control",
    )
    parser.add_argument(
        "--aggregate-every", "--fedavg-every", dest="aggregate_every", type=int, default=50,
        help="Aggregate weights every N slots (FedAvg or gossip)",
    )
    parser.add_argument(
        "--gossip-neighbors", type=int, default=None,
        help="Gossip only: pull at most S random neighbors per round (default: all within 300 m)",
    )
    parser.add_argument(
        "--buffer-capacity", type=int, default=None,
        help="Transitions each agent's replay buffer keeps (default: D3QNAgent's, 10 000 = 50 episodes)",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=1.0,
        help="Multiplies the reward the agents learn from (the CSV keeps the environment's reward)",
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Discount factor of the agents' Q-learning targets (default: D3QNAgent's, 0.99)",
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=None,
        help="Width of each hidden layer of the Q-network (default: two of 128; the paper's Table III uses 500 250 120)",
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05,
        help="Exploration floor, reached at 60%% of the training (the paper's Table III uses 0.1)",
    )
    parser.add_argument(
        "--reward-end-of-slot-queue", action="store_true",
        help="Judge the reward's queue term on the queue each slot leaves, not the one it starts with "
        "(V2VEnvParams.reward_end_of_slot_queue; not the paper's Eq. 22)",
    )
    parser.add_argument(
        "--observe-own-channel", action="store_true",
        help="Add to each agent's state a one-hot of the channel it picked in the last slot "
        "(V2VEnvParams.observe_own_channel; not part of the paper's state)",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/rewards.csv"))
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="'auto' picks cuda if available, else cpu",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device} | aggregation: {args.aggregation}")

    params = V2VEnvParams(
        num_cues=15,
        num_v2v_pairs=args.num_v2v_pairs,
        num_power_levels=3,
        max_steps=200,
        observe_own_channel=args.observe_own_channel,
        reward_end_of_slot_queue=args.reward_end_of_slot_queue,
    )
    # Only the options given on the command line; the rest keep D3QNAgent's defaults.
    agent_kwargs = {
        name: value
        for name, value in (
            ("buffer_capacity", args.buffer_capacity),
            ("gamma", args.gamma),
            ("hidden_sizes", args.hidden_sizes),
        )
        if value is not None
    }
    start = time.perf_counter()
    records, agents = run_training(
        params,
        args.episodes,
        args.seed,
        args.aggregate_every,
        args.log_every,
        device=device,
        aggregation=args.aggregation,
        gossip_neighbors=args.gossip_neighbors,
        agent_kwargs=agent_kwargs,
        reward_scale=args.reward_scale,
        epsilon_end=args.epsilon_end,
    )
    wall_time_s = time.perf_counter() - start
    first_agent = next(iter(agents.values()))

    write_records_csv(records, args.output)
    write_run_metadata(
        args.output,
        run_metadata(
            wall_time_s,
            device,
            method="dte" if args.aggregation == "none" else args.aggregation,
            seed=args.seed,
            episodes=args.episodes,
            num_v2v_pairs=args.num_v2v_pairs,
            aggregate_every=args.aggregate_every,
            gossip_neighbors=args.gossip_neighbors,
            buffer_capacity=first_agent.buffer.capacity,
            gamma=first_agent.gamma,
            observe_own_channel=params.observe_own_channel,
            reward_end_of_slot_queue=params.reward_end_of_slot_queue,
            hidden_sizes=list(first_agent.online.hidden_sizes),
            epsilon_end=args.epsilon_end,
            reward_scale=args.reward_scale,
            model_bytes=model_num_bytes(first_agent.online),
        ),
    )
    print(f"\nSaved: {args.output} ({wall_time_s / 3600:.2f} h)")
    print_convergence_verdict(records, args.episodes)


if __name__ == "__main__":
    main()
