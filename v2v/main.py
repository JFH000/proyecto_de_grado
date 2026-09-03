import numpy as np

from v2v_env.env import V2VEnv
from v2v_env.params import V2VEnvParams


def main():
    # Node 1 scope: M=15 CUEs/channels, K=5 V2V pairs.
    params = V2VEnvParams(num_cues=15, num_v2v_pairs=5, num_power_levels=3, max_steps=200)
    env = V2VEnv(params)
    obs, infos = env.reset(seed=0)

    print(f"Agents: {env.agents}")
    print(f"Observation space (v2v_0): {env.observation_space('v2v_0')}")
    print(f"Action space (v2v_0): {env.action_space('v2v_0')}")

    rewards_log = []
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        rewards_log.append(rewards["v2v_0"])  # shared reward, identical for every agent

    rewards_log = np.array(rewards_log)
    print(f"\nRan {len(rewards_log)} steps with random actions.")
    print(f"Shared reward: min={rewards_log.min():.3f}, mean={rewards_log.mean():.3f}, "
          f"max={rewards_log.max():.3f}, std={rewards_log.std():.3f}")
    print("(Sanity check, not learning: same reward reaches every agent each step, "
          "and values should be finite and vary with the random joint action.)")


if __name__ == "__main__":
    main()
