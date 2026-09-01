"""
Script 2 — Esqueleto MINIMAL de un entorno custom V2V-like con ParallelEnv.

Objetivo: probar si ParallelEnv soporta lo que necesitas para el problema
de Li et al. (FedMARL para V2V), ANTES de invertir tiempo en construir
el entorno real con el modelo de canal completo:

  1. K agentes, cada uno con su PROPIO espacio de observación (no un
     estado global compartido) -- refleja la observabilidad parcial.
  2. Cada agente elige una acción compuesta (canal + nivel de potencia).
  3. Una recompensa GLOBAL compartida entre todos los agentes (no una
     recompensa individual por agente) -- refleja la ecuación 22 del
     paper, donde el reward se calcula en la BS a partir de la acción
     conjunta.

Esto NO es una implementación fiel del paper (faltan CSI real, fading,
interferencia, colas, etc.) -- es un esqueleto para validar que la API
de PettingZoo se comporta como esperas antes de meterle la física real.

Requisitos:
    pip install pettingzoo gymnasium numpy
"""

import functools
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv


class SimpleV2VEnv(ParallelEnv):
    metadata = {"name": "simple_v2v_v0"}

    def __init__(self, num_agents=4, num_channels=5, num_power_levels=4, max_steps=50):
        self.num_agents_ = num_agents
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.max_steps = max_steps

        self.possible_agents = [f"v2v_{i}" for i in range(num_agents)]

        # Placeholder de "estado del mundo" -- aquí luego iría el modelo de
        # canal real (fading, posiciones, etc.). Por ahora es ruido.
        self._channel_quality = None
        self._step_count = 0

    # --- Espacios: uno por agente, tal como exige la observabilidad parcial ---

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Placeholder: [calidad de canal propia (num_channels valores),
        #               backlog de cola normalizado (1 valor)]
        # En la versión real esto sería CSI + interferencia + backlog +
        # selecciones previas de vecinos, como en la Sección IV-A de Li et al.
        return Box(low=0.0, high=1.0, shape=(self.num_channels + 1,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Acción compuesta (canal, nivel de potencia) codificada como un
        # único Discrete -- más simple para empezar. Más adelante puedes
        # cambiar a MultiDiscrete([num_channels, num_power_levels]).
        return Discrete(self.num_channels * self.num_power_levels)

    def _decode_action(self, action_id):
        channel = action_id // self.num_power_levels
        power_level = action_id % self.num_power_levels
        return channel, power_level

    # --- Ciclo de vida del entorno ---

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._step_count = 0

        # Placeholder de calidad de canal aleatoria por agente
        rng = np.random.default_rng(seed)
        self._channel_quality = {
            agent: rng.random(self.num_channels + 1).astype(np.float32)
            for agent in self.agents
        }

        observations = {agent: self._channel_quality[agent] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self._step_count += 1

        decoded = {agent: self._decode_action(a) for agent, a in actions.items()}

        # --- Placeholder de recompensa GLOBAL compartida ---
        # En el paper real (ec. 22), esto combina sum-rate de CUEs,
        # penalización de delay, y penalización de outage/confiabilidad.
        # Aquí, como placeholder, penalizamos si muchos agentes eligen
        # el MISMO canal (proxy tosco de "interferencia por colisión").
        channels_chosen = [ch for ch, _ in decoded.values()]
        collisions = len(channels_chosen) - len(set(channels_chosen))
        global_reward = -float(collisions)  # más colisiones = peor para todos

        # Todos los agentes reciben la MISMA recompensa (recompensa compartida)
        rewards = {agent: global_reward for agent in self.agents}

        # Nueva observación placeholder (aquí recalcularías CSI/interferencia reales)
        rng = np.random.default_rng(self._step_count)
        observations = {
            agent: rng.random(self.num_channels + 1).astype(np.float32)
            for agent in self.agents
        }

        terminations = {agent: False for agent in self.agents}
        truncations = {
            agent: self._step_count >= self.max_steps for agent in self.agents
        }
        infos = {agent: {"channel": decoded[agent][0], "power": decoded[agent][1]}
                 for agent in self.agents}

        if all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


def main():
    env = SimpleV2VEnv(num_agents=4, num_channels=5, num_power_levels=4, max_steps=10)
    observations, infos = env.reset(seed=0)

    print("Agentes:", env.agents)
    print("Observation space (agente 0):", env.observation_space("v2v_0"))
    print("Action space (agente 0):", env.action_space("v2v_0"))

    step = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        step += 1
        print(f"\nPaso {step} — recompensa global compartida: {rewards['v2v_0']}")
        print("  Info por agente (canal, potencia):",
              {a: (i["channel"], i["power"]) for a, i in infos.items()})

    env.close() if hasattr(env, "close") else None
    print(f"\nEpisodio de prueba terminado en {step} pasos.")


if __name__ == "__main__":
    main()
