"""
Script 1 — Entender la API ParallelEnv con un entorno built-in de PettingZoo.

Objetivo: familiarizarte con el ciclo reset/step de ParallelEnv antes de
tocar tu propio entorno custom. Usamos simple_spread (de la suite MPE),
un entorno cooperativo clásico donde varios agentes deben cubrir
posiciones objetivo — es un buen análogo conceptual a "K agentes actuando
simultáneamente" como tus V2V pairs.

Requisitos:
    pip install pettingzoo[mpe]
"""

from mpe2 import simple_spread_v3


def main():
    # N=3 agentes, 25 pasos máximo por episodio, sin penalización local extra
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, render_mode=None)

    # --- 1. reset() devuelve observaciones iniciales para TODOS los agentes ---
    observations, infos = env.reset(seed=42)

    print("=== Agentes activos ===")
    print(env.agents)  # ej: ['agent_0', 'agent_1', 'agent_2']

    print("\n=== Espacios de observación y acción (por agente) ===")
    for agent in env.agents:
        print(f"{agent}:")
        print(f"  observation_space: {env.observation_space(agent)}")
        print(f"  action_space:      {env.action_space(agent)}")

    print("\n=== Observación inicial de un agente ===")
    print(observations["agent_0"])

    # --- 2. Loop principal: todos los agentes actúan en el mismo paso ---
    # A diferencia de AECEnv (turnos secuenciales), aquí construyes UN
    # diccionario de acciones con la acción de cada agente, y step()
    # avanza el entorno para todos a la vez -- esto es lo que necesitas
    # para modelar V2V pairs decidiendo canal+potencia simultáneamente.
    step_count = 0
    while env.agents:  # env.agents se vacía cuando el episodio termina
        actions = {
            agent: env.action_space(agent).sample()  # política aleatoria
            for agent in env.agents
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)

        step_count += 1
        if step_count <= 3:  # solo mostramos los primeros pasos para no saturar
            print(f"\n--- Paso {step_count} ---")
            print("Acciones:", actions)
            print("Recompensas:", rewards)
            print("Terminado?:", terminations)

    print(f"\nEpisodio terminado después de {step_count} pasos.")
    env.close()


if __name__ == "__main__":
    main()
