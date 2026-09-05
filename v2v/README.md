# V2V — Federated Multi-Agent RL

Primer experimento de la tesis (checklist, ver reunión 3): réplica simplificada de
Li et al. (2022), "Federated Multi-Agent Deep RL for Resource Allocation of V2V
Communications" — K pares V2V, cada uno con un D3QN, decidiendo canal y potencia
de transmisión sobre un entorno compartido (`v2v_env/`).

## Scripts de entrenamiento

- `train_node4.py` — K agentes D3QN entrenando de forma independiente, sin
  agregación (DTE puro). Sirve de control base.
- `train_node5.py` — el mismo entorno y agentes, con una capa de FedAvg encima:
  cada `--fedavg-every` slots, los pesos de la red online de cada agente se
  suben, se promedian ponderados por el tamaño de minibatch usado desde la
  última ronda (`agents/federated.py`), y el modelo global agregado se
  difunde de vuelta reemplazando los pesos locales.

## Nota técnica: por qué esto no es HFRL "de libro"

Los K agentes de este experimento **comparten un mismo entorno** — no corren
copias independientes del mismo problema, como asume la definición formal de
Horizontal Federated Reinforcement Learning (HFRL) en Qi, Zhou, Lei & Zheng
(2021), *"Federated Reinforcement Learning: Techniques, Applications, and Open
Challenges"*. Lo que se implementa aquí es más preciso llamarlo **DTE
(Decentralized Training and Execution) + FedAvg centralizado sobre pesos**: un
patrón que ni siquiera el survey de Qi et al. cubre limpiamente, porque el
supuesto de entornos independientes que sustenta HFRL no se cumple.
