# V2V — Federated Multi-Agent RL

Réplica simplificada de Li et al. (2022), "Federated Multi-Agent Deep RL for Resource
Allocation of V2V Communications": K pares V2V comparten las M = 15 bandas de las CUEs
(usuarios celulares) y cada par, con su propio D3QN, decide qué banda reusar y con qué
potencia transmitir. Es la línea base contra la que se compara la agregación
descentralizada de la tesis.

## Estructura

| Ruta | Qué hay |
|---|---|
| `v2v_env/` | Entorno PettingZoo: canal, SINR, cola, recompensa (Sec. IV-A del paper) |
| `agents/` | D3QN con replay priorizado, FedAvg (`federated.py`) y la política Greedy |
| `main.py` | Entrena los K agentes con FedAvg o sin agregación (DTE) |
| `train_greedy.py` | Corre el baseline Greedy, sin entrenamiento |
| `jobs/` | Scripts de Slurm para Hypatia; los logs quedan en `jobs/logs/` |
| `notebooks/` | `formulas_v2v_env.ipynb` (cada fórmula del entorno, verificada contra el código) y `results.ipynb` |
| `tests/` | Pruebas con pytest |

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11 (`.python-version`).

```bash
cd v2v
uv sync
uv run pytest        # todas las pruebas deben pasar
```

`torch` se instala desde el índice de PyTorch para CUDA 11.8. En una máquina sin GPU
funciona igual en CPU.

## Correr cada método

Todos los comandos se corren desde `v2v/`.

```bash
# FedAvg (réplica de Li et al.): los pesos se promedian cada 50 slots
uv run python main.py --aggregation fedavg --episodes 4000 --seed 0 --output runs/k5/fedavg_seed0.csv

# DTE: los mismos agentes, sin compartir pesos nunca (control)
uv run python main.py --aggregation none --episodes 4000 --seed 0 --output runs/k5/dte_seed0.csv

# Greedy: canal con menos vecinos en el slot anterior, potencia máxima
uv run python train_greedy.py --episodes 4000 --seed 0 --output runs/k5/greedy_seed0.csv
```

### Opciones de `main.py`

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--aggregation` | `fedavg` | `fedavg` promedia los pesos de las redes online; `none` es DTE puro |
| `--fedavg-every` | `50` | Cada cuántos slots se agrega (solo con `fedavg`) |
| `--episodes` | `300` | Episodios de 200 slots |
| `--seed` | `0` | Semilla de la corrida (ver abajo) |
| `--num-v2v-pairs` | `5` | K, número de pares V2V (agentes) |
| `--device` | `auto` | `cuda` si hay GPU, si no `cpu` |
| `--log-every` | `10` | Cada cuántos episodios imprime la recompensa promedio |
| `--output` | `runs/rewards.csv` | CSV de salida |

`train_greedy.py` acepta `--episodes`, `--seed`, `--num-v2v-pairs` y `--output`
(por defecto `runs/greedy_rewards.csv`).

### Salida

Los dos scripts escriben las mismas columnas, una fila por episodio y agente:

| Columna | Significado |
|---|---|
| `episode`, `agent` | Episodio y par V2V (`v2v_0` … `v2v_{K-1}`) |
| `reward` | Recompensa compartida sumada sobre los 200 slots del episodio (igual para todos los agentes) |
| `cue_rate_mbps` | Tasa promedio de las CUEs en Mbps, promediada sobre los slots |
| `v2v_delay_ms` | Retardo de la cola de ese par por la ley de Little (Q/λ), promediado sobre los slots |
| `outage_rate` | Fracción de slots en que ese par incumple la restricción de confiabilidad (Ec. 15) |

### Semillas

`--seed` fija la geometría de cada episodio y la inicialización de cada agente. Corridas
con semillas distintas no comparten aleatoriedad. Con la misma semilla, FedAvg, DTE y
Greedy ven exactamente las mismas posiciones episodio a episodio, así que la comparación
entre métodos es pareada.

## Correr en Hypatia

Desde la raíz del repo, en Hypatia:

```bash
make baseline-k5   # 9 trabajos: FedAvg, DTE y Greedy x semillas 0, 1, 2 (K = 5)
make sq            # ver la cola
```

Cada trabajo corre `jobs/baseline.sbatch <metodo> <semilla> [episodios] [K]` y escribe
`v2v/runs/k5/<metodo>_seed<semilla>.csv`. FedAvg y DTE piden una GPU (`-p gpu`) y Greedy
corre en CPU (`-p medium`). El tiempo y los episodios se pueden cambiar sin editar nada:

```bash
make baseline-k5 LEARNER_TIME=12:00:00 BASELINE_EPISODES=2000
```

Para traer los resultados a la máquina local, desde la raíz: `make pull-runs`.

## Supuestos y desviaciones respecto a Li et al. (2022)

Cada uno está documentado también en el código y en `notebooks/formulas_v2v_env.ipynb`.

**Movilidad.** Las posiciones no cambian dentro de un episodio. En cada `reset` se vuelven
a muestrear: las 15 CUEs y los transmisores V2V, uniformes en una celda de 500 m de radio,
y cada receptor a 10–50 m de su transmisor, en una dirección al azar. Como cada episodio
usa una semilla distinta, la geometría cambia de un episodio a otro, y con ella el
conjunto de vecinos a 300 m o menos de cada par. Dentro de un episodio sí cambia el
desvanecimiento rápido, que se sortea en cada slot. Un episodio dura 200 slots de 1 ms,
es decir 0,2 s: a 30 m/s (108 km/h) un vehículo avanza 6 m, poco frente al radio de
vecindad de 300 m, así que mover los vehículos dentro del episodio casi no cambiaría ni el
canal ni los vecinos. El supuesto es que cada episodio es una instantánea independiente de
la red y que la movilidad se modela como cambio de topología entre episodios. La
limitación es que no hay continuidad entre episodios consecutivos: un vehículo no sigue
cerca de los mismos vecinos, como sí pasaría con un modelo de movilidad real. Esto importa
para la agregación entre vecinos, porque es la movilidad la que haría circular los modelos
por la red.

**Canal.** Una sola ley de path-loss en vez de los tres casos LOS/WLOS/NLOS de la Ec. 4.

**Recompensa.** Dos cambios frente a la Ec. 22 impresa (`v2v_env/reward.py`). Primero, el
término de cola usa U(Q_max − Q_k) en vez de U(Q_k − Q_max), para que penalice el desborde
como pide la restricción (19e). Segundo, cada margen se divide por su umbral, para que los
tres términos queden en la misma escala que la penalización A: en unidades crudas el
término de tasa CUE valía ~3×10⁷ por slot y A = −10 no afectaba nada.

**Observaciones.** Los coeficientes de canal entran en dB, centrados en −100 dB y en
unidades de 10 dB, y la cola como fracción de Q_max (`encode_channel_gain` en
`v2v_env/env.py`). No cambia el MDP, solo la escala de las entradas de la red.

**Parámetros que el paper no da.** A, p_o, R_min, la tasa de llegada λ, la duración del
slot τ y la geometría de la celda están en `v2v_env/params.py`, marcados como tales. Con
λ = 1 Mbps la cola casi nunca crece (retardo de ~1 ms frente a un máximo de 100 ms), así
que la restricción de cola casi nunca se activa.

### Por qué esto no es HFRL "de libro"

Los K agentes de este experimento **comparten un mismo entorno**: no corren copias
independientes del mismo problema, como asume la definición formal de Horizontal Federated
Reinforcement Learning (HFRL) en Qi, Zhou, Lei & Zheng (2021), *"Federated Reinforcement
Learning: Techniques, Applications, and Open Challenges"*. Lo que se implementa aquí es más
preciso llamarlo **DTE (Decentralized Training and Execution) + FedAvg centralizado sobre
pesos**: un patrón que ni siquiera el survey de Qi et al. cubre limpiamente, porque el
supuesto de entornos independientes que sustenta HFRL no se cumple.
