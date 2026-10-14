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
| `analysis.py` | Carga de las corridas del barrido y métricas que se calculan al final (episodios hasta converger) |
| `plots.py` | Colores, etiquetas y gráficas compartidas por los notebooks |
| `jobs/` | Scripts de Slurm para Hypatia; los logs quedan en `jobs/logs/` |
| `notebooks/` | `formulas_v2v_env.ipynb` (cada fórmula del entorno, verificada contra el código), `results.ipynb` (K = 5), `greedy.ipynb` y `fedavg.ipynb` (barrido de K) |
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

# Greedy: canal con menos vecinos en el slot anterior (empates al azar), potencia máxima
uv run python train_greedy.py --episodes 4000 --seed 0 --output runs/k5/greedy_seed0.csv
```

### Opciones de `main.py`

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--aggregation` | `fedavg` | `fedavg` promedia los pesos de las redes online; `none` es DTE puro |
| `--fedavg-every` | `50` | Cada cuántos slots se agrega (con `fedavg`) y se mide la dispersión de pesos (con los dos) |
| `--episodes` | `300` | Episodios de 200 slots |
| `--seed` | `0` | Semilla de la corrida (ver abajo) |
| `--num-v2v-pairs` | `5` | K, número de pares V2V (agentes) |
| `--device` | `auto` | `cuda` si hay GPU, si no `cpu` |
| `--log-every` | `10` | Cada cuántos episodios imprime la recompensa promedio |
| `--output` | `runs/rewards.csv` | CSV de salida |

`train_greedy.py` acepta `--episodes`, `--seed`, `--num-v2v-pairs` y `--output`
(por defecto `runs/greedy_rewards.csv`).

### Salida

Los dos scripts escriben las mismas columnas, una fila por episodio y agente. Las marcadas
«celda» son de todo el sistema y se repiten igual en las K filas del episodio.

| Columna | Significado |
|---|---|
| `episode`, `agent` | Episodio y par V2V (`v2v_0` … `v2v_{K-1}`) |
| `reward` | Recompensa compartida sumada sobre los 200 slots del episodio (igual para todos los agentes) |
| `cue_rate_mbps` | Celda. Tasa promedio de las CUEs en Mbps, promediada sobre los slots |
| `v2v_delay_ms` | Retardo de la cola de ese par por la ley de Little (Q/λ), promediado sobre los slots |
| `outage_rate` | Fracción de slots en que ese par incumple la restricción de confiabilidad (Ec. 15) |
| `neighbors` | Pares a 300 m o menos de ese par (fijo dentro del episodio) |
| `co_channel_rate` | Fracción de slots en que el par transmite en un canal donde también transmite otro par. Un par a potencia 0 no ocupa el canal |
| `channels_used` | Celda. Canales distintos con al menos un par transmitiendo, promediado sobre los slots |
| `mean_power_ratio` | Potencia promedio del par como fracción de P_max |
| `max_power_rate` | Fracción de slots a P_max |
| `train_loss` | Pérdida promedio del D3QN de ese par en el episodio. `nan` si no entrenó (Greedy, o antes de llenar el primer minibatch) |
| `weight_dispersion` | Celda. Distancia euclidiana promedio de los pesos de cada agente al promedio de todos, medida justo antes de cada ronda de agregación (en DTE, en los mismos instantes). `nan` en Greedy |
| `comm_models` | Celda. Transferencias de un modelo entre dos nodos en el episodio. FedAvg: 2K por ronda; DTE y Greedy: 0 |
| `comm_bytes` | Celda. `comm_models` × tamaño de un modelo (la red online: 30 397 parámetros float32 = 121 588 bytes con K y M por defecto) |
| `comm_max_load` | Celda. Transferencias en las que participó el nodo más cargado (un vehículo o la BS). En FedAvg es la BS |

Junto a cada CSV queda un JSON con el mismo nombre (`fedavg_seed0.json`) con la configuración
de la corrida, la máquina, la GPU si la hubo y el tiempo total (`wall_time_s`).

Los episodios hasta converger se calculan después, sobre la curva completa
(`analysis.convergence_episode`): primer episodio desde el cual la media móvil de 50 episodios
de la recompensa queda dentro de ±5 % de su valor final (el promedio del último 10 %).
`notebooks/results.ipynb` arma las figuras y las tablas con todas estas métricas.

### Semillas

`--seed` fija la geometría de cada episodio y la inicialización de cada agente. Corridas
con semillas distintas no comparten aleatoriedad. Con la misma semilla, FedAvg, DTE y
Greedy ven exactamente las mismas posiciones episodio a episodio, así que la comparación
entre métodos es pareada.

## Correr en Hypatia

Desde la raíz del repo, en Hypatia:

```bash
make run-greedy   # 18 trabajos de CPU: Greedy x K = 5, 10, ..., 30 x semillas 0, 1, 2
make run-fedavg   # 18 trabajos de GPU: lo mismo con FedAvg
make sq           # ver la cola
```

Cada trabajo corre `jobs/baseline.sbatch <metodo> <semilla> [episodios] [K]` y escribe
`v2v/runs/k<K>/<metodo>_seed<semilla>.csv` y su `.json`. Mientras corre, el progreso
(`Episode 10/4000 | ...`) aparece en `jobs/logs/slurm-v2v-<metodo>-k<K>-s<semilla>-<id>.out`.

- Los targets sincronizan el entorno una sola vez antes de enviar, y los trabajos usan
  `uv run --no-sync`. Si cada trabajo sincronizara el entorno, los que arrancan al mismo tiempo
  se pisarían instalando paquetes y fallarían.
- Saltan las corridas cuyo CSV ya existe: volver a correr el target, con `make sq` vacío,
  relanza solo las que faltan o fallaron.
- El límite de tiempo de FedAvg crece con K (`FEDAVG_MINUTES_PER_K`, 90 min por par: 7.5 h con
  K = 5 y 45 h con K = 30). El CSV se escribe al final, así que un trabajo que se pasa del límite
  se pierde entero.

Todo se puede cambiar sin editar nada:

```bash
make run-fedavg KS="5 10" SEEDS="0" EPISODES=500
make run-fedavg SBATCH=echo   # muestra los sbatch sin enviarlos
```

DTE todavía no tiene target: se lanza con el mismo script, por ejemplo
`sbatch -p gpu --gres=gpu:1 --time=450 jobs/baseline.sbatch dte 0 4000 5` desde `v2v/`, después
de un `uv sync`.

Para traer los resultados a la máquina local, desde la raíz: `make pull-runs`. Después,
`notebooks/greedy.ipynb` y `notebooks/fedavg.ipynb` analizan el barrido de K (todas las métricas
vs. K, con la Fig. 3 de Li et al. como referencia; FedAvg comparado con Greedy semilla por
semilla) y `notebooks/results.ipynb` las curvas de aprendizaje con K = 5.

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
