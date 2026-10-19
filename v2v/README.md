# V2V — Federated Multi-Agent RL

Réplica simplificada de Li et al. (2022), "Federated Multi-Agent Deep RL for Resource
Allocation of V2V Communications": K pares V2V comparten las M = 15 bandas de las CUEs
(usuarios celulares) y cada par, con su propio D3QN, decide qué banda reusar y con qué
potencia transmitir. Es la línea base contra la que se compara la agregación
descentralizada de la tesis (gossip entre vecinos, `agents/gossip.py`), que también está aquí.

## Estructura

| Ruta | Qué hay |
|---|---|
| `v2v_env/` | Entorno PettingZoo: canal, SINR, cola, recompensa (Sec. IV-A del paper) |
| `agents/` | D3QN con replay priorizado, FedAvg (`federated.py`), gossip entre vecinos (`gossip.py`) y la política Greedy |
| `main.py` | Entrena los K agentes con FedAvg o sin agregación (DTE) |
| `train_greedy.py` | Corre el baseline Greedy, sin entrenamiento |
| `analysis.py` | Carga de las corridas del barrido y métricas que se calculan al final (episodios hasta converger) |
| `plots.py` | Colores, etiquetas y gráficas compartidas por los notebooks |
| `jobs/` | Scripts de Slurm para Hypatia; los logs quedan en `jobs/logs/` |
| `notebooks/` | `formulas_v2v_env.ipynb` (cada fórmula del entorno, verificada contra el código), `results.ipynb` (K = 5), `greedy.ipynb`, `fedavg.ipynb`, `dte.ipynb` y `gossip.ipynb` (barrido de K) |
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

# Gossip (la propuesta): cada 50 slots, cada agente promedia con los pares a 300 m o menos, sin la BS
uv run python main.py --aggregation gossip --episodes 4000 --seed 0 --output runs/k5/gossip_seed0.csv

# Greedy: canal con menos vecinos en el slot anterior (empates al azar), potencia máxima
uv run python train_greedy.py --episodes 4000 --seed 0 --output runs/k5/greedy_seed0.csv

# Regla de umbral (no está en el paper): el canal como Greedy; enlace fuerte a potencia máxima,
# enlace débil callado hasta que su cola llega al 90 % de Q_max
uv run python train_greedy.py --policy umbral --episodes 4000 --seed 0 --output runs/k5/umbral_seed0.csv
```

### Opciones de `main.py`

| Flag | Por defecto | Qué hace |
|---|---|---|
| `--aggregation` | `fedavg` | `fedavg` promedia los pesos de las redes online en la BS; `gossip` promedia cada agente con sus vecinos a 300 m o menos; `none` es DTE puro |
| `--aggregate-every` | `50` | Cada cuántos slots se agrega (FedAvg o gossip) y se mide la dispersión de pesos (en todos). `--fedavg-every` es el nombre anterior y sigue funcionando |
| `--gossip-neighbors` | todos | Solo gossip: cada agente toma a lo sumo S vecinos al azar por ronda (como Ge et al., 2024), sorteados de nuevo en cada ronda |
| `--episodes` | `300` | Episodios de 200 slots |
| `--seed` | `0` | Semilla de la corrida (ver abajo) |
| `--num-v2v-pairs` | `5` | K, número de pares V2V (agentes) |
| `--device` | `auto` | `cuda` si hay GPU, si no `cpu` |
| `--gamma` | `0.99` | Descuento γ de los objetivos de Q-learning (el paper no lo da) |
| `--lr` | `0.001` | Tasa de aprendizaje (Adam) de la red Q de cada agente |
| `--observe-own-channel` | apagado | Agrega al estado de cada agente un one-hot del canal que eligió en el slot anterior (no está en el estado del paper; ver abajo) |
| `--buffer-capacity` | `10000` | Transiciones que guarda el replay buffer de cada agente |
| `--reward-scale` | `1` | Multiplica la recompensa de la que aprenden los agentes; el CSV guarda la del entorno |
| `--hidden-sizes` | `128 128` | Ancho de cada capa oculta de la red Q; la Tabla III del paper usa `500 250 120` |
| `--epsilon-end` | `0.05` | Piso de ε, al que llega al 60 % del entrenamiento; la Tabla III usa `0.1` |
| `--reward-end-of-slot-queue` | apagado | El término de cola de la recompensa juzga la cola con que termina el slot, no la de su inicio (no es la Ec. 22; ver abajo) |
| `--local-reward` | apagado | Cada agente aprende de su propia recompensa (su cola, su confiabilidad y el CUE de su canal) en vez de la compartida; el CSV guarda la compartida (no es la Ec. 22; ver abajo) |
| `--log-every` | `10` | Cada cuántos episodios imprime la recompensa promedio |
| `--output` | `runs/rewards.csv` | CSV de salida |

`train_greedy.py` acepta `--policy` (`greedy` o `umbral`), `--episodes`, `--seed`,
`--num-v2v-pairs` y `--output` (por defecto `runs/greedy_rewards.csv`).

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
| `comm_models` | Celda. Transferencias de un modelo entre dos nodos en el episodio. FedAvg: 2K por ronda; gossip: una por vecino de cada agente por ronda; DTE y Greedy: 0 |
| `comm_bytes` | Celda. `comm_models` × tamaño de un modelo (la red online: 30 397 parámetros float32 = 121 588 bytes con K y M por defecto) |
| `comm_max_load` | Celda. Transferencias en las que participó el nodo más cargado (un vehículo o la BS). En FedAvg es la BS; en gossip, el vehículo con más vecinos |

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
make run-greedy   # 6 trabajos de CPU: Greedy, uno por K = 5, 10, ..., 30, con las semillas 0, 1, 2
make run-fedavg   # 6 trabajos de GPU: lo mismo con FedAvg
make run-dte      # 6 trabajos de GPU: lo mismo con DTE
make run-gossip   # 6 trabajos de GPU: lo mismo con gossip
make sq           # ver la cola
make run-cpu-test # prueba de velocidad en CPU: FedAvg corto con K = 10 y 30 en la particion short
```

Cada trabajo corre `jobs/seeds.sbatch <metodo> <K> <episodios> <semillas...>`: las semillas de un
mismo K corren a la vez, compartiendo la GPU (la red es tan pequeña que una corrida sola la deja casi
siempre ociosa, y la QoS de Hypatia da 2 GPUs por usuario). Cada semilla es un proceso aparte,
lanzado con `jobs/baseline.sbatch`, y deja exactamente la misma salida que si corriera sola:
`v2v/runs/k<K>/<metodo>_seed<semilla>.csv` y su `.json`. El progreso de cada semilla
(`Episode 10/4000 | ...`) queda en `jobs/logs/v2v-<metodo>-k<K>-s<semillas>-<id>-s<semilla>.out`;
el `.out` del trabajo (`slurm-...`) solo dice qué semillas lanzó y cuáles fallaron.

- Los targets sincronizan el entorno una sola vez antes de enviar, y los trabajos usan
  `uv run --no-sync`. Si cada trabajo sincronizara el entorno, los que arrancan al mismo tiempo
  se pisarían instalando paquetes y fallarían.
- Saltan las semillas cuyo CSV ya existe y las que ya están en la cola (corriendo o esperando,
  según el nombre del trabajo), así que volver a correr el target solo envía las que faltan o
  fallaron. Si una semilla falla, las otras del mismo trabajo siguen hasta terminar.
- Cada trabajo pide 2 CPUs por semilla (`CPUS_PER_SEED`) y cada proceso usa solo esos hilos.
- El límite de tiempo de FedAvg, DTE y gossip crece con K (`LEARNER_MINUTES_PER_K`, 150 min por
  par: 12.5 h con K = 5 y 75 h con K = 30), con margen para que las semillas vayan algo más lentas
  al compartir la GPU. El CSV se escribe al final, así que un trabajo que se pasa del límite se
  pierde entero.
- Para correr una sola corrida a mano: `sbatch -p gpu --gres=gpu:1 --time=750 jobs/baseline.sbatch
  fedavg 0 4000 5` desde `v2v/`, después de un `uv sync`.

Todo se puede cambiar sin editar nada:

```bash
make run-fedavg KS="5 10" SEEDS="0" EPISODES=500
make run-fedavg SBATCH=echo   # muestra los sbatch sin enviarlos
```

### Diagnóstico del entrenamiento

`make run-diag DIAG_SET=<conjunto>` envía un solo trabajo de GPU (`jobs/diag.sbatch`) que corre a la
vez las variantes de `jobs/diag_variants/<conjunto>.txt`, todas con K = 10, la semilla 0 y 2000
episodios. Cada línea del archivo es una variante: su nombre y las opciones de `main.py` que cambia
(sin opciones es la configuración por defecto; `--aggregation none` o `gossip` cambian el método).
Deja `v2v/runs/diag/<variante>/k10/<método>_seed0.csv`, fuera de `runs/k<K>/`, así que no se mezcla
con el barrido, y se analiza con `notebooks/diagnostico.ipynb` (que lee el mismo archivo). Las
variantes que ya tienen CSV no se vuelven a correr. K, episodios y semilla se cambian con
`make run-diag DIAG_K=5 DIAG_EPISODES=1000 DIAG_SEED=1`.

| Conjunto | Pregunta | Resultado |
|---|---|---|
| `caida` | ¿FedAvg se cae cuando ε llega a su piso por el buffer chico (`--buffer-capacity 100000`) o por la recompensa sin escalar (`--reward-scale 0.01`)? | No: ninguna variante deja de caerse ni le gana a Greedy |
| `aprendizaje` | ¿El agente no encuentra la regla de umbral por la red chica (`--hidden-sizes 500 250 120`), por dejar de explorar (`--epsilon-end 0.1`) o por γ = 0.99 (`--gamma 0.9`)? | No le gana a Greedy ninguna. γ = 0.9 casi quita la caída (−8 % contra −34 %) y termina en ~4700; la red y el ε del paper apenas ayudan |
| `cola` | ¿Aprende a transmitir cuando la cola se llena si el silencio se cobra en el slot que la desborda (`--reward-end-of-slot-queue`), con γ 0.99, 0.9 o 0.5 y con el canal propio en el estado (`--observe-own-channel`)? | No: sola no cambia nada (−33 %); con γ = 0.9 queda como γ = 0.9 sin ella (~4850, la mejor, pero ~800 por debajo de Greedy); γ = 0.5 queda peor |
| `recompensa` | ¿Aprende la regla de umbral si cada agente recibe su propia recompensa (`--local-reward`), sola, con γ = 0.9, y con γ = 0.9 más la cola al final del slot o el canal propio en el estado? | Es la mayor mejora: las cuatro superan a DTE y a toda variante con la recompensa compartida, y a mitad del entrenamiento a Greedy (5800–6050). Al terminar la exploración olvidan una parte; la mejor (con γ = 0.9 y la cola al final del slot) termina en ~5590, casi igual a Greedy |
| `gamma` | ¿Hay un γ mejor cerca de 0.9 (0.95, 0.85, 0.8, 0.7)? | Pendiente |
| `estabilidad` | Sobre la mejor de `recompensa`, ¿una tasa de aprendizaje más baja (`--lr 3e-4` o `1e-4`), más exploración (`--epsilon-end 0.1`) o un buffer más grande (`--buffer-capacity 100000`) evitan que olvide al terminar la exploración? | Pendiente |

Cada conjunto es un trabajo aparte. La QoS deja 2 GPUs a la vez: un tercer trabajo espera en la
cola hasta que se libere una. El notebook muestra `recompensa`, `gamma` y `estabilidad` juntos, con
Greedy, DTE y la regla de umbral como referencias (`make run-umbral` la corre para todo K en CPU).

Para traer los resultados a la máquina local, desde la raíz: `make pull-runs`. Después,
`notebooks/greedy.ipynb`, `fedavg.ipynb`, `dte.ipynb` y `gossip.ipynb` analizan el barrido de K
(todas las métricas vs. K, con la Fig. 3 de Li et al. como referencia; FedAvg comparado con Greedy,
DTE con FedAvg y gossip con FedAvg y DTE, semilla por semilla) y `notebooks/results.ipynb` las
curvas de aprendizaje con K = 5.

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

**Canal propio en el estado (opcional, apagado por defecto).** Con `--observe-own-channel`
(`V2VEnvParams.observe_own_channel`), el estado agrega un one-hot del canal que el par eligió en el
slot anterior, entre los conteos de vecinos y la cola: 4M + 1 entradas en vez de las 3M + 1 del
paper. Los conteos del paper excluyen al propio par, así que sin esto un agente no distingue el
canal que está usando de uno que nadie usa, y no puede aprender a quedarse en un canal que ningún
vecino tomó. Con políticas fijas, K = 10 y 20 episodios, quedarse en el canal mientras ningún vecino
lo use y saltar a uno libre si alguno lo toma sube la recompensa ~20 % frente a Greedy (más con
K = 20). No cambia la dinámica ni la recompensa: con las mismas acciones, el entorno da lo mismo
con y sin esta opción.

**Parámetros que el paper no da.** A, p_o, R_min, la tasa de llegada λ, la duración del
slot τ y la geometría de la celda están en `v2v_env/params.py`, marcados como tales. La Tabla III
tampoco da el descuento γ de D3QN: `D3QNAgent` usa 0.99 y `--gamma` lo cambia. Con λ = 1 Mbps y
potencia máxima, la cola se vacía en cada slot (retardo de ~0 ms frente a un máximo de 100 ms). La
restricción de cola solo se activa cuando un par deja de transmitir: con potencia cero la cola llega
a Q_max en 100 slots, y ese es el único costo que frena a un par de callarse para proteger al CUE de
su canal.

**Cola al final del slot (opcional, apagado por defecto).** Con `--reward-end-of-slot-queue`
(`V2VEnvParams.reward_end_of_slot_queue`), el término de cola de la recompensa juzga la cola que
deja el slot en vez de Q_k^t, la de su inicio. En la Ec. 22 la acción de un slot no cambia su propio
término de cola: el slot de silencio que llena la cola lo paga el siguiente. Con la opción lo paga
el mismo slot. Las observaciones, las métricas (el retardo sigue saliendo de Q_k^t) y la dinámica no
cambian.

**Recompensa local (opcional, apagada por defecto).** En la Ec. 22 la recompensa es una sola para
todos: los K agentes reciben el mismo número en cada slot, y ninguno distingue lo que causó él de lo
que hicieron los demás. El entorno reporta además, en el `info` de cada paso, una recompensa por par
(`local_reward`, `reward.compute_local_rewards`): los mismos tres términos, pero solo los que toca
la acción del par, es decir el CUE del canal que eligió (todos los pares de un canal se juzgan por el
mismo CUE), su propia cola y su propia confiabilidad. No cuenta la interferencia que el par causa a
los otros pares V2V de su canal. Con `--local-reward` cada agente aprende de ella; el CSV sigue
guardando la recompensa compartida, así que las corridas se comparan igual. Sigue la misma regla de
tiempo de la cola que la compartida (`--reward-end-of-slot-queue` la cambia en las dos).

**Regla de umbral (`agents/threshold_policy.py`, no está en el paper).** Una referencia sin
entrenar, solo con información local: el canal como Greedy, potencia máxima si el enlace propio es
fuerte (ganancia media sobre los canales de −91 dB o más, la de un enlace de 25 m) y silencio si es
débil, hasta que la cola llega al 90 % de Q_max. Le gana a Greedy con todo K porque calla a los
enlaces que interfieren al CUE más de lo que ganan en fiabilidad, sin dejar que sus colas se
desborden. Muestra que hay algo que aprender: un agente que no la alcanza no está aprendiendo bien.

### Por qué esto no es HFRL "de libro"

Los K agentes de este experimento **comparten un mismo entorno**: no corren copias
independientes del mismo problema, como asume la definición formal de Horizontal Federated
Reinforcement Learning (HFRL) en Qi, Zhou, Lei & Zheng (2021), *"Federated Reinforcement
Learning: Techniques, Applications, and Open Challenges"*. Lo que se implementa aquí es más
preciso llamarlo **DTE (Decentralized Training and Execution) + FedAvg centralizado sobre
pesos**: un patrón que ni siquiera el survey de Qi et al. cubre limpiamente, porque el
supuesto de entornos independientes que sustenta HFRL no se cumple.
