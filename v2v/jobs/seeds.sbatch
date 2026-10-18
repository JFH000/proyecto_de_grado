#!/bin/bash
# Varias semillas de un mismo metodo y K en un solo trabajo, compartiendo la GPU: la red es tan
# pequena que una sola corrida deja la GPU casi siempre ociosa. Cada semilla es un proceso aparte,
# lanzado con jobs/baseline.sbatch, asi que deja exactamente la misma salida que un trabajo propio.
#   sbatch [opciones de particion] jobs/seeds.sbatch <fedavg|dte|gossip|greedy|umbral> <K> <episodios> <semilla>...
# `make run-<metodo>` (raiz del repo) lo usa para lanzar el barrido de K: un trabajo por K, con
# --cpus-per-task = 2 por semilla. El entorno tiene que estar sincronizado antes (`uv sync`, que los
# targets del Makefile ya hacen).
# Salida por semilla: runs/k<K>/<metodo>_seed<semilla>.csv y su .json. El progreso de cada semilla
# queda en jobs/logs/<nombre del trabajo>-<id>-s<semilla>.out; el .out del trabajo solo resume.
#SBATCH --job-name=v2v-seeds
#SBATCH --mem=32gb
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=END,FAIL

set -euo pipefail

METHOD=${1:?falta el metodo: fedavg, dte, gossip, greedy o umbral}
K=${2:?falta K}
EPISODES=${3:?faltan los episodios}
shift 3
if [ "$#" -eq 0 ]; then
  echo "faltan las semillas" >&2
  exit 1
fi
SEEDS=("$@")

# Hilos de CPU por proceso: los CPUs del trabajo repartidos entre las semillas. Sin esto, cada proceso
# abriria tantos hilos como nucleos tiene el nodo y se estorbarian entre si.
THREADS=$(( ${SLURM_CPUS_PER_TASK:-${#SEEDS[@]}} / ${#SEEDS[@]} ))
if [ "$THREADS" -lt 1 ]; then THREADS=1; fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

LOG_PREFIX="jobs/logs/${SLURM_JOB_NAME:-v2v-seeds}-${SLURM_JOB_ID:-local}"
echo "metodo=${METHOD} K=${K} episodios=${EPISODES} semillas=${SEEDS[*]} hilos_por_semilla=${THREADS}"

# Varios procesos solo pueden compartir una GPU en modo de computo Default. Si la GPU esta en un modo
# exclusivo, el segundo proceso fallaria al iniciar CUDA: mejor parar aqui, antes de lanzar nada.
if [ "${#SEEDS[@]}" -gt 1 ] && command -v nvidia-smi > /dev/null; then
  GPU_MODES=$(nvidia-smi --query-gpu=name,compute_mode --format=csv,noheader 2> /dev/null || true)
  echo "GPU: ${GPU_MODES:-sin informacion de nvidia-smi}"
  if [[ "$GPU_MODES" == *Exclusive* || "$GPU_MODES" == *Prohibited* ]]; then
    echo "la GPU no deja que varios procesos la compartan (${GPU_MODES}); lanza una semilla por trabajo" >&2
    exit 1
  fi
fi

pids=()
for seed in "${SEEDS[@]}"; do
  log="${LOG_PREFIX}-s${seed}.out"
  echo "semilla ${seed} -> runs/k${K}/${METHOD}_seed${seed}.csv (progreso en ${log})"
  bash jobs/baseline.sbatch "$METHOD" "$seed" "$EPISODES" "$K" > "$log" 2>&1 &
  pids+=("$!")
done

# Espera a todas: si una falla, las demas siguen hasta terminar y el trabajo sale con error.
failed=()
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed+=("${SEEDS[$i]}")
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "fallaron las semillas ${failed[*]} (ver sus .out); las demas terminaron" >&2
  exit 1
fi
echo "terminaron todas las semillas: ${SEEDS[*]}"
