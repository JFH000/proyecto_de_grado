#!/bin/bash
# Diagnostico de la caida de FedAvg cuando termina la exploracion (epsilon llega a 0.05): cuatro
# variantes del entrenamiento con un mismo K y una misma semilla, a la vez en una GPU.
#   sbatch [opciones de particion] jobs/diag.sbatch [K] [episodios] [semilla]   (por defecto 10 2000 0)
# `make run-diag` (raiz del repo) lo envia. Con 2000 episodios, epsilon llega al piso en el
# episodio 1200 (el 60 % del entrenamiento), asi que la caida deberia verse antes de la mitad.
#   base         la configuracion de las corridas de la tesis
#   buffer100k   replay buffer de 100 000 transiciones (500 episodios) en vez de 10 000 (50)
#   reward0.01   los agentes aprenden de la recompensa x 0.01 (el CSV guarda la del entorno)
#   ambos        las dos cosas
# Salida: runs/diag/<variante>/k<K>/fedavg_seed<semilla>.csv y su .json, fuera de runs/k<K>/
# para no mezclarse con el barrido. Progreso: jobs/logs/<nombre del trabajo>-<id>-<variante>.out.
# Analisis: notebooks/diagnostico.ipynb.
#SBATCH --job-name=v2v-diag
#SBATCH --mem=32gb
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1

K=${1:-10}
EPISODES=${2:-2000}
SEED=${3:-0}

NAMES=(base buffer100k reward0.01 ambos)
FLAGS=(
  ""
  "--buffer-capacity 100000"
  "--reward-scale 0.01"
  "--buffer-capacity 100000 --reward-scale 0.01"
)

THREADS=$(( ${SLURM_CPUS_PER_TASK:-${#NAMES[@]}} / ${#NAMES[@]} ))
if [ "$THREADS" -lt 1 ]; then THREADS=1; fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

LOG_PREFIX="jobs/logs/${SLURM_JOB_NAME:-v2v-diag}-${SLURM_JOB_ID:-local}"
echo "K=${K} episodios=${EPISODES} semilla=${SEED} variantes=${NAMES[*]} hilos_por_variante=${THREADS}"

# Igual que jobs/seeds.sbatch: varios procesos solo comparten la GPU en modo de computo Default.
if command -v nvidia-smi > /dev/null; then
  GPU_MODES=$(nvidia-smi --query-gpu=name,compute_mode --format=csv,noheader 2> /dev/null || true)
  echo "GPU: ${GPU_MODES:-sin informacion de nvidia-smi}"
  if [[ "$GPU_MODES" == *Exclusive* || "$GPU_MODES" == *Prohibited* ]]; then
    echo "la GPU no deja que varios procesos la compartan (${GPU_MODES})" >&2
    exit 1
  fi
fi

pids=()
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  out="runs/diag/${name}/k${K}/fedavg_seed${SEED}.csv"
  log="${LOG_PREFIX}-${name}.out"
  echo "${name} -> ${out} (progreso en ${log})"
  # FLAGS sin comillas a proposito: cada variante son cero o mas opciones separadas por espacios.
  # shellcheck disable=SC2086
  uv run --no-sync python main.py --aggregation fedavg --seed "$SEED" --episodes "$EPISODES" \
    --num-v2v-pairs "$K" --device auto --output "$out" ${FLAGS[$i]} > "$log" 2>&1 &
  pids+=("$!")
done

failed=()
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed+=("${NAMES[$i]}")
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "fallaron las variantes ${failed[*]} (ver sus .out); las demas terminaron" >&2
  exit 1
fi
echo "terminaron todas las variantes: ${NAMES[*]}"
