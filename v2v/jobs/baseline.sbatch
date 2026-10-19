#!/bin/bash
# Una corrida: un metodo, una semilla y un K por trabajo.
#   sbatch [opciones de particion] jobs/baseline.sbatch <fedavg|dte|gossip|greedy|umbral> <semilla> [episodios] [K]
# Particion, GPU y tiempo se pasan al enviar; `make run-<metodo>` (raiz del repo) lanza el barrido
# de K. El entorno tiene que estar sincronizado antes (`uv sync`, que los targets
# del Makefile ya hacen): el trabajo usa `uv run --no-sync` para no tocarlo mientras otros corren.
# Salida: runs/k<K>/<metodo>_seed<semilla>.csv y su .json
# Con V2V_SWEEP=<nombre> en el entorno (`make run-<metodo> SWEEP=<nombre>`), la salida va a
# runs/<nombre>/k<K>/ y FedAvg, DTE y gossip corren con las opciones de main.py de
# jobs/sweeps/<nombre>.txt. Greedy y la regla de umbral no aprenden: para ellos solo cambia la carpeta.
#SBATCH --job-name=v2v-base
#SBATCH --mem=32gb
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1  # el progreso ("Episode ...") aparece en el .out mientras corre

METHOD=${1:?falta el metodo: fedavg, dte, gossip, greedy o umbral}
SEED=${2:?falta la semilla}
EPISODES=${3:-4000}
K=${4:-5}
SWEEP=${V2V_SWEEP:-}
ROOT=runs
FLAGS=()
if [ -n "$SWEEP" ]; then
  SWEEP_FILE="jobs/sweeps/${SWEEP}.txt"
  if [ ! -f "$SWEEP_FILE" ]; then
    echo "no existe ${SWEEP_FILE}" >&2
    exit 1
  fi
  ROOT="runs/${SWEEP}"
  read -r -a FLAGS <<< "$(grep -v '^[[:space:]]*#' "$SWEEP_FILE" | tr '\n' ' ')"
fi
OUT="${ROOT}/k${K}/${METHOD}_seed${SEED}.csv"

echo "metodo=${METHOD} semilla=${SEED} episodios=${EPISODES} K=${K} barrido=${SWEEP:-(ninguno)} -> ${OUT}"
if [ "${#FLAGS[@]}" -gt 0 ]; then echo "opciones de main.py (FedAvg, DTE y gossip): ${FLAGS[*]}"; fi

case "$METHOD" in
  fedavg)
    uv run --no-sync python main.py --aggregation fedavg --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ${FLAGS[@]+"${FLAGS[@]}"} ;;
  dte)
    uv run --no-sync python main.py --aggregation none --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ${FLAGS[@]+"${FLAGS[@]}"} ;;
  gossip)
    uv run --no-sync python main.py --aggregation gossip --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ${FLAGS[@]+"${FLAGS[@]}"} ;;
  greedy)
    uv run --no-sync python train_greedy.py --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --output "$OUT" ;;
  umbral)
    uv run --no-sync python train_greedy.py --policy umbral --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --output "$OUT" ;;
  *)
    echo "metodo desconocido: ${METHOD} (usa fedavg, dte, gossip, greedy o umbral)" >&2
    exit 1 ;;
esac
