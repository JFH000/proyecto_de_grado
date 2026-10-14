#!/bin/bash
# Una corrida: un metodo, una semilla y un K por trabajo.
#   sbatch [opciones de particion] jobs/baseline.sbatch <fedavg|dte|gossip|greedy> <semilla> [episodios] [K]
# Particion, GPU y tiempo se pasan al enviar; `make run-<metodo>` (raiz del repo) lanza el barrido
# de K. El entorno tiene que estar sincronizado antes (`uv sync`, que los targets
# del Makefile ya hacen): el trabajo usa `uv run --no-sync` para no tocarlo mientras otros corren.
# Salida: runs/k<K>/<metodo>_seed<semilla>.csv y su .json
#SBATCH --job-name=v2v-base
#SBATCH --mem=32gb
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1  # el progreso ("Episode ...") aparece en el .out mientras corre

METHOD=${1:?falta el metodo: fedavg, dte, gossip o greedy}
SEED=${2:?falta la semilla}
EPISODES=${3:-4000}
K=${4:-5}
OUT="runs/k${K}/${METHOD}_seed${SEED}.csv"

echo "metodo=${METHOD} semilla=${SEED} episodios=${EPISODES} K=${K} -> ${OUT}"

case "$METHOD" in
  fedavg)
    uv run --no-sync python main.py --aggregation fedavg --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ;;
  dte)
    uv run --no-sync python main.py --aggregation none --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ;;
  gossip)
    uv run --no-sync python main.py --aggregation gossip --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ;;
  greedy)
    uv run --no-sync python train_greedy.py --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --output "$OUT" ;;
  *)
    echo "metodo desconocido: ${METHOD} (usa fedavg, dte, gossip o greedy)" >&2
    exit 1 ;;
esac
