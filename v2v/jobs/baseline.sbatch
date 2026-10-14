#!/bin/bash
# Linea base (tarea 1.6): un metodo y una semilla por trabajo.
#   sbatch [opciones de particion] jobs/baseline.sbatch <fedavg|dte|greedy> <semilla> [episodios] [K]
# Particion, GPU y tiempo se pasan al enviar; `make baseline-k5` (raiz del repo) lanza los 9 trabajos.
# Salida: runs/k<K>/<metodo>_seed<semilla>.csv
#SBATCH --job-name=v2v-base
#SBATCH --mem=32gb
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=END,FAIL

set -euo pipefail

METHOD=${1:?falta el metodo: fedavg, dte o greedy}
SEED=${2:?falta la semilla}
EPISODES=${3:-4000}
K=${4:-5}
OUT="runs/k${K}/${METHOD}_seed${SEED}.csv"

echo "metodo=${METHOD} semilla=${SEED} episodios=${EPISODES} K=${K} -> ${OUT}"

case "$METHOD" in
  fedavg)
    uv run python main.py --aggregation fedavg --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ;;
  dte)
    uv run python main.py --aggregation none --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --device auto --output "$OUT" ;;
  greedy)
    uv run python train_greedy.py --seed "$SEED" --episodes "$EPISODES" \
      --num-v2v-pairs "$K" --output "$OUT" ;;
  *)
    echo "metodo desconocido: ${METHOD} (usa fedavg, dte o greedy)" >&2
    exit 1 ;;
esac
