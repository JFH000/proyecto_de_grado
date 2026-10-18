#!/bin/bash
# Diagnostico del entrenamiento: varias variantes de main.py con un mismo K y una misma semilla, a la
# vez en una GPU. Cada variante es una linea de jobs/diag_variants/<conjunto>.txt:
#   <nombre> [opciones de main.py...]
# Sin opciones es la configuracion por defecto; `--aggregation none` o `gossip` cambian el metodo.
#   sbatch [opciones de particion] jobs/diag.sbatch <conjunto> [K] [episodios] [semilla]   (10 2000 0)
# `make run-diag DIAG_SET=<conjunto>` (raiz del repo) lo envia.
# Salida: runs/diag/<nombre>/k<K>/<metodo>_seed<semilla>.csv y su .json, fuera de runs/k<K>/ para no
# mezclarse con el barrido. Salta las variantes cuyo CSV ya existe: un mismo nombre es siempre la misma
# configuracion, asi que una variante que aparece en varios conjuntos (como base) corre una sola vez.
# Progreso: jobs/logs/<nombre del trabajo>-<id>-<variante>.out. Analisis: notebooks/diagnostico.ipynb.
#SBATCH --job-name=v2v-diag
#SBATCH --mem=32gb
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1

SET=${1:?falta el conjunto de variantes (un archivo de jobs/diag_variants/, sin el .txt)}
K=${2:-10}
EPISODES=${3:-2000}
SEED=${4:-0}
VARIANTS_FILE="jobs/diag_variants/${SET}.txt"
if [ ! -f "$VARIANTS_FILE" ]; then
  echo "no existe ${VARIANTS_FILE}" >&2
  exit 1
fi

# Variantes del conjunto que todavia no tienen CSV. Las lineas vacias y las que empiezan con # no cuentan.
NAMES=()
FLAGS=()
METHODS=()
while read -r name flags; do
  if [ -z "$name" ] || [[ "$name" == \#* ]]; then continue; fi
  method=fedavg
  if [[ " $flags " == *" --aggregation none "* ]]; then method=dte; fi
  if [[ " $flags " == *" --aggregation gossip "* ]]; then method=gossip; fi
  out="runs/diag/${name}/k${K}/${method}_seed${SEED}.csv"
  if [ -f "$out" ]; then
    echo "ya existe ${out}: no se vuelve a correr ${name}"
    continue
  fi
  NAMES+=("$name")
  FLAGS+=("$flags")
  METHODS+=("$method")
done < "$VARIANTS_FILE"

if [ "${#NAMES[@]}" -eq 0 ]; then
  echo "todas las variantes de ${SET} ya tienen CSV"
  exit 0
fi

THREADS=$(( ${SLURM_CPUS_PER_TASK:-${#NAMES[@]}} / ${#NAMES[@]} ))
if [ "$THREADS" -lt 1 ]; then THREADS=1; fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

LOG_PREFIX="jobs/logs/${SLURM_JOB_NAME:-v2v-diag}-${SLURM_JOB_ID:-local}"
echo "conjunto=${SET} K=${K} episodios=${EPISODES} semilla=${SEED} variantes=${NAMES[*]} hilos_por_variante=${THREADS}"

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
  out="runs/diag/${name}/k${K}/${METHODS[$i]}_seed${SEED}.csv"
  log="${LOG_PREFIX}-${name}.out"
  echo "${name} -> ${out} (progreso en ${log})"
  # FLAGS sin comillas a proposito: cada variante son cero o mas opciones separadas por espacios.
  # Van despues de --aggregation fedavg, asi que una variante puede cambiar el metodo.
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
