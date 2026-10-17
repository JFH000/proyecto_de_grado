.PHONY: cpu gpu sq run-greedy run-fedavg run-dte run-gossip run-diag pull-runs paper paper-watch paper-clean

cpu:
	srun --mem=64gb --time=12:00:00 -p medium --pty bash -i

gpu:
	srun --mem=64gb --time=12:00:00 --gres=gpu:1 -p gpu --pty bash -i

sq:
	squeue -u $(USER)

# Barrido de K: K pares V2V x semillas. Un trabajo por K que corre todas sus semillas a la vez,
# compartiendo la GPU (jobs/seeds.sbatch); cada semilla deja v2v/runs/k<K>/<metodo>_seed<semilla>.csv
# y su .json (configuracion, maquina y tiempo), igual que si corriera sola.
# - Antes de enviar, sincroniza el entorno una sola vez: los trabajos usan `uv run --no-sync` y no
#   lo tocan (si cada uno lo sincroniza, los que arrancan juntos se pisan y fallan).
# - Salta las semillas cuyo CSV ya existe y las que ya estan en la cola (corriendo o esperando),
#   asi que volver a correr el target solo envia las que faltan o fallaron.
# Todo se puede cambiar sin editar nada, por ejemplo:
#   make run-fedavg KS="5 10" SEEDS="0" EPISODES=500
#   make run-fedavg SBATCH=echo     # muestra los sbatch sin enviarlos
KS := 5 10 15 20 25 30
SEEDS := 0 1 2
EPISODES := 4000
SBATCH := sbatch
SQUEUE := squeue -u $(USER) -h -o %j
CPUS_PER_SEED := 2
GREEDY_TIME := 02:00:00
# Limite de tiempo de cada trabajo de FedAvg, DTE y gossip: K x LEARNER_MINUTES_PER_K minutos (con 150:
# K = 5 -> 12.5 h, K = 30 -> 75 h). Una corrida sola de FedAvg tarda en la GPU de Hypatia ~2.7 h con
# K = 5 (medido) y ~22 h con K = 30 (estimado); con 3 semillas compartiendo la GPU cada una va algo
# mas lenta, y el limite deja margen para eso. DTE y gossip entrenan las mismas redes.
# El CSV se escribe solo al final: un trabajo que se pasa del limite se pierde entero.
LEARNER_MINUTES_PER_K := 150

# $(call submit,<metodo>,<opciones de sbatch>): por cada K, junta las semillas que faltan y las envia
# en un solo trabajo con CPUS_PER_SEED CPUs por semilla. $$k esta disponible en las opciones.
# Los nombres de los trabajos son v2v-<metodo>-k<K>-s<semillas separadas por puntos>, por ejemplo
# v2v-fedavg-k10-s0.1.2; los trabajos de una sola semilla (v2v-fedavg-k5-s2) siguen el mismo formato.
define submit
cd v2v && uv sync --frozen && queued="$$($(SQUEUE) 2>/dev/null || true)" && \
for k in $(KS); do \
	seeds=""; \
	for seed in $(SEEDS); do \
		out=runs/k$$k/$(1)_seed$$seed.csv; \
		if [ -f $$out ]; then echo "ya existe $$out"; continue; fi; \
		if printf '%s\n' "$$queued" | grep -Eq "^v2v-$(1)-k$$k-s([0-9]+[.])*$$seed([.][0-9]+)*$$"; then \
			echo "ya esta en la cola: $(1) K=$$k semilla $$seed"; continue; \
		fi; \
		seeds="$$seeds $$seed"; \
	done; \
	if [ -z "$$seeds" ]; then continue; fi; \
	n=$$(echo $$seeds | wc -w); \
	$(SBATCH) $(2) --cpus-per-task=$$((n * $(CPUS_PER_SEED))) -J v2v-$(1)-k$$k-s$$(echo $$seeds | tr ' ' '.') \
		jobs/seeds.sbatch $(1) $$k $(EPISODES) $$seeds; \
done
endef

# Greedy: CPU (particion medium), un trabajo por K. Correr en Hypatia desde la raiz del repo;
# `make sq` muestra la cola.
run-greedy:
	$(call submit,greedy,-p medium --time=$(GREEDY_TIME))

# FedAvg, DTE y gossip: una GPU por trabajo, un trabajo por K (6 con los valores por defecto).
# Correr en Hypatia. La QoS de la particion gpu deja usar 2 GPUs a la vez por usuario.
run-fedavg:
	$(call submit,fedavg,-p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K))))

run-dte:
	$(call submit,dte,-p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K))))

run-gossip:
	$(call submit,gossip,-p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K))))

# Diagnostico de la caida de FedAvg al terminar la exploracion: 4 variantes del entrenamiento
# (base, buffer de 100 000, recompensa x 0.01 y las dos) con un K y una semilla, a la vez en una
# GPU (jobs/diag.sbatch). Salida en v2v/runs/diag/<variante>/k<K>/, fuera del barrido; se analiza
# con notebooks/diagnostico.ipynb. El limite de tiempo deja margen: la variante con buffer de
# 100 000 muestrea mas lento.
DIAG_K := 10
DIAG_EPISODES := 2000
DIAG_SEED := 0
DIAG_TIME := 20:00:00

run-diag:
	cd v2v && uv sync --frozen && $(SBATCH) -p gpu --gres=gpu:1 --cpus-per-task=8 --time=$(DIAG_TIME) \
		-J v2v-diag-k$(DIAG_K) jobs/diag.sbatch $(DIAG_K) $(DIAG_EPISODES) $(DIAG_SEED)

# Trae v2v/runs/ de Hypatia a esta maquina (correr en la maquina local).
HYPATIA_HOST := hypatia
HYPATIA_RUNS_DIR := ~/proyecto_de_grado/v2v/runs

pull-runs:
	mkdir -p v2v/runs
	scp -r $(HYPATIA_HOST):$(HYPATIA_RUNS_DIR)/* v2v/runs/

# Tesis (thesis-template/main.tex -> thesis-template/main.pdf), en la maquina local con LaTeX.
paper:
	cd thesis-template && latexmk -pdf -interaction=nonstopmode main.tex

# Recompila cada vez que se guarda un .tex o .bib (Ctrl+C para salir).
paper-watch:
	cd thesis-template && latexmk -pdf -pvc -interaction=nonstopmode main.tex

# Borra los archivos auxiliares (deja main.pdf).
paper-clean:
	cd thesis-template && latexmk -c main.tex
