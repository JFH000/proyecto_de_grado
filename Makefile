.PHONY: cpu gpu sq run-greedy run-fedavg run-dte run-gossip pull-runs paper paper-watch paper-clean

cpu:
	srun --mem=64gb --time=12:00:00 -p medium --pty bash -i

gpu:
	srun --mem=64gb --time=12:00:00 --gres=gpu:1 -p gpu --pty bash -i

sq:
	squeue -u $(USER)

# Barrido de K: K pares V2V x semillas, un trabajo por K y semilla. Cada uno deja
# v2v/runs/k<K>/<metodo>_seed<semilla>.csv y su .json (configuracion, maquina y tiempo).
# Antes de enviar, sincroniza el entorno una sola vez (los trabajos usan `uv run --no-sync` y no lo
# tocan: si cada trabajo lo sincroniza, los que arrancan juntos se pisan y fallan). Salta las
# corridas cuyo CSV ya existe, asi que volver a correr el target relanza solo las que faltan
# (esperar a que `make sq` este vacio, o se envian dos veces las que siguen corriendo).
# Todo se puede cambiar sin editar nada, por ejemplo:
#   make run-fedavg KS="5 10" SEEDS="0" EPISODES=500
#   make run-fedavg SBATCH=echo     # muestra los sbatch sin enviarlos
KS := 5 10 15 20 25 30
SEEDS := 0 1 2
EPISODES := 4000
SBATCH := sbatch
GREEDY_TIME := 02:00:00
# Limite de tiempo de cada trabajo de FedAvg, DTE y gossip: K x LEARNER_MINUTES_PER_K minutos (con 90:
# K = 5 -> 7.5 h, K = 30 -> 45 h). Estimado para FedAvg en la GPU de Hypatia con 4000 episodios: ~2.7 h
# con K = 5 y ~22 h con K = 30; DTE y gossip entrenan las mismas redes, asi que usan el mismo limite.
# El CSV se escribe solo al final: un trabajo que se pasa del limite se pierde entero.
LEARNER_MINUTES_PER_K := 90

# $(call submit,<metodo>,<opciones de sbatch>), dentro de un for con $$k y $$seed.
define submit
cd v2v && uv sync --frozen && for k in $(KS); do for seed in $(SEEDS); do 	out=runs/k$$k/$(1)_seed$$seed.csv; 	if [ -f $$out ]; then echo "ya existe $$out"; continue; fi; 	$(SBATCH) $(2) -J v2v-$(1)-k$$k-s$$seed jobs/baseline.sbatch $(1) $$seed $(EPISODES) $$k; done; done
endef

# Greedy: CPU (particion medium), 18 trabajos con los valores por defecto. Correr en Hypatia
# desde la raiz del repo; `make sq` muestra la cola.
run-greedy:
	$(call submit,greedy,-p medium --time=$(GREEDY_TIME))

# FedAvg, DTE y gossip: una GPU por trabajo, 18 trabajos cada uno con los valores por defecto.
# Correr en Hypatia.
run-fedavg:
	$(call submit,fedavg,-p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K))))

run-dte:
	$(call submit,dte,-p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K))))

run-gossip:
	$(call submit,gossip,-p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K))))

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
