.PHONY: cpu gpu sq pull-runs baseline-k5 paper paper-watch paper-clean

cpu:
	srun --mem=64gb --time=12:00:00 -p medium --pty bash -i

gpu:
	srun --mem=64gb --time=12:00:00 --gres=gpu:1 -p gpu --pty bash -i

sq:
	squeue -u $(USER)

HYPATIA_HOST := hypatia
HYPATIA_RUNS_DIR := ~/proyecto_de_grado/v2v/runs

pull-runs:
	mkdir -p v2v/runs
	scp -r $(HYPATIA_HOST):$(HYPATIA_RUNS_DIR)/* v2v/runs/

# Linea base K=5 (tarea 1.6): FedAvg, DTE y Greedy x 3 semillas = 9 trabajos -> v2v/runs/k5/.
# Correr en Hypatia desde la raiz del repo. Ajusta LEARNER_TIME a lo que tarde una corrida de
# BASELINE_EPISODES episodios (p. ej. lo que tardo hfrl_4000); se puede sobreescribir:
#   make baseline-k5 LEARNER_TIME=12:00:00 BASELINE_EPISODES=2000
SEEDS := 0 1 2
BASELINE_EPISODES := 4000
LEARNER_TIME := 24:00:00
GREEDY_TIME := 02:00:00
LEARNER_SBATCH = -p gpu --gres=gpu:1 --time=$(LEARNER_TIME)
GREEDY_SBATCH = -p medium --time=$(GREEDY_TIME)

baseline-k5:
	cd v2v && for seed in $(SEEDS); do \
		sbatch $(LEARNER_SBATCH) -J v2v-fedavg-s$$seed jobs/baseline.sbatch fedavg $$seed $(BASELINE_EPISODES) 5; \
		sbatch $(LEARNER_SBATCH) -J v2v-dte-s$$seed jobs/baseline.sbatch dte $$seed $(BASELINE_EPISODES) 5; \
		sbatch $(GREEDY_SBATCH) -J v2v-greedy-s$$seed jobs/baseline.sbatch greedy $$seed $(BASELINE_EPISODES) 5; \
	done

# Tesis (thesis-template/main.tex -> thesis-template/main.pdf), en la maquina local con LaTeX.
paper:
	cd thesis-template && latexmk -pdf -interaction=nonstopmode main.tex

# Recompila cada vez que se guarda un .tex o .bib (Ctrl+C para salir).
paper-watch:
	cd thesis-template && latexmk -pdf -pvc -interaction=nonstopmode main.tex

# Borra los archivos auxiliares (deja main.pdf).
paper-clean:
	cd thesis-template && latexmk -c main.tex
