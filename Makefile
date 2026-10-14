.PHONY: cpu gpu sq run-greedy pull-runs paper paper-watch paper-clean

cpu:
	srun --mem=64gb --time=12:00:00 -p medium --pty bash -i

gpu:
	srun --mem=64gb --time=12:00:00 --gres=gpu:1 -p gpu --pty bash -i

sq:
	squeue -u $(USER)

# Barrido de K: K pares V2V x semillas. Cada trabajo deja v2v/runs/k<K>/<metodo>_seed<semilla>.csv
# y su .json (configuracion, maquina y tiempo). Se puede achicar sin editar nada, por ejemplo:
#   make run-greedy KS="5 10" SEEDS="0" EPISODES=500
KS := 5 10 15 20 25 30
SEEDS := 0 1 2
EPISODES := 4000
GREEDY_TIME := 02:00:00

# Greedy en Hypatia: un trabajo de CPU por K y semilla (18 con los valores por defecto).
# Correr desde la raiz del repo; `make sq` muestra la cola.
run-greedy:
	cd v2v && for k in $(KS); do for seed in $(SEEDS); do 		sbatch -p medium --time=$(GREEDY_TIME) -J v2v-greedy-k$$k-s$$seed 			jobs/baseline.sbatch greedy $$seed $(EPISODES) $$k; 	done; done

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
