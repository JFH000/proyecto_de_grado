.PHONY: cpu gpu sq pull-runs

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