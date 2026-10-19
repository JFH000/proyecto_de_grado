.PHONY: cpu gpu sq run-greedy run-umbral run-fedavg run-dte run-gossip run-diag run-cpu-test pull-runs paper paper-watch paper-clean

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
# - SWEEP=<nombre> es un barrido con nombre: deja v2v/runs/<nombre>/k<K>/ y corre FedAvg, DTE y gossip
#   con las opciones de main.py de v2v/jobs/sweeps/<nombre>.txt (Greedy y la regla de umbral no
#   aprenden: solo cambia su carpeta). Sin SWEEP, v2v/runs/k<K>/ y las opciones por defecto.
# Todo se puede cambiar sin editar nada, por ejemplo:
#   make run-fedavg KS="5 10" SEEDS="0" EPISODES=500
#   make run-fedavg SBATCH=echo     # muestra los sbatch sin enviarlos
#   make run-gossip SWEEP=propia RUN_ON=cpu
KS := 5 10 15 20 25 30
SEEDS := 0 1 2
# 2000 y no 4000 (los barridos de runs/k<K>/): en esos, la recompensa de FedAvg llegaba a su mejor
# punto antes del episodio 2000, mientras epsilon bajaba, y despues solo caia. Los diagnosticos, con
# los que se eligio la configuracion, tambien son de 2000 episodios.
EPISODES := 2000
SWEEP :=
RUNS_DIR = runs$(if $(SWEEP),/$(SWEEP))
JOB_TAG = $(if $(SWEEP),-$(SWEEP))
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

# Donde corren FedAvg, DTE, gossip y el diagnostico: gpu (por defecto) o cpu, la particion short.
# La red es tan pequena que la GPU casi no acelera: en run-cpu-test (oct. 2026) una corrida sola con
# 2 CPUs tardo 4.2 s por episodio con K = 10 y 13.7 s con K = 30, contra ~8 y ~25 s con 3 corridas
# compartiendo una GPU. Y short deja 128 CPUs por usuario (un barrido entero a la vez) contra 2 GPUs.
# Ejemplo: make run-fedavg RUN_ON=cpu
RUN_ON := gpu
# Limite en CPU: K x CPU_MINUTES_PER_K minutos. A 13.7 s por episodio con K = 30, 4000 episodios son
# ~15 h (~31 min por par); 60 deja el doble de margen (K = 30 -> 30 h; short permite hasta 5 dias).
CPU_MINUTES_PER_K := 60
# Memoria en CPU por semilla. short no da mas de 4 GB por CPU: con los 32 GB de jobs/seeds.sbatch,
# pensados para la GPU, Slurm subiria los CPUs del trabajo para cubrirlos. Una corrida en CPU usa
# menos de 1 GB (run-cpu-test, K = 30).
CPU_MEM_GB_PER_SEED := 4
ifeq ($(RUN_ON),cpu)
LEARNER_SBATCH = -p short --mem=$$((n * $(CPU_MEM_GB_PER_SEED)))G --time=$$((k * $(CPU_MINUTES_PER_K)))
DIAG_SBATCH = -p short
else ifeq ($(RUN_ON),gpu)
LEARNER_SBATCH = -p gpu --gres=gpu:1 --time=$$((k * $(LEARNER_MINUTES_PER_K)))
DIAG_SBATCH = -p gpu --gres=gpu:1
else
$(error RUN_ON tiene que ser gpu o cpu, no "$(RUN_ON)")
endif

# $(call submit,<metodo>,<opciones de sbatch>): por cada K, junta las semillas que faltan y las envia
# en un solo trabajo con CPUS_PER_SEED CPUs por semilla. $$k esta disponible en las opciones.
# Los nombres de los trabajos son v2v-<metodo>[-<barrido>]-k<K>-s<semillas separadas por puntos>, por
# ejemplo v2v-fedavg-k10-s0.1.2 o v2v-fedavg-propia-k10-s0.1.2; los de una sola semilla
# (v2v-fedavg-k5-s2) siguen el mismo formato.
define submit
cd v2v && uv sync --frozen && \
$(if $(SWEEP),{ [ -f jobs/sweeps/$(SWEEP).txt ] || { echo "no existe v2v/jobs/sweeps/$(SWEEP).txt" >&2; exit 1; }; } &&) \
queued="$$($(SQUEUE) 2>/dev/null || true)" && \
for k in $(KS); do \
	seeds=""; \
	for seed in $(SEEDS); do \
		out=$(RUNS_DIR)/k$$k/$(1)_seed$$seed.csv; \
		if [ -f $$out ]; then echo "ya existe $$out"; continue; fi; \
		if printf '%s\n' "$$queued" | grep -Eq "^v2v-$(1)$(JOB_TAG)-k$$k-s([0-9]+[.])*$$seed([.][0-9]+)*$$"; then \
			echo "ya esta en la cola: $(1)$(JOB_TAG) K=$$k semilla $$seed"; continue; \
		fi; \
		seeds="$$seeds $$seed"; \
	done; \
	if [ -z "$$seeds" ]; then continue; fi; \
	n=$$(echo $$seeds | wc -w); \
	$(SBATCH) $(2) --cpus-per-task=$$((n * $(CPUS_PER_SEED))) --export=ALL,V2V_SWEEP=$(SWEEP) \
		-J v2v-$(1)$(JOB_TAG)-k$$k-s$$(echo $$seeds | tr ' ' '.') jobs/seeds.sbatch $(1) $$k $(EPISODES) $$seeds; \
done
endef

# Greedy: CPU (particion medium), un trabajo por K. Correr en Hypatia desde la raiz del repo;
# `make sq` muestra la cola.
run-greedy:
	$(call submit,greedy,-p medium --time=$(GREEDY_TIME))

# Regla de umbral (agents/threshold_policy.py, no esta en el paper): referencia de lo que se logra
# solo con informacion local. CPU, igual que Greedy.
run-umbral:
	$(call submit,umbral,-p medium --time=$(GREEDY_TIME))

# FedAvg, DTE y gossip: un trabajo por K (6 con los valores por defecto). Correr en Hypatia. Con
# RUN_ON=gpu, una GPU por trabajo (la QoS deja 2 a la vez por usuario); con RUN_ON=cpu, en short.
run-fedavg:
	$(call submit,fedavg,$(LEARNER_SBATCH))

run-dte:
	$(call submit,dte,$(LEARNER_SBATCH))

run-gossip:
	$(call submit,gossip,$(LEARNER_SBATCH))

# Diagnostico del entrenamiento: las variantes de v2v/jobs/diag_variants/$(DIAG_SET).txt (una por
# linea: nombre y opciones de main.py) con un K y una semilla, a la vez en una GPU (jobs/diag.sbatch).
# Salida en v2v/runs/diag/<variante>/k<K>/, fuera del barrido; las variantes que ya tienen CSV no se
# vuelven a correr. Se analiza con notebooks/diagnostico.ipynb. Con RUN_ON=cpu va a la particion short.
# Ejemplo:
#   make run-diag DIAG_SET=cola
# El limite de tiempo deja margen: 4 variantes juntas tardaron 5.0-5.6 h con K = 10 y 2000 episodios.
# No mas: con la particion llena, Slurm solo mete un trabajo en los huecos (backfill) si su limite cabe.
DIAG_SET := aprendizaje
DIAG_K := 10
DIAG_EPISODES := 2000
DIAG_SEED := 0
DIAG_TIME := 08:00:00

run-diag:
	cd v2v && uv sync --frozen && $(SBATCH) $(DIAG_SBATCH) --cpus-per-task=8 --time=$(DIAG_TIME) \
		-J v2v-diag-$(DIAG_SET)-k$(DIAG_K) jobs/diag.sbatch $(DIAG_SET) $(DIAG_K) $(DIAG_EPISODES) $(DIAG_SEED)

# Prueba de velocidad en CPU: una corrida corta de FedAvg por cada K de CPU_TEST_KS en la particion
# short (120 CPUs por usuario, contra 2 GPUs en gpu), con CPUS_PER_SEED CPUs, los mismos que tendria
# cada corrida de un barrido en CPU. La red es tan pequena que la GPU casi no acelera: con 3-4 corridas
# compartiendo una GPU, cada episodio tarda ~8-9 s con K = 10 y ~25 s con K = 30. Deja
# v2v/runs/cputest/k<K>/fedavg_seed0.csv, fuera del barrido; wall_time_s / episodios, en su .json, es
# lo que se compara. Correr en Hypatia desde la raiz del repo. Resultado (oct. 2026): 4.2 s por episodio
# con K = 10 y 13.7 s con K = 30, la mitad que en GPU; de ahi RUN_ON=cpu.
CPU_TEST_KS := 10 30
CPU_TEST_EPISODES := 20

run-cpu-test:
	cd v2v && uv sync --frozen && for k in $(CPU_TEST_KS); do \
		$(SBATCH) -p short --cpus-per-task=$(CPUS_PER_SEED) --mem=8gb --time=02:00:00 -J v2v-cputest-k$$k \
			--output=jobs/logs/%x-%j.out \
			--wrap "export OMP_NUM_THREADS=$(CPUS_PER_SEED) MKL_NUM_THREADS=$(CPUS_PER_SEED) OPENBLAS_NUM_THREADS=$(CPUS_PER_SEED) PYTHONUNBUFFERED=1; \
				uv run --no-sync python main.py --episodes $(CPU_TEST_EPISODES) --num-v2v-pairs $$k --device cpu \
				--log-every 5 --output runs/cputest/k$$k/fedavg_seed0.csv"; \
	done

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
