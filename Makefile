.PHONY: cpu gpu sq paper paper-watch paper-clean

cpu:
	srun --mem=64gb --time=12:00:00 -p medium --pty bash -i

gpu:
	srun --mem=64gb --time=12:00:00 --gres=gpu:1 -p gpu --pty bash -i

sq:
	squeue -u $(USER)

# Tesis (thesis-template/main.tex -> thesis-template/main.pdf), en la maquina local con LaTeX.
paper:
	cd thesis-template && latexmk -pdf -interaction=nonstopmode main.tex

# Recompila cada vez que se guarda un .tex o .bib (Ctrl+C para salir).
paper-watch:
	cd thesis-template && latexmk -pdf -pvc -interaction=nonstopmode main.tex

# Borra los archivos auxiliares (deja main.pdf).
paper-clean:
	cd thesis-template && latexmk -c main.tex
