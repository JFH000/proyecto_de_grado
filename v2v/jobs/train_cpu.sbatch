#!/bin/bash
#SBATCH --job-name=v2v
#SBATCH --partition=medium
#SBATCH --mem=128gb
#SBATCH --time=10:00:00
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=ALL

uv run python main.py --device cpu --episodes 300 --output runs/rewards.csv
