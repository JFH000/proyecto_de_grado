#!/bin/bash
#SBATCH --job-name=v2v
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=ALL

uv run python main.py --device cuda --episodes 300 --output runs/rewards.csv
