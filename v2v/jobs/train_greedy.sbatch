#!/bin/bash
#SBATCH --job-name=v2v-greedy
#SBATCH --partition=medium
#SBATCH --mem=128gb
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err
#SBATCH --mail-user=jf.hernandezp1@uniandes.edu.co
#SBATCH --mail-type=ALL

uv run python train_greedy.py --episodes 300 --output runs/greedy_rewards.csv
