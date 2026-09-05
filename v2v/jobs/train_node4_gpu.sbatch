#!/bin/bash
#SBATCH --job-name=v2v
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/slurm-%x-%j.out
#SBATCH --error=jobs/logs/slurm-%x-%j.err

uv run python train_node4.py --device cuda --episodes 300 --output runs/node4_no_fedavg_rewards.csv
