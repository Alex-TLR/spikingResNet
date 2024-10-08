#!/usr/bin/bash
#SBATCH --job-name=SNNResNet18
#SBATCH --time=95:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100

CONTAINER_PATH=../container/snn.sif
srun --ntasks=1 --cpus-per-task=8 --mem 32GB --partition=gpu -G1 --constraint=h100 apptainer exec --nv "$CONTAINER_PATH" python main.py
