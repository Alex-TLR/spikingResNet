#!/usr/bin/bash
#SBATCH --job-name=SNNResNet18
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=v100s

CONTAINER_PATH=../container/snn.sif
srun --ntasks=1 --cpus-per-task=4 --mem 32GB --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py
