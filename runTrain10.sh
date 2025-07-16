#!/usr/bin/bash
#SBATCH --job-name=SNNResNet10
#SBATCH --time=95:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=v100s
#SBATCH --output=SNNResnet10-2.log

CONTAINER_PATH=../container/snn.sif
srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 11 --mode train --dataset SVHN --batch_size 64 --epochs 200 --pretrained True
