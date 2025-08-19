#!/usr/bin/bash
#SBATCH --job-name=SNNResNet18
#SBATCH --time=92:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --exclude=wn202
#SBATCH --output=SNNResnet10-T50-E400-%j.log

CONTAINER_PATH=../container/snn.sif
srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --case 03 --model 10 --mode train --dataset CIFAR10 --batch_size 128 --epochs 400 --time_steps 50 --auto_aug True
