#!/usr/bin/bash
#SBATCH --job-name=SNNResNet18
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=v100s
#SBATCH --output=case07-resnet10.log

CONTAINER_PATH=../container/snn.sif
#CIFAR10 case 06
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode test --dataset CIFAR10 --case 06 --batch_size 64

#SVNH case 07
srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 11 --mode test --dataset SVHN --case 07 --batch_size 64

#MNIST case 08
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode test --dataset MNIST --case 08 --batch_size 64

#FMNIST case 09
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode test --dataset FMNIST --case 09 --batch_size 64

#KMNIST case 10
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode test --dataset KMNIST --case 10 --batch_size 64


