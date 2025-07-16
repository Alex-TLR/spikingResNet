#!/usr/bin/bash
#SBATCH --job-name=SNNResNet18
#SBATCH --time=95:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=v100s

module load CUDA
#CONTAINER_PATH=../container/snn.sif
CONTAINER_PATH=/d/hpc/home/dsluga/SNN/container/snn.sb
#DESTINATION=/d/hpc/home/dsluga/SNN/spikingResNet/weights/spike
#dir=$WORKDIR
#cp *.py $dir/
#cp -R models $dir/models
#cp -R utils $dir/utils
#cp -R metrics $dir/metrics
#cp -R clustering $dir/clustering
#cd $dir
echo "Running in $(pwd)"
#mkdir weights
#mkdir weights/spike
#CIFAR10
srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode train --dataset CIFAR10 --batch_size 64 --epochs 350 --pretrained True

#MNIST
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode train --dataset MNIST --batch_size 64 --epochs 5 --pretrained False

#FMNIST
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode train --dataset FMNIST --batch_size 64 --epochs 40 --pretrained False

#KMNIST
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode train --dataset KMNIST --batch_size 64 --epochs 40 --pretrained False

#SVNH
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --mode train --dataset SVHN --batch_size 64 --epochs 200 --pretrained True

#cp ./weights/spike/* $DESTINATION