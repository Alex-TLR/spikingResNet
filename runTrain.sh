#!/usr/bin/bash
#SBATCH --job-name=SNNResNet-exp1-cifar100-10
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --tasks=1
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --output=exp1-cifar100-10.log
#STEPS=${SLURM_ARRAY_TASK_ID}
#echo "Running training with ${STEPS} time steps"
CONTAINER_PATH=../container/snn.sif
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 11 --mode train --dataset SVHN --batch_size 64 --epochs 200 --pretrained True
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 4 --case 42 --loss cross_entropy
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 10 --case 42 --loss count_loss
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --case 42 --loss count_loss
srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --config experiments/train/config_exp1_train_spike_aug_cifar100_10.yaml
