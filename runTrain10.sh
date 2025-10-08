#!/usr/bin/bash
#SBATCH --job-name=SNNResNet18
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=2,4,8,12,16
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --output=SNNResnet18-E1-steps-%a.log
STEPS=${SLURM_ARRAY_TASK_ID}
echo "Running training with ${STEPS} time steps"
CONTAINER_PATH=../container/snn.sif
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 11 --mode train --dataset SVHN --batch_size 64 --epochs 200 --pretrained True
srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 18 --case PCC-21-${STEPS} --time_steps_train ${STEPS} --expansion 1
