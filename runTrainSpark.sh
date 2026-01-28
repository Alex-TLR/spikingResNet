#!/usr/bin/bash
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 11 --mode train --dataset SVHN --batch_size 64 --epochs 200 --pretrained True
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 4 --case 42 --loss mse_count_loss
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 10 --case 42 --loss mse_count_loss
tmux new -s work
enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/config_exp6_train_spike_cross_entropy_no_aug_cifar10_r18.yaml > exp6-1.log 2>&1
tmux detach
