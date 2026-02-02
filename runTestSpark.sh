#!/usr/bin/bash
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 11 --mode train --dataset SVHN --batch_size 64 --epochs 200 --pretrained True
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 4 --case 42 --loss mse_count_loss
#srun --ntasks=1 --cpus-per-task=8 --partition=gpu -G1 apptainer exec --nv "$CONTAINER_PATH" python main.py --model 10 --case 42 --loss mse_count_loss

#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp2_test_experiment_1.yaml > exp2-test_experiment_1.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp3_test_experiment_1.yaml > exp3-test_experiment_1.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp4_test_experiment_1.yaml > exp4-test_experiment_1.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp5_test_experiment_1.yaml > exp5-test_experiment_1.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp6_test_experiment_1.yaml > exp6-test_experiment_1.log 2>&1

#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp1_test_experiment_2.yaml > exp1-test_experiment_2.log 2>&1
enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp2_test_experiment_2.yaml > exp2-test_experiment_2.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp3_test_experiment_2.yaml > exp3-test_experiment_2.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp4_test_experiment_2.yaml > exp4-test_experiment_2.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp5_test_experiment_2.yaml > exp5-test_experiment_2.log 2>&1
#enroot start --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all --mount .:/workspace snn python main.py --config experiments/test/config_exp6_test_experiment_2.yaml > exp6-test_experiment_2.log 2>&1

