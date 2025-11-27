import torch
from test import test_accuracy, test_accuracy_population
from feature import feature_extraction_spike
from train import training, training_population
import numpy as np
from utils.Utils import Utils
# To enable downloading some datasets from pytorch
import ssl
import argparse
import sys
from config import ExperimentConfig
from datetime import datetime
from metrics.Metrics import statistics, statistics_test_population, statistics_test_1, statistics_exp_1

def get_parser():
    parser = argparse.ArgumentParser(description="SpikingResNet Training Configuration")

    # General parameters
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help="Device to run training on")
    parser.add_argument('--mode', type=str, default='test', help="train or test mode")

    # Dataset parameters
    parser.add_argument('--dataset_ID', type=str, default='CIFAR10', help="Dataset to use")
    parser.add_argument('--case', type=str, default='06', help="Case identifier for the experiment")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")

        # Model parameters
    parser.add_argument('--model', type=int, default=18, help="Model architecture")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of output classes")
    parser.add_argument('--time_steps_train', type=int, default=50, help="Number of time steps for spiking models")
    parser.add_argument('--time_steps_extract', type=int, default=50, help="Number of time steps for feature extraction")
    parser.add_argument('--expansion', type=int, default=1, help="Expansion factor for the model")
    parser.add_argument('--loss', type=str, default="mse_count_loss", help="Loss function")
    parser.add_argument('--auto_aug', type=bool, default=False, help="Use auto augmentation")
    parser.add_argument('--population_coding', type=bool, default=False, help="Use population coding")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and validation")
    parser.add_argument('--pretrained', type=bool, default=True, help="Start from checkpoint if True")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw'], default='adam', help="Optimizer to use")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 regularization)")

    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'none'], default='cosine', help="Learning rate scheduler")
    parser.add_argument('--step_size', type=int, default=30, help="Step size for step scheduler")
 
    return parser

def get_provided_args():
    """Get list of arguments that were explicitly provided via command line"""
    provided = set()
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--'):
            arg_name = arg[2:]  # Remove '--'
            provided.add(arg_name)
    
    return provided

if __name__ == "__main__":
    # Define defaut arguments and call training

    '''
    dataSet:        dataset like MNIST, CIFAR10
    modelType:      convolutional or spiking neural network, 'conv' or 'spike' 
    batchSize:      batch size

    ResNetModel 1:  Convolutional neural network based on Conv2D, and LIFs
    '''


    ssl._create_default_https_context = ssl._create_unverified_context

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Get which arguments were explicitly provided
    provided_args = get_provided_args()
    
    # Create config with experiment defaults
    config = ExperimentConfig()
    
    # Override ONLY explicitly provided arguments (not parser defaults)
    config.update_from_args(args, provided_args)

    # Convert to dictionary for function calls
    # config_dict = config.to_dict()
    
    # Print current configuration
    print(f"Running experiment with:")
    print(f"  Dataset: {config.dataset_ID}")
    print(f"  Model: ResNet{config.resnet_model}")
    print(f"  Case: {config.case}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Expansion: {config.expansion}")
    print(f"  Loss: {config.loss}")
    print(f"  Time steps Train: {config.num_time_steps_train}")
    print(f"  Epochs: {config.epochs}")


    # # Get image size
    # channels, rows, cols = Utils.get_image_size(dataset_train, dataSet_ID)
    # print(f"Image size: {channels, rows, cols}")

    # Create features case
    Utils.make_features_dir(config.model_type, config.case)

    if config.mode == 'test':

        # Training/testing
        # set pretrained=True if continious training is needed

        if config.loss != 'mse_count_loss':
            training(config)
            acc_spk, acc_mem = test_accuracy(config)
        elif config.loss == 'mse_count_loss':
            training_population(config)
            acc_spk, acc_mem = test_accuracy_population(config)
        else:
            print("Something's wrong with the expansion parameter. It should be 1 or greater than 1.")

        # Feature extraction
        feature_extraction_spike(config)

        # Statistics
        statistics(config, acc_spk, acc_mem)

        # Vizualizer
        # Utils.visualize_feature(str(config.dataset_ID), str(config.dataset_feat[2]), config.case, feature_type='features')

    elif config.mode == 'test_population':
        # if config.expansion == 1:
        if config.loss != 'mse_count_loss':
            training(config)
            acc_spk, acc_mem = test_accuracy(config)
        # elif config.expansion > 1:
        elif config.loss == 'mse_count_loss':
            training_population(config)
            acc_spk, acc_mem = test_accuracy_population(config)
        # Feature extraction for different steps
        config.override_feature_extraction = True
        config.methods = ['NCM', 'KNN'] # specify only one for test_1
        statistics_test_population(config, acc_spk, acc_mem)

        print("Done testing different extraction time steps.")

    elif config.mode == 'test_1':
        if config.num_time_steps_train == 1:
            if config.loss != 'mse_count_loss':
                training(config)
                acc_spk, acc_mem = test_accuracy(config)
            elif config.loss == 'mse_count_loss':
                training_population(config)
                acc_spk, acc_mem = test_accuracy_population(config)
            else:
                print("Something's wrong with the expansion parameter. It should be 1 or greater than 1.")

            # Feature extraction
            # feature_extraction_spike(config)

            # Statistics
            # statistics(config, acc_spk, acc_mem)
        else:
            print("For this mode, the training time steps should be set to 1.")
        # Feature extraction for different steps
        config.override_feature_extraction = True
        config.methods = ['NCM']
        statistics_test_1(config, acc_spk, acc_mem)

        print("Done testing for 1-step training.")

    elif config.mode == 'multi_train':
        #resnet_models = [4, 10, 18]
        expansions = [1, 5, 10, 25, 50]
        seeds = [42, 1987, 1991, 2020, 2024]

        for resnet_model in [config.resnet_model]:
            config.resnet_model = resnet_model
            print(f"Starting multi-train for ResNet{resnet_model}")

            for expansion in expansions:
                config.expansion = expansion

                for seed in seeds:
                    config.seed = seed

                    if config.loss == 'cross_entropy':
                        if config.expansion == 1:
                            training(config)
                            acc_spk, acc_mem = test_accuracy(config)
                        else:
                            training_population(config)
                            acc_spk, acc_mem = test_accuracy_population(config)
                    else:
                        training_population(config)
                        acc_spk, acc_mem = test_accuracy_population(config)

                    # if config.num_time_steps_train >= 1:
                    #     if config.expansion == 1:
                    #         config.loss = 'count_loss'
                    #         training(config)
                    #         acc_spk, acc_mem = test_accuracy(config)
                    #     else:
                    #         training_population(config)
                    #         acc_spk, acc_mem = test_accuracy_population(config)
                    # else:
                    #     print("Something's wrong with the expansion parameter. It should be 1 or greater than 1.")

    elif config.mode == 'multi_test':
        
        expansions = [1, 5, 10, 25, 50]
        seeds = [42, 1987, 1991, 2020, 2024]
        resnet_models = [4, 10, 18]  # Add ResNet models to iterate over

        # Prepare results file
        results_filename = f"results/multi_test_{config.dataset_ID}_{config.auto_aug}_{config.loss}.txt"
        with open(results_filename, 'w') as f:
            # Write header information
            f.write(f"{'='*60}\n")
            f.write(f"Spiking ResNet Multi-Test Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Experiment Configuration:\n")
            f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
            f.write(f"  Case: {config.case}\n")
            f.write(f"  Batch Size: {config.batch_size}\n")
            f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
            f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
            f.write(f"  Number of epochs: {config.epochs}\n")
            f.write(f"  Fitting method: {config.fit}\n")
            f.write(f"  Loss function: {config.loss}\n")
            f.write(f"  Augmentation: {config.auto_aug}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n")

            for resnet_model in resnet_models:
                config.resnet_model = resnet_model

                # Write ResNet model header
                f.write(f"ResNet Model: {resnet_model}\n")
                f.write(f"{'Expansion':<12}{'Spike Accuracy':<20}{'Membrane Accuracy':<20}\n")
                f.write(f"{'-'*60}\n")

                for expansion in expansions:
                    config.expansion = expansion
                    config.override_feature_extraction = True

                    spike_accuracies = []
                    membrane_accuracies = []
                    missing_weights = False

                    for seed in seeds:
                        config.seed = seed

                        # Perform test accuracy for the current seed
                        if config.expansion == 1:
                            # training(config)
                            acc_spk, acc_mem = test_accuracy(config)
                        elif config.expansion != 1:
                            # training_population(config)
                            acc_spk, acc_mem = test_accuracy_population(config)

                        # If the test functions signal missing weights, mark as not trained
                        if acc_spk == -1 and acc_mem == -1:
                            missing_weights = True
                            # No point in testing other seeds for this expansion if weights are missing
                            break

                        spike_accuracies.append(acc_spk)
                        membrane_accuracies.append(acc_mem)

                    # If any seed indicated missing weights, write "not trained"
                    if missing_weights or len(spike_accuracies) == 0:
                        f.write(f"{expansion:<12}{'not trained':<20}{'not trained':<20}\n")
                    else:
                        # Average accuracies over all seeds
                        avg_spike_accuracy = sum(spike_accuracies) / len(spike_accuracies)
                        avg_membrane_accuracy = sum(membrane_accuracies) / len(membrane_accuracies)

                        # Write numeric results to file
                        f.write(f"{expansion:<12}{avg_spike_accuracy:<20.2f}{avg_membrane_accuracy:<20.2f}\n")

                f.write(f"{'='*60}\n\n")

        print("Done multi-test for different expansions and ResNet models.")

    elif config.mode == 'ex_1':
        # python3 tools/plot_ex1_from_json.py results/ex_1/EX1_CIFAR10_L_mse_count_loss_data.json
        seeds = [42, 1987]
        expansions = [10, 25, 50]
        resnet_models = [4, 10] 
        config.near_ood = ['CIFAR10', 'CIFAR100', 'tImage200']
        config.far_ood = ['CIFAR10', 'MNIST', 'SVHN', 'Textures', 'Places365']

        config.override_feature_extraction = True
        statistics_exp_1(config, seeds, expansions, resnet_models)
        print("Done experiment 1.")

    else:
        print("Mode not recognized. Use 'test' or 'test_population'.")


    # Vizualizer
    # Utils.visualize_feature('CIFAR10', 'Food101', case, feature_type='voltages')
    
    # **Knowledge Distillation**: When `--teacher` is enabled, the script implements teacher-student learning using KL divergence loss. This helps transfer knowledge from a pre-trained teacher network to the student model.
