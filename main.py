import torch
from test import test_metrics, test_accuracy, test_accuracy_population, test_accuracy_population_2
from feature import feature_extraction_conv, feature_extraction_spike
from train import training, training_population, training_population_2
import numpy as np
from utils.Utils import Utils
# To enable downloading some datasets from pytorch
import ssl
import argparse
import sys
from config import ExperimentConfig
from datetime import datetime

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
    # config.update_from_args(args, provided_args)

    # Convert to dictionary for function calls
    # config_dict = config.to_dict()
    
    # Print current configuration
    print(f"Running experiment with:")
    print(f"  Dataset: {config.dataset_ID}")
    print(f"  Model: ResNet{config.resnet_model}")
    print(f"  Case: {config.case}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Expansion: {config.expansion}")
    print(f"  Epochs: {config.epochs}")


    # # Get image size
    # channels, rows, cols = Utils.get_image_size(dataset_train, dataSet_ID)
    # print(f"Image size: {channels, rows, cols}")

    # Create features case
    Utils.make_features_dir(config.model_type, config.case)

    # Training/testing
    # set pretrained=True if continious training is needed

    if config.expansion == 1:
        training(config)
        acc_spk, acc_mem = test_accuracy(config)
    elif config.expansion > 1:
        training_population_2(config)
        acc_spk, acc_mem = test_accuracy_population_2(config)
        # pass
    else:
        print("Something's wrong with the expansion parameter. It should be 1 or greater than 1.")

    # Feature extraction
    feature_extraction_spike(config)

    # Statistics
    feature_types = ['features', 'spikes', 'probs', 'voltages']
    results_filename = f'results/OoD_case_{config.case}_{config.dataset_ID}_ResNet{config.resnet_model}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_E_{config.expansion}_A_{config.auto_aug}.txt'
    ood_datasets = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]

    # Open file for writing
    with open(results_filename, 'w') as f:
        # Write header information
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Out-of-Distribution Detection Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Out-of-Distribution Datasets: {', '.join(ood_datasets)}\n")
        f.write(f"  Model: ResNet{config.resnet_model}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        if config.expansion == 1:
            f.write(f"  No population coding.\n")
        elif config.expansion > 1:
            f.write(f"  Population coding with {config.expansion} expansions.\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Accuracy on spikes: {acc_spk:05.2f}\n")
        f.write(f"Accuracy on membrane: {acc_mem:05.2f}\n")
        f.write(f"{'='*60}\n\n")
        
        all_results = {}
        methods = ['NCM', 'MD', 'KNN', 'FKM', 'CKM']
        
        # Process each feature type
        for feature_type in feature_types:
            print(f"Processing statistics for feature type: {feature_type}")
            f.write(f"Feature Type: {feature_type.upper()}\n")
            f.write(f"{'-'*40}\n")
            
            try:
                stats = test_metrics(case=config.case, nameID=config.dataset_ID, methods=methods, features=feature_type)
                
                if stats is not None and len(stats) > 0:
                    formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
                    all_results[feature_type] = formatted_stats
                    
                    # Build the exact format you want
                    # Each method gets 5 characters, separated by ' | ' (3 chars)
                    # AUROC section: 5 methods = 5 chars + 4 separators = 37 chars total
                    # AUPR section: same = 37 chars
                    # FPR95 section: same = 37 chars
                    span = len(methods) * 6 + (len(methods) - 1) * 3
                    
                    # Top header with metric spans
                    auroc_span = "AUROC".center(span)
                    aupr_span = "AUPR".center(span)
                    fpr95_span = "FPR95".center(span)

                    top_header = f"  {'Dataset':10s}: {auroc_span} | {aupr_span} | {fpr95_span}"
                    
                    # Method header - repeat methods 3 times
                    methods_line = ' | '.join([f'{method:>6s}' for method in methods])
                    method_header = f"  {'':10s}: {methods_line} | {methods_line} | {methods_line}"

                    # Separator line matching the total width
                    total_width = len(method_header)
                    separator = '-' * total_width
                    
                    # Write formatted table
                    f.write(f"{top_header}\n")
                    f.write(f"{method_header}\n")
                    f.write(f"{separator}\n")
                    
                    # Data rows
                    for i, row in enumerate(formatted_stats):
                        if i < len(ood_datasets):
                            dataset_name = ood_datasets[i]
                            formatted_row = ' | '.join([f'{cell:>6s}' for cell in row])
                            f.write(f"  {dataset_name:<10s}: {formatted_row}\n")

                    # Console output
                    # print(f"  Results for {feature_type}:")
                    # print(f"{top_header}")
                    # print(f"{method_header}")  
                    # print(f"{separator}")
                    # for i, row in enumerate(formatted_stats):
                    #     if i < len(ood_datasets):
                    #         dataset_name = ood_datasets[i]
                    #         formatted_row = ' | '.join([f'{cell:>6s}' for cell in row])
                    #         print(f"  {dataset_name:<10s}: {formatted_row}")
                else:
                    f.write("  No data available\n")
                    print(f"  No data available for {feature_type}")
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                f.write(f"  {error_msg}\n")
                print(f"  {error_msg}")
            finally:
                f.write("\n")
                print()

    print(f"\nDetailed statistics saved to: {results_filename}")

    # stats = test_metrics(case=config.case, nameID=config.dataset_ID, features='features')
    # formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
    # for row in formatted_stats:
    #     print(' '.join(row))

    # Vizualizer
    # Utils.visualize_feature('CIFAR10', 'Food101', case, feature_type='voltages')

    # Ponovljivost
    # _seed_ = 2020
    # import random
    # import math
    # random.seed(2020)
    # torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    # torch.cuda.manual_seed_all(_seed_)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # check out
    
    # **Knowledge Distillation**: When `--teacher` is enabled, the script implements teacher-student learning using KL divergence loss. This helps transfer knowledge from a pre-trained teacher network to the student model.
