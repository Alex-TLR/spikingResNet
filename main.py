import torch
from test import test_accuracy
from feature import feature_extraction_spike
from train import training
from utils.Utils import Utils
import ssl
import argparse
import os
from config import ExperimentConfig, load_config_from_yaml
from datetime import datetime
from metrics.Metrics import statistics, statistics_test_population, statistics_test_1, statistics_exp_1, statistics_exp_2, statistics_exp_3

def get_parser():
    parser = argparse.ArgumentParser(description="SpikingResNet Training Configuration")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML configuration file")
    return parser

def print_experiment_info(config, mode_description=""):
    """Print current experiment configuration."""
    if mode_description:
        print(f"\n{'='*70}")
        print(f"{mode_description}")
        print(f"{'='*70}")
    print(f"  Dataset: {config.dataset_ID}")
    print(f"  Model: ResNet{config.resnet_model} ({config.model_type})")
    print(f"  Case: {config.case}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Expansion: {config.expansion}")
    print(f"  Loss: {config.loss}")
    print(f"  Seed: {config.seed}")
    print(f"  Time steps (train/extract): {config.num_time_steps_train}/{config.num_time_steps_extract}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Auto augmentation: {config.auto_aug}")
    if mode_description:
        print(f"{'='*70}\n")
    else:
        print()

if __name__ == "__main__":
    # Define defaut arguments and call training
    ssl._create_default_https_context = ssl._create_unverified_context

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config from YAML file
    config = load_config_from_yaml(args.config)

    print(f"\n{'='*70}")
    print(f"SpikingResNet Experiment - Mode: {config.mode.upper()}")
    if config.mode == 'test':
        print(f"Test Type: {config.test_type}")
    print(f"{'='*70}")
    print(f"Initial Configuration:")
    print(f"  Dataset: {config.dataset_ID}")
    print(f"  Case: {config.case}")
    print(f"{'='*70}\n")

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    

    if config.mode == 'test':
        # Ensure results directory exists for outputs
        # Create features case
        Utils.make_features_dir(config.model_type, config.case)
        os.makedirs('results', exist_ok=True)
        # Handle different test types
        if config.test_type == 'standard':
            # Standard test: train, test, extract features, compute statistics
            training(config)
            acc_spk, acc_mem = test_accuracy(config)
            
            # Feature extraction
            feature_extraction_spike(config)
            
            # Statistics
            statistics(config, acc_spk, acc_mem)

        elif config.test_type == 'population':
            # Test with different extraction time steps
            if config.loss != 'mse_count_loss':
                training(config)
                acc_spk, acc_mem = test_accuracy(config)
            statistics_test_population(config, acc_spk, acc_mem)
            print("Done testing different extraction time steps.")
        elif config.test_type == 'single_step':
            training(config)
            acc_spk, acc_mem = test_accuracy(config)
            statistics_test_1(config, acc_spk, acc_mem)
            print("Done testing for 1-step training.")
        elif config.test_type == 'accuracy':
            # Multi-test: test accuracy with different seeds/expansions/models
            seeds = config.seeds
            expansions = config.expansions
            resnet_models = config.resnet_models
            # Prepare results file
            results_filename = f"results/multi_test_{config.model_type}_{config.dataset_ID}_{config.auto_aug}_{config.loss}.txt"
            with open(results_filename, 'w') as f:
                # Write header information
                f.write(f"{'='*60}\n")
                f.write(f"Spiking ResNet Multi-Test Results\n")
                f.write(f"{'='*60}\n")
                f.write(f"Experiment Configuration:\n")
                f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
                f.write(f"  Model type: {config.model_type}\n")
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

                        spike_accuracies = []
                        membrane_accuracies = []
                        missing_weights = False

                        for seed in seeds:
                            config.seed = seed

                            # Perform test accuracy for the current seed
                            if config.model_type == 'spike':
                                acc_spk, acc_mem = test_accuracy(config)
                                if acc_spk == -1 and acc_mem == -1:
                                    missing_weights = True
                                    break

                                spike_accuracies.append(acc_spk)
                                membrane_accuracies.append(acc_mem)
                            
                            elif config.model_type == 'conv':
                                print(f"Find accuracy..")
                                acc_mem = test_accuracy(config)
                                print(f"Accuracy found: {acc_mem}")

                                if acc_mem == -1:
                                    missing_weights = True
                                    break

                                membrane_accuracies.append(acc_mem)
                            else:
                                print("Model type not recognized. Use 'spike' or 'conv'.")

                        # If any seed indicated missing weights, write "not trained"
                        if missing_weights or len(membrane_accuracies) == 0:
                            f.write(f"{expansion:<12}{'not trained':<20}{'not trained':<20}\n")
                        else:
                            if config.model_type == 'spike':
                                avg_spike_accuracy = sum(spike_accuracies) / len(spike_accuracies)
                                avg_membrane_accuracy = sum(membrane_accuracies) / len(membrane_accuracies)
                                f.write(f"{expansion:<12}{avg_spike_accuracy:<20.2f}{avg_membrane_accuracy:<20.2f}\n")
                            elif config.model_type == 'conv':
                                avg_membrane_accuracy = sum(membrane_accuracies) / len(membrane_accuracies)
                                f.write(f"{expansion:<12}{'-':<20}{avg_membrane_accuracy:<20.2f}\n")

                    f.write(f"{'='*60}\n\n")

            print("Done testing accuracy for different expansions and ResNet models.")

        elif config.test_type == 'experiment_1':
            # Experiment 1: KNN-based OOD detection
            print("Experiment 1 on knn-based ood detection.")
            seeds = config.seeds
            expansions = config.expansions
            resnet_models = config.resnet_models

            statistics_exp_1(config, seeds, expansions, resnet_models)
            print("Done experiment 1.")
        elif config.test_type == 'experiment_2':
            # Experiment 2: Post-hoc OOD detectors
            print("Experiment 2 on post-hoc ood detectors.")
            seeds = config.seeds
            expansions = config.expansions
            resnet_models = config.resnet_models
            statistics_exp_2(config, seeds, expansions, resnet_models)
            print("Done experiment 2.")
        elif config.test_type == 'experiment_3':
            # Experiment 3: Combined KNN (penultimate features) + VIM (scoring-based)
            print("Experiment 3 on combined KNN and VIM with single extraction pass.")
            seeds = config.seeds
            expansions = config.expansions
            resnet_models = config.resnet_models
            statistics_exp_3(config, seeds, expansions, resnet_models)
            print("Done experiment 3.")
        
        else:
            print(f"Unknown test_type: {config.test_type}. Valid options: 'standard', 'population', 'single_step', 'accuracy', 'experiment_1', 'experiment_2', 'experiment_3'")

    elif config.mode == 'train':
        seeds = config.seeds
        expansions = config.expansions
        resnet_models = config.resnet_models
        
        # Ensure weights directories exist for saving checkpoints
        os.makedirs('weights/spike/exp'+str(config.case), exist_ok=True)
        os.makedirs('weights/conv/exp'+str(config.case), exist_ok=True)
        
        total_runs = len(seeds) * len(expansions) * len(resnet_models)
        current_run = 0
        
        print(f"Training sweep: {len(resnet_models)} models × {len(expansions)} expansions × {len(seeds)} seeds = {total_runs} total runs\n")
        
        for resnet_model in resnet_models:
            config.resnet_model = resnet_model

            for expansion in expansions:
                config.expansion = expansion

                for seed in seeds:
                    config.seed = seed
                    current_run += 1
                    
                    # Print configuration for this specific training run
                    print_experiment_info(
                        config, 
                        f"Training Run {current_run}/{total_runs}: ResNet{resnet_model} | Expansion={expansion} | Seed={seed}"
                    )
                    training(config)
                    if config.model_type == 'spike':
                        acc_spk, acc_mem = test_accuracy(config)
                        print(f"[SNN] Spike Accuracy: {acc_spk}, Membrane Accuracy: {acc_mem}")
                                
                    elif config.model_type == 'conv':
                        acc_mem = test_accuracy(config)
                        print(f"[CNN] Membrane Accuracy: {acc_mem}")

        print(f"\n{'='*70}")
        print(f"Training sweep completed: {total_runs} runs finished")
        print(f"{'='*70}\n")

    else:
        print(f"Mode not recognized: {config.mode}. Valid options: 'test', 'train'")
