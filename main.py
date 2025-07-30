import torch
from test import test_metrics, test_accuracy
from feature import feature_extraction_conv, feature_extraction_spike
from train import training
import numpy as np
from utils.Utils import Utils
# To enable downloading some datasets from pytorch
import ssl
import argparse

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

if __name__ == "__main__":
    # Define defaut arguments and call training

    '''
    dataSet:        dataset like MNIST, CIFAR10
    modelType:      convolutional or spiking neural network, 'conv' or 'spike' 
    batchSize:      batch size

    ResNetModel 1:  Convolutional neural network based on Conv2D, and LIFs
    '''
    parser=get_parser()
    args = parser.parse_args()
    # Case 01:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: spikeConvNN1 (ResNetModel = 1)
    # Number of classes: 10
    # Batch size: 32 

    # Case 02:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet9Model (ResNetModel = 9)
    # Number of classes: 10
    # Batch size: 32

    # Case 03:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Number of classes: 10
    # Batch size: 32

    # Case 04:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: spikeConvNN2 (ResNetModel = 2)
    # Number of classes: 10
    # Batch size: 32

    # Case 05:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: spikeLinearNet1 (ResNetModel = 21)
    # Number of classes: 10
    # Batch size: 32

    # Case 06:
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet18Model (ResNetModel = 18)
    # Number of classes: 10
    # Batch size: 32

    # Case 07:
    # spike-ResNet10 model with trainable initialization
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Number of classes: 10
    # Batch size: 32

    # Case 08:
    # SEW model ResNet18 test accuracy is 82.53%
    # Network model: SEW ResNet18 (ResNetModel = 22)

    # Case 09:
    # spike-ResNet20 model 
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet20Model (ResNetModel = 20)
    # Number of classes: 10
    # Batch size: 32

    # Define case
    # match args.case:
    #     case '06':
    #         case = '06'
    #         dataSet_ID = 'CIFAR10'
    #         dataSet_feat = ['CIFAR10','SVHN','Food101'] 
    #     case '07':
    #         case = '07'
    #         dataSet_ID = 'SVHN'
    #         dataSet_feat = ['CIFAR10','SVHN','Food101']
    #     case '08':
    #         case = '08'
    #         dataSet_ID = 'MNIST'
    #         dataSet_feat = ['MNIST','FMNIST','KMNIST','Letters']
    #     case '09':
    #         case = '09'
    #         dataSet_ID = 'FMNIST'
    #         dataSet_feat = ['MNIST','FMNIST','KMNIST','Letters']
    #     case '10':
    #         case = '10'
    #         dataSet_ID = 'KMNIST'
    #         dataSet_feat = ['MNIST','FMNIST','KMNIST','Letters']

    # Define case 
    case = '09'
    # dataSet_ID = 'Letters'
    # dataSet_feat = ['MNIST', 'FMNIST', 'KMNIST', 'Letters']  

    dataSet_ID = 'CIFAR10'
    dataSet_feat = ['CIFAR10','SVHN','Food101']

    modelType = 'spike'
    # batchSize = args.batch_size
    # numberOfClasses = args.num_classes
    # ResNetModel = args.model

    batchSize = 128
    numberOfClasses = 10
    ResNetModel = 20
    ssl._create_default_https_context = ssl._create_unverified_context
    # # Load datase
    # dataset_train, dataset_test = Utils.load_data(dataSet_ID)

    # # Get image size
    # channels, rows, cols = Utils.get_image_size(dataset_train, dataSet_ID)
    # print(f"Image size: {channels, rows, cols}")

    # Create features case
    Utils.make_features_dir(modelType, case)
    
    # Training
    # set pretrained=True if continious training is needed
    
    # Davor
    # if args.mode == 'train':
    #     dataSet_ID = args.dataset_ID
    #     training(dataSet_ID, modelType, batchSize, numberOfClasses, ResNetModel, epochs=args.epochs, fullTrain=True, pretrained=args.pretrained)
    # if args.mode == 'test':
    #     # Test accuracy of trained model on test
    #     test_accuracy(dataSet_ID, modelType, batchSize, numberOfClasses, ResNetModel)
        
    #     # Feature extraction
    #     for i in range(len(dataSet_feat)):
    #         feature_extraction_spike(dataSet_ID, dataSet_feat[i], ResNetModel, case, numberOfClasses, batchSize)
    #############################

    training(dataSet_ID, modelType, batchSize, numberOfClasses, ResNetModel, epochs=400, fullTrain=True, auto_aug=True, pretrained=False)
    
    # Test accuracy of trained model on test
    test_accuracy(dataSet_ID, modelType, batchSize, numberOfClasses, ResNetModel)
    
    # Feature extraction
    # for i in range(len(dataSet_feat)):
    #     feature_extraction_spike(dataSet_ID, dataSet_feat[i], ResNetModel, case, numberOfClasses, batchSize)

    # Statistics
    # stats = test_metrics(case=case, nameID=dataSet_ID)
    # formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
    # for row in formatted_stats:
    #     print(' '.join(row))

    # Vizualizer
    # Utils.visualize_feature('Letters', 'FMNIST', case)

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
