from test import test_with_output, test_accuracy
from feature import feature_extraction_conv, feature_extraction_spike
from train import training
import numpy as np
from utils.Utils import Utils
# To enable downloading some datasets from pytorch
import ssl


if __name__ == "__main__":
    # Define defaut arguments and call training

    '''
    dataSet:        dataset like MNIST, CIFAR10
    modelType:      convolutional or spiking neural network, 'conv' or 'spike' 
    batchSize:      batch size

    ResNetModel 1:  Convolutional neural network based on Conv2D, and LIFs
    '''

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

    # Case 03: (old)
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet10Model (ResNetModel = 10)
    # Number of classes: 10
    # Batch size: 32

    # Case 03:
    # This is the same case as 03, with downsampling bug fixed
    # InDistribution: MNIST
    # OutOfDistribution: 'FMNIST', 'KMNIST', 'EMNIST', 'Letters'
    # Network model: SpikeResNet10ModelAlt (ResNetModel = 11)
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
    # Batch size: 40

    # Define case 
    case = '05'
    dataSet_ID = 'FMNIST'
    dataSet_feat = ['MNIST', 'FMNIST', 'KMNIST', 'Letters']  

    # dataSet_ID = 'SVHN'
    # dataSet_feat = ['CIFAR10','SVHN','Food101']

    modelType = 'spike'
    batchSize = 64
    numberOfClasses = 10
    ResNetModel = 21
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
    # training(dataSet_ID, modelType, batchSize, numberOfClasses, ResNetModel, fullTrain=True, pretrained=False)
    
    # Test accuracy of trained model on test
    # test_accuracy(dataSet_ID, modelType, batchSize, numberOfClasses, ResNetModel)
    
    # Feature extraction
    for i in range(len(dataSet_feat)):
        feature_extraction_spike(dataSet_ID, dataSet_feat[i], ResNetModel, case, numberOfClasses, batchSize)

    # Statistics
    stats = test_with_output(case=case, nameID=dataSet_ID)
    formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
    for row in formatted_stats:
        print(' '.join(row))

    # Vizualizer
    # Utils.visualize_feature('FMNIST', 'KMNIST', case)

