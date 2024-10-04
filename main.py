from test import test_with_output, test_accuracy
from feature import feature_extraction_conv, feature_extraction_spike, visualize_feature
from train import training
import numpy as np


if __name__ == "__main__":
    # Define defaut arguments and call training

    '''
    dataSet:        dataset like MNIST, CIFAR10
    modelType:      convolutional or spiking neural network, 'conv' or 'spike' 
    batchSize:      batch size
    '''
    dataSet = 'SVHN'
    modelType = 'spike'
    batchSize = 24
    numberOfClasses = 10
    numberOfChannels = 3
    ResNetModel = 1
    # ResNetModel =1 is the first cplain conv model
    
    case = '11'
    training(dataSet, modelType, batchSize, numberOfClasses, numberOfChannels, ResNetModel, case=case, fullTrain=True)
    test_accuracy(dataSet, 'spike', numberOfClasses, numberOfChannels, ResNetModel)
    # visualize_feature('10')
    
    # feature_extraction_spike(dataSet, ResNetModel, case, numberOfClasses, numberOfChannels)
    # feature_extraction_spike('FMNIST', ResNetModel, case, numberOfClasses, numberOfChannels)
    # feature_extraction_spike('KMNIST', ResNetModel, case, numberOfClasses, numberOfChannels)
    # feature_extraction_conv('FMNIST')
    # feature_extraction_spike('Places365', '08')

    # stats = test_with_output(case=case, nameID='MNIST')

    # formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
    # for row in formatted_stats:
    #     print(' '.join(row))


    # TODO: code manually every interesting LIF node and repeat the results
    # TODO: check the rest of the apporaches, ODIN energy-based
    # TODO: make utilities for automatic data processing
    # TODO make utilities for graphics
    # TODO: ResNet18
    # TODO update feature extraction conv to enable diferent base 
    # TODO utilize training cases, or make the weights names more flexibile 
    # TODO move main to different file


    '''
    case 01: weights/spike/resnet9_MNIST_params.pth threshold is 1.0, LIFs are snn.Leaky(beta=beta, threshold=threshold, base dataset MNIST
    case 02: OBSOLETE, REMOVE konvoluciona mreza 10-D izlaza
    case 03: weights/spike/resnet9_MNIST_025_params.pth threshold je 0.25 beta je 0.95
    case 04: conv Resnet9 weights/conv/resnet9_weights_MNIST.pth
                          weights/conv/resnet9_weights_KMNIST.pth 
                          weights/conv/resnet9_weights_FMNIST.pth

    case 05: weights/spike/resnet9_weights_MNIST.pth spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)
    case 06: weights/spike/resnet9_weights_MNIST_rate.pth rate encoding spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)
    case 07: cifar10 vs svhn weights/spike/resnet9_weights_CIFAR10.pth spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)
    case 08: spikeResNet18 CIFAR10 obucavanje
    case 10: MNIST ResNET10 novi model
    case 11: spike CNN 1
    '''