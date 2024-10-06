from utils.Utils import Utils
from metrics.Metrics import Metrics
from torch.utils.data import random_split
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from models.resnet9 import ResNet9Model 
from models.spikeresnet import spikeConvNN1, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model  
import snntorch.functional as SF
import numpy as np
from snntorch import utils
from snntorch import spikegen


def training(dataSet, modelType, batchSize, numOfClasses, ResNetModel, fullTrain=False):
    '''
    dataSet:        defines the data set for training (for example MNIST, FMNIST, KMNIST)
    modelType:      convolutional or spiking neural network
    case:           case needs to contain the details of the case scenario
    fullTrain:      define if training is done on complete training set or train/valid split is used
    ResNetModel:    determines number of layers in model
    '''

    # Load datase
    dataset_train, dataset_test = Utils.load_data(dataSet)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)
    print(f"Image size: {channels, rows, cols}")

    # Define batch size
    batchSize = batchSize
    
    if (fullTrain == False):
        train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, False)
    else:
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)
    
    # Get the device
    device = Utils.get_device()

    # Define training parameters
    # Number of classes
    if ((dataSet == 'MNIST') or (dataSet == 'KMNIST') or (dataSet == 'FMNIST')):
        numberOfClasses = 10
    elif ((dataSet == 'CIFAR10') or (dataSet == 'SVHN')):
        numberOfClasses = 10
    else:
        numberOfClasses = numOfClasses

    # Number of channels
    numberOfChannels = channels

    # Gradient clipping 
    gClip = 0.1

    # Weight decay
    wDecay = 0.0001

    if modelType == 'conv':    

        # Learning rate (KEY)
        lr = [0.0001]

        # Number of epochs
        numberOfEpochs = [20]

        # Loss function
        lossFunction = nn.CrossEntropyLoss()

        # Model
        if ResNetModel == 9:
            model = ResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses)
        else:
            print("Not defined")
            return -1
        
        # Move model to device
        model.to(device)

        summary(model, (3, 32, 32))

        # Keeps accuracy and loss for both training and validation in each epoch
        H = []

        # Training
        ##########

        for i in range(len(lr)):
            lrCurrent = lr[i]
            nEpochs = numberOfEpochs[i]
            if (fullTrain == False):
                model, H = model.fit_conv(nEpochs, model, lossFunction, lrCurrent, train_loader, val_loader, H, device, wd=wDecay, gd=gClip)
            elif (fullTrain == True):
                model, H = model.fit_conv_full_train(nEpochs, model, lossFunction, lrCurrent, train_loader, H, device, wd=wDecay, gd=gClip)
        print("\n")

        weightPath = 'weights/conv/resnet9_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)

        if (fullTrain == False):
            # Plot loss for train/validation data
            tLoss = [v[0] for v in H]
            vLoss = [v[2] for v in H]

            plt.figure(figsize = (5, 5))
            plt.plot(tLoss, '-bx')
            plt.plot(vLoss, '-rx')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(['Training', 'Validation'])
            plt.title('Loss/epochs')
            plt.show()

        print("Check test images.")
        testAcc = []
        testLoss = []
        for batch, labels in test_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            output, _ = model(batch)
            l = lossFunction(output, labels)
            testLoss.append(l.item())
            a = model.accuracy(output, labels)
            testAcc.append(a.item())
        # Test stats
        meanA = sum(testAcc) / len(testAcc)
        meanL = sum(testLoss) / len(testLoss)
        print(f'Test loss is {meanL:.2f}. Test accuracy is {meanA:.2f}.')

        return None

    elif modelType == 'spike':

        # Keeps accuracy and loss for both training and validation in each epoch
        H = []

        # For spiking neural network we need number of steps
        numberOfSteps = 50
        beta = 0.95
        threshold = 0.25

        # # Weight decay
        # wDecay = 0.0001

        # # Gradient clipping 
        # gClip = 0.1

        # Define model
        if ResNetModel == 1:
            model = spikeConvNN1(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 9:
            model = SpikeResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 18:
            model = SpikeResNet18Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        else:
            print("Not defined")
            return -1
        model = model.to(device)

        if dataSet == 'CIFAR10':
            numberOfEpochs = 100
        else:
            numberOfEpochs = 20

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=wDecay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=numberOfEpochs, steps_per_epoch=len(train_loader))

        # Loss function
        loss_fn = SF.ce_rate_loss()

        # Training
        if (fullTrain == True):
            H = model.fit_spike_full_train(model, numberOfEpochs, sched, optimizer, loss_fn, train_loader, numberOfSteps, gClip, device)
        else:
            print("Train/valid split not defined")
            return None

        weightPath = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)


def generate_latex(nameID, stats):
    if nameID == 'MNIST':
        namesOOD = ['FMNIST', 'KMNIST']
    elif nameID == 'FMNIST':
        namesOOD = ['MNIST', 'KMNIST']
    elif nameID == 'KMNIST':
        namesOOD = ['MNIST', 'FMNIST']
        
    latex_code = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \begin{document}

    \title{Sample Document}
    \author{Author Name}
    \date{\today}
    \maketitle

    \section{Introduction}
    This is a sample LaTeX document generated using Python.

    \section{Math Example}
    Here is an example of a mathematical expression:
    \[
    E = mc^2
    \]

    \end{document}
    """

    # Save to a .tex file
    with open('sample_document.tex', 'w') as f:
        f.write(latex_code)
    


# if __name__ == "__main__":
#     # Define defaut arguments and call training
#     training('CIFAR10', 'conv', case='08', fullTrain=True)
#     # test_accuracy('CIFAR10', 'spike')
#     # feature_extraction_spike('MNIST')
#     # feature_extraction_spike('FMNIST')
#     # feature_extraction_spike('KMNIST')
#     # feature_extraction_conv('FMNIST')
#     # feature_extraction_spike('SVHN')

#     # stats = test_with_output(case='07', nameID='CIFAR10')

#     # formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
#     # for row in formatted_stats:
#     #     print(' '.join(row))


#     # TODO: code manually every interesting LIF node and repeat the results
#     # TODO: check the rest of the apporaches, ODIN energy-based
#     # TODO: make utilities for automatic data processing
#     # TODO make utilities for graphics
#     # TODO: ResNet18
#     # TODO update feature extraction conv to enable diferent base 
#     # TODO utilize training cases, or make the weights names more flexibile 
#     # TODO move main to different file


#     '''
#     case 01: weights/spike/resnet9_MNIST_params.pth threshold is 1.0, LIFs are snn.Leaky(beta=beta, threshold=threshold, base dataset MNIST
#     case 02: OBSOLETE, REMOVE konvoluciona mreza 10-D izlaza
#     case 03: weights/spike/resnet9_MNIST_025_params.pth threshold je 0.25 beta je 0.95
#     case 04: conv Resnet9 weights/conv/resnet9_weights_MNIST.pth
#                           weights/conv/resnet9_weights_KMNIST.pth 
#                           weights/conv/resnet9_weights_FMNIST.pth

#     case 05: weights/spike/resnet9_weights_MNIST.pth spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)
#     case 06: weights/spike/resnet9_weights_MNIST_rate.pth rate encoding spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)
#     case 07: cifar10 vs svhn weights/spike/resnet9_weights_CIFAR10.pth spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)
#     case 08: spikeResNet18
#     '''