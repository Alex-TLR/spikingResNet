from utils.Utils import Utils, SVHNDataset
from metrics.Metrics import Metrics
from torch.utils.data import random_split
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from models.resnet9 import ResNet9Model 
from models.spikeresnet9 import SpikeResNet9Model 
import snntorch.functional as SF
import numpy as np
from snntorch import utils
import time 
from snntorch import spikegen
from test import test_with_output


def training(dataSet, modelType, case='00', fullTrain=False):
    '''
    dataSet:        defines the data set for training (for example MNIST, FMNIST, KMNIST)
    modelType:      convolutional or spiking neural network
    case:           case needs to contain the details of the case scenario
    fullTrain:      define if training is done on complete training set or train/valid split is used
    '''

    # Load datase
    dataset_train, dataset_test = Utils.load_data(dataSet)

    # Get image size
    Utils.get_image_size(dataset_train, dataSet)

    # Define batch size
    batchSize = 32
    
    if (fullTrain == False):
        train_loader, val_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, False)
    else:
        train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)

        
    # Get the device
    device = Utils.get_device()

    # Define training parameters

    if modelType == 'conv':

        # Number of classes
        numberOfClasses = 10

        # Number of channels
        numberOfChannels = 3

        # Learning rate (KEY)
        lr = [0.0001]

        # Gradient clipping 

        gClip = 0.1

        # Number of epochs
        numberOfEpochs = [20]

        # Weight decay
        wDecay = 0.0001

        # Loss function
        lossFunction = nn.CrossEntropyLoss()

        # Model
        model = ResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses)

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

        case = case 
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
        
        # Number of classes
        numberOfClasses = 10

        # Number of epochs
        numberOfEpochs = 10

        # Keeps accuracy and loss for both training and validation in each epoch
        H = []

        # For spiking neural network we need number of steps
        numberOfSteps = 50
        beta = 0.95
        threshold = 0.25

        # Define model
        model = SpikeResNet9Model(numberOfChannels=1, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))

        # Loss function
        loss_fn = SF.ce_rate_loss()

        # Training
        if (fullTrain == True):
            H = model.fit_spike_full_train(model, numberOfEpochs, optimizer, loss_fn, train_loader, numberOfSteps, device)
        else:
            print("Train/valid split not defined")
            return None

        weightPath = 'weights/spike/resnet9_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)


def forward_pass_feature_extraction(model, numSteps, data):
        feat_trace = []
        prob_trace = []
        utils.reset(model)

        for _ in range(numSteps):
            _, feat_out, prob_out = model(data)
            feat_trace.append(feat_out)
            prob_trace.append(prob_out)

        feat_trace = torch.stack(feat_trace)
        prob_trace = torch.stack(prob_trace)

        return feat_trace, prob_trace


def forward_pass_rate_feature_extraction(model, numSteps, data):
        feat_trace = []
        prob_trace = []
        utils.reset(model)

        spike_data = spikegen.rate(data, num_steps=numSteps)
        for i in range(numSteps):
            _, feat_out, prob_out = model(spike_data[i])
            feat_trace.append(feat_out)
            prob_trace.append(prob_out)

        feat_trace = torch.stack(feat_trace)
        prob_trace = torch.stack(prob_trace)

        return feat_trace, prob_trace


def feature_extraction_conv(dataSet):
    '''
    Feature extraction out of regular ResNet9
    '''
    dataset_train, dataset_test = Utils.load_data(dataSet)

    device = Utils.get_device()

    trainDataSize = len(dataset_train)
    print("Train data size: ", trainDataSize)
    testDataSize = len(dataset_test)
    print("Test data size: ", testDataSize)

    # Number of classes
    numberOfClasses = 10

    # Batch size
    batchSize = 50

    # There is not validation data
    train_loader = DataLoader(dataset_train, batchSize)
    test_loader = DataLoader(dataset_test, batchSize)

    # Define model
    model = ResNet9Model(numberOfChannels=1, numberOfClasses=numberOfClasses)
    # print(model)
    # Load weights
    weightsName = 'weights/conv/' + 'resnet9_weights_' + dataSet + '.pth'
    model.load_state_dict(torch.load(weightsName, weights_only=True))
    model = model.to(device)

    # extractor = CustomResNetConv(num_classes=numberOfClasses, network=model) 
    # print(extractor)
    # extractor = extractor.to(device)

    Feat_train = np.zeros((trainDataSize, 512))
    Prob_train = np.zeros((trainDataSize, 10))
    Tags_train = []
    i = 0    
    for batch, labels in train_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        probs, feats = model(batch)
        labels = labels.cpu().detach().numpy()
        Feat_train[startIndex:endIndex, :] = feats.cpu().detach().numpy()
        Prob_train[startIndex:endIndex, :] = probs.cpu().detach().numpy()
        Tags_train.append(labels.flatten())
        del batch, labels, probs
        i += 1

    # Tags_train = np.array(Tags_train)
    Tags_train = np.concatenate(Tags_train)
    print("Tags_train shape is ", Tags_train.shape)

    Feat_test = np.zeros((testDataSize, 512))
    Prob_test = np.zeros((testDataSize, 10))
    Tags_test = []
    i = 0
    for batch, labels in test_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        probs, feats = model(batch)
        labels = labels.cpu().detach().numpy()
        Feat_test[startIndex:endIndex, :] = feats.cpu().detach().numpy()
        Prob_test[startIndex:endIndex, :] = probs.cpu().detach().numpy()
        Tags_test.append(labels.flatten())
        del batch, labels, probs
        i += 1

    # Tags_test = np.array(Tags_test)
    Tags_test = np.concatenate(Tags_test)
    print("Tags_test shape is ", Tags_test.shape)

    fileName = 'features/case_04/' + dataSet + '-on_mnist' + '.npz'
    np.savez(fileName, arr1=Feat_train, arr2=Prob_train, arr3=Tags_train, arr4=Feat_test, arr5=Prob_test, arr6=Tags_test)

    return None


def feature_extraction_spike(dataSet):
    '''
    Spiking models only
    dataSet:        the data set from which we extract feature
                    at the moment, features are extracted only on MNIST-trained network.
    
    '''

    dataset_train, dataset_test = Utils.load_data(dataSet)

    device = Utils.get_device()

    trainDataSize = len(dataset_train)
    print("Train data size: ", trainDataSize)
    testDataSize = len(dataset_test)
    print("Test data size: ", testDataSize)

    # Number of classes
    numberOfClasses = 10

    # Batch size
    batchSize = 20

    # For spiking neural network we need number of steps
    numberOfSteps = 50

    # Parameter of the LIF neuron
    beta = 0.95

    # Threshold
    threshold = 0.25

    # There is not validation data
    # train_loader = DataLoader(dataset_train, batchSize)
    # test_loader = DataLoader(dataset_test, batchSize)
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)


    # Loss function
    loss_fn = SF.ce_rate_loss()

    # Define model
    model = SpikeResNet9Model(numberOfChannels=3, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
    # Load weights
    # weightsName = 'weights/spike/' + 'resnet9_MNIST_025_params.pth'
    dataSetID = 'CIFAR10'
    weightsName = 'weights/spike/resnet9_weights_' + dataSetID + '.pth'
    model.load_state_dict(torch.load(weightsName, weights_only=True))
    model = model.to(device)

    Feat_train = np.zeros((trainDataSize, 512))
    Prob_train = np.zeros((trainDataSize, 10))
    Tags_train = []
    i = 0    
    for batch, labels in train_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        # print(i)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        # f, p = forward_pass_feature_extraction(model, numberOfSteps, batch)
        f, p = forward_pass_rate_feature_extraction(model, numberOfSteps, batch)
        f = f.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
            
        features = f.sum(axis=0)
        probs = p.max(axis=0)
        labels = labels.cpu().detach().numpy()
        Feat_train[startIndex:endIndex, :] = features
        Prob_train[startIndex:endIndex, :] = probs
        # print("labels shape is ", labels.shape)
        # Tags_train.append(labels)
        Tags_train.append(labels.flatten())
        del batch, labels, features, probs
        i += 1

    # Tags_train = np.array(Tags_train)
    Tags_train = np.concatenate(Tags_train)
    print("Tags_train shape is ", Tags_train.shape)

    Feat_test = np.zeros((testDataSize, 512))
    Prob_test = np.zeros((testDataSize, 10))
    Tags_test = []
    i = 0
    for batch, labels in test_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        # f, p = forward_pass_feature_extraction(model, numberOfSteps, batch)
        f, p = forward_pass_rate_feature_extraction(model, numberOfSteps, batch)
        f = f.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
            
        features = f.sum(axis=0)
        probs = p.max(axis=0)
        labels = labels.cpu().detach().numpy()
        Feat_test[startIndex:endIndex, :] = features
        Prob_test[startIndex:endIndex, :] = probs
        # Tags_test.append(labels)
        Tags_test.append(labels.flatten())
        del batch, labels, features, probs
        i += 1

    # Tags_test = np.array(Tags_test)
    Tags_test = np.concatenate(Tags_test)
    print("Tags_test shape is ", Tags_test.shape)

    fileName = 'features/case_07/' + dataSet + '-on_cifar10' + '.npz'
    np.savez(fileName, arr1=Feat_train, arr2=Prob_train, arr3=Tags_train, arr4=Feat_test, arr5=Prob_test, arr6=Tags_test)

    return None


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
    


if __name__ == "__main__":
    # Define defaut arguments and call training
    # training('SVHN', 'conv', case='07', fullTrain=False)
    # feature_extraction_spike('MNIST')
    # feature_extraction_spike('FMNIST')
    # feature_extraction_spike('KMNIST')
    # feature_extraction_conv('FMNIST')
    # feature_extraction_spike('SVHN')

    stats = test_with_output(case='07', nameID='CIFAR10')

    formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
    for row in formatted_stats:
        print(' '.join(row))


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
    case 07: cifar10 vs svhn
    '''