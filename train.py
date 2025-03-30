from utils.Utils import Utils
#from metrics.Metrics import Metrics
from torch.utils.data import random_split
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
from torchsummary import summary
from models.resnet9 import ResNet9Model 
from models.spikeresnet import spikeConvNN1, spikeConvNN2,  SpikeResNet9Model, SpikeResNet10Model, SpikeResNet10ModelAlt, SpikeResNet18Model  
from models.plain import spikeLinearNet1
import snntorch.functional as SF
import numpy as np
from snntorch import utils
from snntorch import spikegen
import sys
import time 


def training(dataSet, modelType, batchSize, numOfClasses, ResNetModel, fullTrain=False, pretrained=False):
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
    # print(f"Image size: {channels, rows, cols}")
    sys.stdout.flush() 

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

        # Define model
        if ResNetModel == 1:
            model = spikeConvNN1(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 2:
            model = spikeConvNN2(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 9:
            model = SpikeResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 11:
            model = SpikeResNet10ModelAlt(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 18:
            model = SpikeResNet18Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 21:
            model = spikeLinearNet1(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        else:
            print("Not defined")
            return -1

        if (dataSet == 'CIFAR10') or (dataSet == 'SVHN'):
            numberOfEpochs = 200
        else:
            numberOfEpochs = 40
        model = model.to(device)
        

        # Loss function
        loss_fn = SF.ce_rate_loss()
        # Optimizer for gray 5e-4
        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=wDecay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=numberOfEpochs, steps_per_epoch=len(train_loader))
        # sched = None
        # Training
        startEpoch = 0
        if (fullTrain == True):
            if(pretrained == True):
                weightsName = 'weights/spike/' + 'resnet' + str(ResNetModel) + dataSet + '_checkpoint_' + '.pth'
                print("Try to load checkpoint: "+weightsName)
                try:
                    file = torch.load(weightsName)
                    model.load_state_dict(file["model"])
                    optimizer.load_state_dict(file["optimizer"])
                    sched.load_state_dict(file["lr_scheduler"])
                    startEpoch=file["epochs"] + 1
                    print(f"numberOfEpochs: {numberOfEpochs}, startEpoch: {startEpoch}, steps_per_epoch: {len(train_loader)}, total_steps: {sched.total_steps}")
                    sched._step_count = startEpoch * len(train_loader)
                    print("Checkpoint loaded.")
                except:
                    print("No valid checkpoint found. Starting from scratch.")
                print("Training started")
                sys.stdout.flush()

            start_time = time.time()
            H = model.fit_spike_full_train(model, startEpoch, numberOfEpochs, ResNetModel, dataSet, sched, optimizer, loss_fn, train_loader, numberOfSteps, gClip, device, checkpointPeriod=1)
            end_time = time.time()  # Record end time
            execution_time = end_time - start_time  # Calculate execution time
            print(f"Training time: {execution_time:.4f} seconds")
        else:
            print("Train/valid split not defined")
            return None

        weightPath = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)
        print("Training done.")
        return None

# TODO
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
    