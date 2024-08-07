from utils.Utils import Utils
from torch.utils.data import random_split
import torch.nn.functional as f 
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader
from models.resnet9 import ResNet9Model
from models.spikeresnet9 import SpikeResNet9Model, FeatureExtractor, CustomClassifier, CustomResNet
import snntorch.functional as SF

def training(dataSet, modelType, fullTrain=False):
    '''
    dataSet:        defines the data set for training (for example MNIST, FMNIST, KMNIST)
    modelType:      convolutional or spiking neural network
    fullTrain:      define if training is done on complete training set or train/valid split is used
    '''

    torch.manual_seed(42)

    dataset_train, dataset_test = Utils.load_data(dataSet)

    # Get the image size
    print("Get image size ...")
    train_tensor, train_label = dataset_train[0]
    imageSize = train_tensor.size()
    print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')
    inputSize = imageSize[0] * imageSize[1] * imageSize[2]

    if (fullTrain == False):
        print("Define training/validation split ...")
        dataSize = len(dataset_train)
        tSize = int(0.8 * dataSize)
        vSize = dataSize - tSize

        train_data, val_data = random_split(dataset_train, [tSize, vSize])
        test_data = dataset_test
        print("Train data length ", len(train_data))
        print("Valid data length ", len(val_data))
        print("Test data length ", len(test_data))

    else:
        train_data = dataset_train
        test_data = dataset_test
        print("Train data length ", len(train_data))
        print("Test data length ", len(test_data))

    device = Utils.get_device()

    # Define training parameters

    if modelType == 'conv':

        # Number of classes
        numberOfClasses = 10

        # Batch size
        batchSize = 64

        # Learning rate (KEY)
        lr = [0.0001]

        # Gradient clipping
        gClip = 0.1

        # Number of epochs
        numberOfEpochs = [10]

        # Weight decay
        wDecay = 0.0001

        # Loss function
        lossFunction = nn.CrossEntropyLoss()

        # Model
        model = ResNet9Model(numberOfChannels=1, numberOfClasses=numberOfClasses)

        # Move model to device
        model.to(device)

        # Keeps accuracy and loss for both training and validation in each epoch
        H = []

        if (fullTrain == True):
            train_loader = DataLoader(train_data, batchSize, shuffle=True)
            test_loader = DataLoader(test_data, batchSize)
        else:
            train_loader = DataLoader(train_data, batchSize, shuffle=True)
            val_loader = DataLoader(val_data, batchSize)
            test_loader = DataLoader(test_data, batchSize)

        # Training
        ##########

        for i in range(len(lr)):
            lrCurrent = lr[i]
            nEpochs = numberOfEpochs[i]
            model, H = model.fit(nEpochs, model, lossFunction, lrCurrent, train_loader, val_loader, H, device, wd=wDecay, gd=gClip)
        print("\n")

        weightPath = 'weights/conv/restnet9_weights_' + dataSet + '.pth'
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
            output = model(batch)
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

        # Batch size
        batchSize = 16

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

        if (fullTrain == True):
            train_loader = DataLoader(train_data, batchSize, shuffle=True)
            test_loader = DataLoader(test_data, batchSize)
        else:
            train_loader = DataLoader(train_data, batchSize, shuffle=True)
            val_loader = DataLoader(val_data, batchSize)
            test_loader = DataLoader(test_data, batchSize)

        # Training
        if (fullTrain == True):
            H = model.fitS_train(model, numberOfEpochs, optimizer, loss_fn, train_loader, numberOfSteps, device)
        else:
            print("Train/valid split not defined")
            return None

        weightPath = 'weights/spike/restnet9_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)


if __name__ == "__main__":
    # Define defaut arguments and call training
    training('MNIST', 'spike', fullTrain=True)