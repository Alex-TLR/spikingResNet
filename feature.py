from snntorch import utils
import torch
from snntorch import spikegen
from utils.Utils import Utils, SVHNDataset
from torch.utils.data import DataLoader
from models.resnet9 import ResNet9Model 
from models.spikeresnet import SpikeResNet9Model 
import numpy as np
import snntorch.functional as SF


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
