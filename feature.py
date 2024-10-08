from snntorch import utils
import torch
from snntorch import spikegen
from utils.Utils import Utils, SVHNDataset
from torch.utils.data import DataLoader
from models.resnet9 import ResNet9Model 
from models.spikeresnet import spikeConvNN1, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model
import numpy as np
import snntorch.functional as SF
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from metrics.Metrics import Metrics


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
        feats = feats.cpu().detach().numpy()
        probs = probs.cpu().detach().numpy()
        print(f'feats.shape is {feats.shape} and probs.shape is {probs.shape}')
        Feat_train[startIndex:endIndex, :] = feats
        Prob_train[startIndex:endIndex, :] = probs
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


def feature_extraction_spike(dataSet_ID, dataSet_feat, ResNetModel, case, numOfClasses):
    '''
    Spiking models only
    dataSet:        the data set from which we extract feature
                    at the moment, features are extracted only on MNIST-trained network.
    dataSet_ID:     in-definition dataset
    dataSet_feat:   dataset for feature extraction
    ResNetModel:    select resnet model type (resnet10, resnet18)
    numOfClasses:   number of classes
    numOfChannels:  number of input channels
    '''

    print(dataSet_feat)
    dataset_train, dataset_test = Utils.load_data(dataSet_feat)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet_feat)
    print(f"Image size: {channels, rows, cols}")

    device = Utils.get_device()

    trainDataSize = len(dataset_train)
    print("Train data size: ", trainDataSize)
    testDataSize = len(dataset_test)
    print("Test data size: ", testDataSize)
    print(f"Network models is {ResNetModel}")

    # Number of classes
    numberOfClasses = numOfClasses

    # Batch size
    batchSize = 12

    # For spiking neural network we need number of steps
    numberOfSteps = 50

    # Parameter of the LIF neuron
    beta = 0.95

    # Threshold
    threshold = 0.25

    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet_feat, True)

    # Loss function
    loss_fn = SF.ce_rate_loss()

    # Define model
    if ResNetModel == 1:
        model = spikeConvNN1(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        featSize = 256
    elif ResNetModel == 9:
        model = SpikeResNet9Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        featSize = 512
    elif ResNetModel == 10:
        model = SpikeResNet10Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        featSize = 512
    elif ResNetModel == 18:
        model = SpikeResNet18Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        featSize = 512
    else:
        print("Not defined")
        return -1
    
    # Load weights
    # Loading the weights for the ID-trained network
    weightsName = 'weights/spike/resnet' + str(ResNetModel) + '_weights_' + dataSet_ID + '.pth'
    model.load_state_dict(torch.load(weightsName, weights_only=True))
    model = model.to(device)

    Feat_train = np.zeros((trainDataSize, featSize))
    Prob_train = np.zeros((trainDataSize, 10))
    Tags_train = []
    i = 0    
    for batch, labels in train_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        s, f, p = model(batch, numberOfSteps)
        f = f.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
        s = s.cpu().detach().numpy()
        s = s.sum(axis=0)
        # print(f'feats.shape is {f.shape} and probs.shape is {p.shape}')
        features = f.sum(axis=0)
        probs = p.max(axis=0)
        # if i < 5:
        #     # print(f'{i}\n{s}\n{probs}')
        #     plotProb(p, batchSize, numberOfSteps)
        probs = Metrics.softmax(probs)
        # print(f'feats.shape is {features.shape} and probs.shape is {probs.shape}')
        labels = labels.cpu().detach().numpy()
        Feat_train[startIndex:endIndex, :] = features
        Prob_train[startIndex:endIndex, :] = probs
        Tags_train.append(labels.flatten())
        del batch, labels, features, probs
        i += 1

    # Tags_train = np.array(Tags_train)
    Tags_train = np.concatenate(Tags_train)
    print("Tags_train shape is ", Tags_train.shape)

    Feat_test = np.zeros((testDataSize, featSize))
    Prob_test = np.zeros((testDataSize, 10))
    Tags_test = []
    i = 0
    for batch, labels in test_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        startIndex = i*batchSize
        endIndex = startIndex + batchSize

        _, f, p = model(batch, numberOfSteps)
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

    fileName = 'features/spike/case_' + case + '/' + dataSet_feat + '-on_' + dataSet_ID.lower() + '.npz'
    print(fileName)
    np.savez(fileName, arr1=Feat_train, arr2=Prob_train, arr3=Tags_train, arr4=Feat_test, arr5=Prob_test, arr6=Tags_test)

    return None

def plotProb(prob, batchSize, numOfSteps):
    '''
    Plot one probability 
    '''
    print(f'Probs shape is {prob.shape}')
    # for i in range(batchSize):
    #     p = prob[:, i, 0].squeeze()

    #     plt.figure(figsize=(10, 5))
    #     plt.plot(p, marker='o')  # Convert to NumPy for plotting
    #     plt.title('Memebrane voltage')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Voltage')
    #     plt.grid(True)
    #     plt.show()
    # Determine the number of columns and rows
    num_cols = 3
    num_rows = int(np.ceil(batchSize / num_cols))

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i in range(batchSize):
        p = prob[:, i, 1].squeeze()

        axes[i].plot(p, marker='o')  # Plot on the ith subplot
        axes[i].set_title(f'Membrane Voltage for Sample {i+1}')
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel('Voltage')
        axes[i].grid(True)

    # Hide any unused subplots
    for j in range(batchSize, num_rows * num_cols):
        fig.delaxes(axes[j])
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    probs_ = prob.max(axis=0)
    print(f'Max value is {probs_}')



def visualize_feature(case):
    '''
    plot reduced features
    '''

    tsne = TSNE(n_components=2, random_state=42)

    folderName = 'features/case_' + case + '/'
    print(folderName)
    files = os.listdir(folderName)
    print(files)

    # files = [f for f in os.listdir(folderName) if os.path.isfile(os.path.join(folderName, f))]
    # print(files)

    features = []

    border = 0

    for i in range(len(files)):
        fileName = files[i]
        print(fileName)
        filePath = folderName + fileName
        F = np.load(filePath)
        feat_train = F['arr1']  # In-Distribution training set features
        prob_train = F['arr2']  # In-Distribution training set outputs (usually with no softmax applied)
        tags_train = F['arr3']  # In-Distribution training set labels
        feat_test  = F['arr4']  # Out-of-Distribution training set features
        prob_test  = F['arr5']  # Out-of-Distribution training set outputs (usually with no softmax applied)
        tags_test  = F['arr6']  # Out-of-Distribution training set labels

        if i == 0:
            features1 = feat_test
            border = len(feat_test)
        if i == 1:
            features2 = feat_train
             

    print(f'features1.shape is {features1.shape}')
    print(f'features2.shape is {features2.shape}')
    features = np.vstack((features1, features2))
    print(f'features.shape is {features.shape}')
    features_tsne = tsne.fit_transform(features)

    features1_tsne = features_tsne[:border]
    features2_tsne = features_tsne[border:]

    np.savez('temp.npz', arr1=features1_tsne, arr2=features2_tsne)

    print(f'features_tsne shape is {features_tsne.shape}')

    plt.figure(figsize=(10, 8))

    # Plot the first set of features
    plt.scatter(features1_tsne[:, 0], features1_tsne[:, 1], c='blue', label='Dataset 1', alpha=0.5)

    # Plot the second set of features
    plt.scatter(features2_tsne[:, 0], features2_tsne[:, 1], c='red', label='Dataset 2', alpha=0.5)

    # Add labels and legend
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Feature Vectors')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    # Initialize interactive mode
    # plt.ion()

    # # Create a 3D plot
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the first set of features
    # ax.scatter(features1_tsne[:, 0], features1_tsne[:, 1], features1_tsne[:, 2], 
    #         c='blue', label='Dataset 1', alpha=0.6, s=50)

    # # Plot the second set of features
    # ax.scatter(features2_tsne[:, 0], features2_tsne[:, 1], features2_tsne[:, 2], 
    #         c='red', label='Dataset 2', alpha=0.6, s=50)

    # # Add labels and legend
    # ax.set_xlabel('t-SNE Dimension 1')
    # ax.set_ylabel('t-SNE Dimension 2')
    # ax.set_zlabel('t-SNE Dimension 3')
    # ax.set_title('3D t-SNE Visualization of Feature Vectors')
    # ax.legend()

    # # Show the plot
    # plt.show()

    # # Disable interactive mode
    # plt.ioff()


    return None