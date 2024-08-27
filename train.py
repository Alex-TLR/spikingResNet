from utils.Utils import Utils
from metrics.Metrics import Metrics
from torch.utils.data import random_split
import torch.nn.functional as f 
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader
from models.resnet9 import ResNet9Model #, CustomResNetConv
from models.spikeresnet9 import SpikeResNet9Model #, FeatureExtractor, CustomClassifier, CustomResNet
import snntorch.functional as SF
import numpy as np
from snntorch import utils
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
import time 

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

        # Number of channels
        numberOfChannels = 3

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
        model = ResNet9Model(numberOfChannels=numberOfChannels, numberOfClasses=numberOfClasses)

        # Move model to device
        model.to(device)

        summary(model, (3, 32, 32))

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
            if (fullTrain == False):
                model, H = model.fit(nEpochs, model, lossFunction, lrCurrent, train_loader, val_loader, H, device, wd=wDecay, gd=gClip)
            elif (fullTrain == True):
                model, H = model.fit_full_train(nEpochs, model, lossFunction, lrCurrent, train_loader, H, device, wd=wDecay, gd=gClip)
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

        weightPath = 'weights/spike/resnet9_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)

# TODO
# revise the definition
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
    batchSize = 50

    # For spiking neural network we need number of steps
    numberOfSteps = 50

    # Parameter of the LIF neuron
    beta = 0.95

    # Threshold
    threshold = 0.25

    # There is not validation data
    train_loader = DataLoader(dataset_train, batchSize)
    test_loader = DataLoader(dataset_test, batchSize)

    # Loss function
    loss_fn = SF.ce_rate_loss()

    # Define model
    model = SpikeResNet9Model(numberOfChannels=1, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
    # Load weights
    # weightsName = 'weights/spike/' + 'resnet9_MNIST_025_params.pth'
    dataSetID = 'MNIST'
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

        f, p = forward_pass_feature_extraction(model, numberOfSteps, batch)
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

        f, p = forward_pass_feature_extraction(model, numberOfSteps, batch)
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

    fileName = 'features/case_05/' + dataSet + '-on_mnist' + '.npz'
    np.savez(fileName, arr1=Feat_train, arr2=Prob_train, arr3=Tags_train, arr4=Feat_test, arr5=Prob_test, arr6=Tags_test)

    return None


# TODO make time consumption analysis
def test_with_output(case, nameID):
    '''
    case:       for example case 01 is '01'
    nameID:     name of the In Distribution features, example 'MNIST'
    '''
    if nameID == 'MNIST':
        namesOOD = ['FMNIST', 'KMNIST']
        suffixID = '-on_mnist'
    elif nameID == 'FMNIST':
        namesOOD = ['MNIST', 'KMNIST']
        suffixID = '-on_fmnist'
    elif nameID == 'KMNIST':
        namesOOD = ['MNIST', 'FMNIST']
        suffixID = '-on_kmnist'

    methods = ['MSP', 'NCM', 'KNN', 'NNDR', 'MD']
    # methods = ['MD', 'NNDR']
    stats = np.zeros((len(namesOOD), len(methods)*3), dtype=np.float64)
    IDpath = 'features/case_' + case + '/' + nameID + suffixID + '.npz'

    ID = np.load(IDpath)
    ID_feat_train = ID['arr1']  # In-Distribution training set features
    ID_prob_train = ID['arr2']  # In-Distribution training set outputs (usually with no softmax applied)
    ID_tags_train = ID['arr3']  # In-Distribution training set labels
    ID_feat_test  = ID['arr4']  # Out-of-Distribution training set features
    ID_prob_test  = ID['arr5']  # Out-of-Distribution training set outputs (usually with no softmax applied)
    ID_tags_test  = ID['arr6']  # Out-of-Distribution training set labels

    for i in range(len(namesOOD)):
        OODpath = 'features/case_' + case + '/' + namesOOD[i] + suffixID + '.npz'
        OOD = np.load(OODpath)
        OOD_feat_train = OOD['arr1']  # In-Distribution test set features
        OOD_prob_train = OOD['arr2']  # In-Distribution test set outputs (usually with no softmax applied)
        OOD_tags_train = OOD['arr3']  # In-Distribution test set labels
        OOD_feat_test  = OOD['arr4']  # Out-of-Distribution test set features
        OOD_prob_test  = OOD['arr5']  # Out-of-Distribution test set outputs (usually with no softmax applied)
        OOD_tags_test  = OOD['arr6']  # Out-of-Distribution test set labels

        # Methodology includes the IN/OOD classification where ID are positive samples taken from ID_test_set
        # and OOD are negative samples taken from OOD_train_set (or maybe OOD_train_set + OOD_test_set)

        for j in range(len(methods)):

            # iterate trought each of classification methods and calculate the metrics
            if methods[j] == 'MSP':
                # This is the baseline method that uses the outputs of the network number_of_classes-D 
                start_time = time.time() 
                ID_labels   = np.ones((len(ID_prob_test)))
                OOD_labels  = np.zeros((len(OOD_prob_test)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _ , ID_distances = Metrics.MSP(ID_prob_test, threshold)
                _, OOD_distances = Metrics.MSP(OOD_prob_test, threshold)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.MSP(ID_prob_test, threshold)
                OOD_predictions, _ = Metrics.MSP(OOD_prob_test, threshold)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"MSP Execution time: {execution_time:.4f} seconds")


            elif methods[j] == 'NCM':
                # All other methods use features
                number_classes = ID_prob_train.shape[1]
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = Metrics.NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes)
                _, OOD_distances = Metrics.NCM(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes)
                OOD_predictions, _ = Metrics.NCM(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 


            elif methods[j] == 'KNN':
                number_neighbors = 5 
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances = Metrics.KNN(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                _, OOD_distances = Metrics.KNN(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.KNN(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                OOD_predictions, _ = Metrics.KNN(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 


            elif methods[j] == 'NNDR':
                number_neighbors = 5 
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = Metrics.NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                _, OOD_distances = Metrics.NNDR(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                OOD_predictions, _ = Metrics.NNDR(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 


            elif methods[j] == 'MD':
                number_classes = ID_prob_train.shape[1]
                number_features = ID_feat_train.shape[1]
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = Metrics.MD(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
                _, OOD_distances = Metrics.MD(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes, number_features)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.MD(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
                OOD_predictions, _ = Metrics.MD(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes, number_features)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 

            else:
                pass

    return stats

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
    training('MNIST', 'spike', fullTrain=True)
    # feature_extraction_spike('MNIST')
    # feature_extraction_spike('FMNIST')
    # feature_extraction_spike('KMNIST')
    # feature_extraction_conv('FMNIST')

    # stats = test_with_output(case='05', nameID='MNIST')

    # formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
    # for row in formatted_stats:
    #     print(' '.join(row))


    # TODO: code manually every interesting LIF node and repeat the results
    # TODO: check the details about Baseline
    # TODO: check the rest of the apporaches, ODIN energy-based
    # TODO: make utilities for automatic data processing
    # TODO make utilities for graphics
    # TODO: ResNet18
    # TODO update feature extraction conv to enable diferent base 


    '''
    case 01: threshold is 1.0, LIFs are snn.Leaky(beta=beta, threshold=threshold, base dataset MNIST
    case 02: konvoluciona mreza 10-D izlaza
    case 03: threshold je 0.25 beta je 0.95
    case 04: conv Resnet9

    all convResnet9 grayscale are trained

    cas 05: spike-ResNet9 threshold je 0.25 beta je 0.95, last neuron is LI (not LIF)

    '''