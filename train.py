from utils.Utils import Utils
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
    train_tensor, train_label = dataset_train.data[0]
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

        weightPath = 'weights/spike/restnet9_weights_' + dataSet + '.pth'
        torch.save(model.state_dict(), weightPath)

# TODO
# revise the definition
def forward_pass(model, numSteps, data):
        feat_trace = []
        prob_trace = []
        utils.reset(model)

        for _ in range(numSteps):
            feat_out, _, prob_out = model(data)
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
    numberOfSteps = 25

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
    weightsName = 'weights/spike/' + 'resnet9_MNIST_025_params.pth'
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

        f, p = forward_pass(model, numberOfSteps, batch)
        f = f.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
            
        features = f.sum(axis=0)
        probs = p.sum(axis=0)
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

        f, p = forward_pass(model, numberOfSteps, batch)
        f = f.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
            
        features = f.sum(axis=0)
        probs = p.sum(axis=0)
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

    fileName = 'features/case_03/' + dataSet + '-on_mnist' + '.npz'
    np.savez(fileName, arr1=Feat_train, arr2=Prob_train, arr3=Tags_train, arr4=Feat_test, arr5=Prob_test, arr6=Tags_test)

    return None

def softmax(x):
    # Subtract the max value for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def MSP(test_data, threshold):
    '''
    test_data
    threshold
    '''
    # print("Test_data \n", test_data[0:10, :])
    s = softmax(test_data)
    # print("Softmax \n", s[0:10, :])
    max_probabilities = np.max(s, axis=1)
    predictions = (max_probabilities > threshold).astype(np.int32)
    return predictions, max_probabilities


def NCM(train_X, train_Y, test_X, threshold, num_classes, num_features):
    '''
    train_X:            matrix with train features
    train_Y:            array with train labels
    test_X:             matrix with test features
    num_classes:        number of classes
    num_features:       feature dimension
    '''

    # centroids_X = np.zeros((num_classes, num_features))
    # num_samples_per_class = np.zeros((num_classes))

    # for i in range(len(train_X)): 
    #     centroids_X[train_Y[i]] += train_X[i]
    #     num_samples_per_class[train_Y[i]] += 1
    # for i in range(len(centroids_X)):
    #     centroids_X[i] /= num_samples_per_class[i]

    # # Compute centroids
    centroids_X = np.array([train_X[train_Y == c].mean(axis=0) for c in range(num_classes)])

    predictions = np.zeros(len(test_X))
    distances = np.zeros(len(test_X))

    for i in range(len(test_X)):
        dist = np.zeros(num_classes)
        for j in range(num_classes):
            dist[j] = np.sqrt(np.sum((centroids_X[j] - test_X[i])**2))
        distances[i] = -dist.min()
        predictions[i] = 1 if distances[i] > threshold else 0

    return predictions, distances

def KNN(train_X, train_Y, test_X, threshold, number_neighbors):
    '''
    train_X:            matrix with train features
    train_Y:            array with train labels
    test_X:             matrix with test features
    threshold:          threshold
    num_classes:        number of classes
    num_features:       feature dimension
    '''

    neigh = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'euclidean')
    neigh.fit(train_X, train_Y)

    predictions = neigh.predict(test_X)
    neigh_dist, _ = neigh.kneighbors(test_X, return_distance = True)

    # distances = []
    # for i in range(len(neigh_dist)):
    #     distances.append(-np.average(neigh_dist[i]))
    #     if(distances[i] > threshold):
    #         predictions[i] = 1
    #     else:
    #         predictions[i] = 0

    distances = -np.average(neigh_dist, axis=1)
    predictions = (distances > threshold).astype(np.int32)

    # Calculate average distances and apply threshold
    # avg_distances = -np.mean(neigh_dist, axis=1)
    # predictions = (avg_distances > threshold).astype(int)

    return predictions, distances

def NNDR(train_X, train_Y, test_X, threshold, n_neighbors):
    '''
    train_X:            matrix with train features
    train_Y:            array with train labels
    test_X:             matrix with test features
    threshold:          threshold
    n_neighbors:        number of neighbors
    '''
    distances = []
    m = []

    neigh = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'euclidean')
    neigh.fit(train_X, train_Y)

    predictions = neigh.predict(test_X)
    neigh_dist, neigh_ind = neigh.kneighbors(test_X, return_distance = True)

    for i in range(len(neigh_dist)):
        for j in range(len(neigh_dist[i])):
            if(train_Y[neigh_ind[i][j]] != train_Y[neigh_ind[i][0]]):
                break
        m.append(j)

    for i in range(len(neigh_dist)):
        distances.append(-neigh_dist[i][0]/neigh_dist[i][m[i]])
    if(distances[i] > threshold):
        predictions[i] = 1
    else:
        predictions[i] = 0

    return predictions, distances

def MD(train_X, train_Y, test_X, threshold, n_classes, n_features):
    '''
    train_X:            matrix with train features
    train_Y:            array with train labels
    test_X:             matrix with test features
    threshold:          threshold
    n_classes:          number of classes
    n_features:         dimensionality of feature vector
    '''
    mean_vectors = np.zeros((n_classes, n_features))

    b = 0

    for i in range(0, n_classes):
        for j in range(len(train_Y)):
            if(train_Y[j] == i):
                b = b + 1
                mean_vectors[i] = mean_vectors[i] + train_X[j]
        mean_vectors[i] = mean_vectors[i] / b
        b = 0

    pom = np.zeros((len(train_Y), n_features))

    cov = np.zeros((n_features, n_features))

    for i in range(0, n_classes):
        for j in range(len(train_Y)):
            if(train_Y[j] == i):
                pom[j] = train_X[j]-mean_vectors[i]

    data = np.asmatrix(pom).transpose()

    cov = np.cov(data, bias=False)

    #print(cov)
    cov = cov+0.001*np.identity(n_features)

    cov_inv = np.linalg.inv(cov)

    MDK = np.zeros((len(test_X), n_classes))
    C = np.zeros((len(test_X), 1))
    arg = np.zeros((len(test_X), 1))
    prediction = []

    cov_inv = np.linalg.inv(cov)

    for i in range(len(test_X)):
        for j in range(len(mean_vectors)):
            MDK[i][j] = math.pow(distance.mahalanobis(test_X[i], mean_vectors[j], cov_inv), 2)

        C[i] = -np.min(MDK[i])
        arg[i] = np.argmin(MDK[i])
        if(C[i] > threshold):
            prediction.append(1)
        else:
            prediction.append(0)

    return prediction, C

def find_threshold(test_Y, test_Dist, positive_label, drop = True):
    '''
    da vidimo sta je sta?
    test_Y:         set of '1' and '0' representing true cases
    test_Dist:      max values of coresponding softmax values, should be that the max softmax value match true cases
    positive_label: the insignia of positive label, positive is ID, negative is OOD
    drop:           drop_intermediate ??
    A boolean flag indicating whether to drop some points on the ROC curve to make it more interpretable. 
    If set to True, the function will drop points where the false positive rate (FPR) is the same as in previous points.
    '''
    fpr, tpr, thresholds = roc_curve(test_Y, test_Dist, pos_label = positive_label, drop_intermediate = drop)
    # print("fpr ", fpr)
    # print("tpr ", tpr)
    # print("threshold ", thresholds)

    # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.grid(True)
    # plt.show()

    '''
    fpr: Array of false positive rates. The false positive rate at different threshold values. 
    It represents the proportion of actual negatives that are incorrectly classified as positives.
    tpr: Array of true positive rates (also known as recall or sensitivity). 
    The true positive rate at different threshold values. It represents the proportion of actual positives that 
    are correctly classified as positives.
    thresholds: Array of threshold values used to compute the fpr and tpr. 
    These thresholds are used to determine the decision boundary for classifying the instances as positive or negative.
    
    '''

    dtpr80 = np.absolute(tpr-0.8)
    threshold_tpr80 = thresholds[dtpr80.argmin()]

    dtpr95 = np.absolute(tpr-0.95)
    threshold_tpr95 = thresholds[dtpr95.argmin()]

    return threshold_tpr80, threshold_tpr95


def test_OOD_ID(case, nameID, nameOoD):
    '''
    case:       for example case 01 is '01'
    nameID:     name of the In Distribution features, example 'MNIST-on_mnist'
    nameOoD:    name of the Out Of Distribution features 
    '''
    IDpath = 'features/case_' + case + '/' + nameID + '.npz'
    OoDpath = 'features/case_' + case + '/' + nameOoD + '.npz'

    print(IDpath)
    print(OoDpath)

    ID = np.load(IDpath)
    OoD = np.load(OoDpath)

    ID_feat_train = ID['arr1']
    ID_prob_train = ID['arr2']
    ID_tags_train = ID['arr3']
    ID_feat_test = ID['arr4']
    ID_prob_test = ID['arr5']
    ID_tags_test = ID['arr6']

    OoD_feat_train = OoD['arr1']
    OoD_prob_train = OoD['arr2']
    OoD_tags_train = OoD['arr3']
    OoD_feat_test = OoD['arr4']
    OoD_prob_test = OoD['arr5']
    OoD_tags_test = OoD['arr6']

    print("Maximum softmax probability (MSP):")
    start_time = time.time() 
    # Make labels
    ID_Y = np.ones((len(ID_prob_test)))
    OoD_Y = np.zeros((len(OoD_prob_train)))
    test_Y = np.concatenate((ID_Y, OoD_Y))

    threshold = 0

    ID_predictions, ID_distances = MSP(ID_prob_test, threshold)
    OOD_predictions, OOD_distances = MSP(OoD_prob_train, threshold)
    test_distances = np.concatenate((ID_distances, OOD_distances))
    threshold_tpr80, threshold_tpr95 = find_threshold(test_Y, test_distances, 1, drop = False)
    print(f"MSP: Find distances with the thresold value {threshold_tpr95}.")
    threshold = threshold_tpr95
    ID_predictions, ID_distances = MSP(ID_prob_test, threshold)
    OOD_predictions, OOD_distances = MSP(OoD_prob_train, threshold)
    test_distances = np.concatenate((ID_distances, OOD_distances))
    pred = np.concatenate((ID_predictions, OOD_predictions))
    cm = confusion_matrix(test_Y, pred, labels=[0, 1])
    print("MPS: Confusion matrix\n",cm)
    TPR = cm[1][1] / (cm[1][1] + cm[1][0])
    FPR = cm[0][1] / (cm[0][1] + cm[0][0])
    print(f'MPS: TPR is {TPR:.4f}, FPR is {FPR:.4f}.')
    auroc = roc_auc_score(test_Y, test_distances)
    print(f'MPS: Area under ROC is {auroc}.')
    apr = average_precision_score(test_Y, test_distances)
    print(f'MPS: Average precision score is {apr}.')
    precision, recall, thrs = precision_recall_curve(test_Y, test_distances)
    aupr = auc(recall, precision)
    print(f'MPS: AUPR is {aupr}.')
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"MSP Execution time: {execution_time:.4f} seconds")

    print("NCM:")
    start_time = time.time() 
    number_classes = 10 # postaviti po obiljezjima
    number_features = 512 # takodje kao i gore 
    threshold = 0

    # Make labels
    ID_Y = np.ones((len(ID_prob_test)))
    OoD_Y = np.zeros((len(OoD_prob_train)))
    test_Y = np.concatenate((ID_Y, OoD_Y))

    # ID_feat_train obiljezja trening ID skupa
    # ID_feat_test obiljezja testnog ID skupa 
    # OoD_feat_train obiljezja OOD trening skupa, mozda bi trebalo biti konzistentno i koristiti OoD_feat_test
    # ID_tags_train labele za trening ID skup
    ID_predictions, ID_distances = NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
    OOD_predictions, OOD_distances = NCM(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_classes, number_features)
    test_distances = np.concatenate((ID_distances, OOD_distances))
    threshold_fpr80, threshold_fpr95 = find_threshold(test_Y, test_distances, 1, drop = False)
    threshold = threshold_fpr95
    print(f"NCM: Find distances with the thresold value {threshold}.")
    ID_predictions, ID_distances = NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
    OOD_predictions, OOD_distances = NCM(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_classes, number_features)
    pred=np.concatenate((ID_predictions, OOD_predictions))
    cm = confusion_matrix(test_Y, pred, labels = [0, 1])
    TPR = cm[1][1]/(cm[1][1]+cm[1][0])
    FPR = cm[0][1]/(cm[0][1]+cm[0][0])
    print("NCM: Confusion matrix\n",cm)
    print(f'NCM: TPR is {TPR:.4f}, FPR is {FPR:.4f}.')
    auroc = roc_auc_score(test_Y, test_distances)
    print(f'NCM: Area under ROC is {auroc}.')
    apr = average_precision_score(test_Y, test_distances)
    print(f'NCM: Average precision score is {apr}.')
    precision, recall, thrs = precision_recall_curve(test_Y, test_distances)
    aupr = auc(recall, precision)
    print(f'NCM: AUPR is {aupr}.')
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"NCM Execution time: {execution_time:.4f} seconds")

    print("KNN: ")
    start_time = time.time() 
    number_neighbors = 5 # po cemu se postavlja
    threshold = 0
    ID_Y = np.ones((len(ID_prob_test)))
    OoD_Y = np.zeros((len(OoD_prob_train)))
    test_Y = np.concatenate((ID_Y, OoD_Y))

    # ID_feat_train obiljezja trening ID skupa
    # ID_feat_test obiljezja testnog ID skupa 
    # OoD_feat_train obiljezja OOD trening skupa, mozda bi trebalo biti konzistentno i koristiti OoD_feat_test
    # ID_tags_train labele za trening ID skup
    ID_predictions, ID_distances = KNN(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
    OOD_predictions, OOD_distances = KNN(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_neighbors)
    test_distances = np.concatenate((ID_distances, OOD_distances))
    threshold_fpr80, threshold_fpr95 = find_threshold(test_Y, test_distances, 1, drop = False)
    threshold = threshold_fpr95
    print(f"KNN: Find distances with the thresold value {threshold}.")
    ID_predictions, ID_distances = KNN(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
    OOD_predictions, OOD_distances = KNN(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_neighbors)
    pred=np.concatenate((ID_predictions, OOD_predictions))
    cm = confusion_matrix(test_Y, pred, labels = [0, 1])
    TPR = cm[1][1]/(cm[1][1]+cm[1][0])
    FPR = cm[0][1]/(cm[0][1]+cm[0][0])
    print("KNN: Confusion matrix\n",cm)
    print(f'KNN: TPR is {TPR:.4f}, FPR is {FPR:.4f}.')
    auroc = roc_auc_score(test_Y, test_distances)
    print(f'KNN: Area under ROC is {auroc}.')
    apr = average_precision_score(test_Y, test_distances)
    print(f'KNN: Average precision score is {apr}.')
    precision, recall, thrs = precision_recall_curve(test_Y, test_distances)
    aupr = auc(recall, precision)
    print(f'KNN: AUPR is {aupr}.')
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"KNN Execution time: {execution_time:.4f} seconds")

    print("NNDR: ")
    start_time = time.time() 
    number_neighbors = 1000 # po cemu se postavlja
    threshold = 0
    ID_Y = np.ones((len(ID_prob_test)))
    OoD_Y = np.zeros((len(OoD_prob_train)))
    test_Y = np.concatenate((ID_Y, OoD_Y))

    # ID_feat_train obiljezja trening ID skupa
    # ID_feat_test obiljezja testnog ID skupa 
    # OoD_feat_train obiljezja OOD trening skupa, mozda bi trebalo biti konzistentno i koristiti OoD_feat_test
    # ID_tags_train labele za trening ID skup
    ID_predictions, ID_distances = NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
    OOD_predictions, OOD_distances = NNDR(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_neighbors)
    test_distances = np.concatenate((ID_distances, OOD_distances))
    threshold_fpr80, threshold_fpr95 = find_threshold(test_Y, test_distances, 1, drop = False)
    threshold = threshold_fpr95
    print(f"NNDR: Find distances with the thresold value {threshold}.")
    ID_predictions, ID_distances = NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
    OOD_predictions, OOD_distances = NNDR(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_neighbors)
    pred=np.concatenate((ID_predictions, OOD_predictions))
    cm = confusion_matrix(test_Y, pred, labels = [0, 1])
    TPR = cm[1][1]/(cm[1][1]+cm[1][0])
    FPR = cm[0][1]/(cm[0][1]+cm[0][0])
    print("NNDR: Confusion matrix\n",cm)
    print(f'NNDR: TPR is {TPR:.4f}, FPR is {FPR:.4f}.')
    auroc = roc_auc_score(test_Y, test_distances)
    print(f'NNDR: Area under ROC is {auroc}.')
    apr = average_precision_score(test_Y, test_distances)
    print(f'NNDR: Average precision score is {apr}.')
    precision, recall, thrs = precision_recall_curve(test_Y, test_distances)
    aupr = auc(recall, precision)
    print(f'NNDR: AUPR is {aupr}.')
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"NNDR Execution time: {execution_time:.4f} seconds")

    print("MD: ")
    start_time = time.time() 
    threshold = 0
    number_classes = 10 # postaviti po obiljezjima
    number_features = 512 # takodje kao i gore 
    ID_Y = np.ones((len(ID_prob_test)))
    OoD_Y = np.zeros((len(OoD_prob_train)))
    test_Y = np.concatenate((ID_Y, OoD_Y))

    # ID_feat_train obiljezja trening ID skupa
    # ID_feat_test obiljezja testnog ID skupa 
    # OoD_feat_train obiljezja OOD trening skupa, mozda bi trebalo biti konzistentno i koristiti OoD_feat_test
    # ID_tags_train labele za trening ID skup
    ID_predictions, ID_distances = MD(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
    OOD_predictions, OOD_distances = MD(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_classes, number_features)
    test_distances = np.concatenate((ID_distances, OOD_distances))
    threshold_fpr80, threshold_fpr95 = find_threshold(test_Y, test_distances, 1, drop = False)
    threshold = threshold_fpr95
    print(f"MD: Find distances with the thresold value {threshold}.")
    ID_predictions, ID_distances = MD(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
    OOD_predictions, OOD_distances = MD(ID_feat_train, ID_tags_train, OoD_feat_train, threshold, number_classes, number_features)
    pred=np.concatenate((ID_predictions, OOD_predictions))
    cm = confusion_matrix(test_Y, pred, labels = [0, 1])
    TPR = cm[1][1]/(cm[1][1]+cm[1][0])
    FPR = cm[0][1]/(cm[0][1]+cm[0][0])
    print("MD: Confusion matrix\n",cm)
    print(f'MD: TPR is {TPR:.4f}, FPR is {FPR:.4f}.')
    auroc = roc_auc_score(test_Y, test_distances)
    print(f'MD: Area under ROC is {auroc}.')
    apr = average_precision_score(test_Y, test_distances)
    print(f'MD: Average precision score is {apr}.')
    precision, recall, thrs = precision_recall_curve(test_Y, test_distances)
    aupr = auc(recall, precision)
    print(f'MD: AUPR is {aupr}.')
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"MD Execution time: {execution_time:.4f} seconds")

    return None

def metrics(labels, predictions, distances):
    '''
    labels:         labels of test samples
    predictions:    predictions of test samples
    distances:      
    '''
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    TPR = cm[1][1] / (cm[1][1] + cm[1][0])
    FPR = cm[0][1] / (cm[0][1] + cm[0][0])
    auroc = roc_auc_score(labels, distances)
    apr = average_precision_score(labels, distances)
    precision, recall, _ = precision_recall_curve(labels, distances)
    aupr = auc(recall, precision)

    return auroc, aupr, FPR

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
    # methods = ['MSP']
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
                ID_labels   = np.ones((len(ID_prob_test)))
                OOD_labels  = np.zeros((len(OOD_prob_test)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _ , ID_distances = MSP(ID_prob_test, threshold)
                _, OOD_distances = MSP(OOD_prob_test, threshold)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_tpr95 = find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = MSP(ID_prob_test, threshold)
                OOD_predictions, _ = MSP(OOD_prob_test, threshold)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 


            elif methods[j] == 'NCM':
                # All other methods use features
                number_classes = ID_prob_train.shape[1]
                number_features = ID_feat_train.shape[1]
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
                _, OOD_distances = NCM(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes, number_features)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_fpr95 = find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_fpr95
                ID_predictions, _  = NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
                OOD_predictions, _ = NCM(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes, number_features)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, fpr95 = metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 


            elif methods[j] == 'KNN':
                number_neighbors = 5 
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances = KNN(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                _, OOD_distances = KNN(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_fpr95 = find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_fpr95
                ID_predictions, _  = KNN(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                OOD_predictions, _ = KNN(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                auroc, aupr, fpr95 = metrics(test_labels, test_predictions, test_distances)
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 


            elif methods[j] == 'NNDR':
                number_neighbors = 1000 
                ID_labels = np.ones((len(ID_prob_test)))
                OOD_labels = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                threshold = 0

                _, ID_distances  = NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                _, OOD_distances = NNDR(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_fpr95 = find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_fpr95
                ID_predictions, _  = NNDR(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_neighbors)
                OOD_predictions, _ = NNDR(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_neighbors)
                auroc, aupr, fpr95 = metrics(test_labels, test_predictions, test_distances)
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

                _, ID_distances  = MD(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
                _, OOD_distances = MD(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes, number_features)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                _, threshold_fpr95 = find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_fpr95
                ID_predictions, _  = MD(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes, number_features)
                OOD_predictions, _ = MD(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes, number_features)
                auroc, aupr, fpr95 = metrics(test_labels, test_predictions, test_distances)
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
    training('SVHN', 'conv', fullTrain=True)
    # feature_extraction_spike('KMNIST')
    # feature_extraction_conv('FMNIST')
    # test_OOD_ID(case = '02', nameID='MNIST-on_mnist', nameOoD='KMNIST-on_mnist')

    # stats = test_with_output(case='03', nameID='MNIST')

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
'''

'''
all convResnet9 grayscale are trained


'''