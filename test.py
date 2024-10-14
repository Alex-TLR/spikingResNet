import numpy as np
import time 
from utils.Utils import Utils
from metrics.Metrics import Metrics
from models.spikeresnet import spikeConvNN1, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model  
import torch 
import snntorch.functional as SF 

def test_accuracy(dataSet, modelType, batchSize, numberOfClasses, ResNetModel):
    '''
    check accuracy of trained model on ID test data
    '''

    # Load database
    dataset_train, dataset_test = Utils.load_data(dataSet)

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)

    # Load data
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)

    # Get the device
    device = Utils.get_device()

    if modelType == 'conv':
        pass

    elif modelType == 'spike':

        # For spiking neural network we need number of steps
        numberOfSteps = 50
        beta = 0.95
        threshold = 0.25

                # Define model
        if ResNetModel == 1:
            model = spikeConvNN1(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 9:
            model = SpikeResNet9Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 18:
            model = SpikeResNet18Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        else:
            print("Not defined")
            return -1
        # Load weights
        weightsName = 'weights/spike/' + 'resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
        model.load_state_dict(torch.load(weightsName, weights_only=True))
        model = model.to(device)

        # Loss function
        loss_fn = SF.ce_rate_loss()

        print("Check test images.")
        testAcc = []
        testLoss = []
        for batch, labels in test_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            # Generate predictions/ forward pass
            spikes, _, _ = model(batch, numberOfSteps)
            l = loss_fn(spikes, labels)
            testLoss.append(l.item())
            a = model.accuracy_spike(model, numberOfSteps, batch, labels, device)
            testAcc.append(a.item())
            del batch, labels
        # Test stats
        meanA = sum(testAcc) / len(testAcc)
        meanL = sum(testLoss) / len(testLoss)
        print(f'Test loss is {meanL:.2f}. Test accuracy is {meanA*100:.2f}.')

        return None



# TODO make time consumption analysis
def test_with_output(case, nameID):
    '''
    case:       for example case 01 is '01'
    nameID:     name of the In Distribution features, example 'MNIST'
    '''
    if nameID == 'MNIST':
        namesOOD = ['FMNIST', 'KMNIST', 'EMNIST', 'Letters']
        suffixID = '-on_mnist'
    elif nameID == 'FMNIST':
        namesOOD = ['MNIST', 'KMNIST', 'EMNIST', 'Letters']
        suffixID = '-on_fmnist'
    elif nameID == 'KMNIST':
        namesOOD = ['MNIST', 'FMNIST', 'EMNIST', 'Letters']
        suffixID = '-on_kmnist'
    elif nameID == 'CIFAR10':
        namesOOD = ['SVHN']
        suffixID = '-on_cifar10'
    elif nameID == 'SVHN':
        namesOOD = ['CIFAR10']
        suffixID = '-on_svhn'

    methods = ['MSP', 'NCM', 'KNN', 'NNDR', 'MD']
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
                start_time = time.time()
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
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"NCM Execution time: {execution_time:.4f} seconds")


            elif methods[j] == 'KNN':
                start_time = time.time()
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
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"KNN Execution time: {execution_time:.4f} seconds")


            elif methods[j] == 'NNDR':
                start_time = time.time()
                number_neighbors = 10000 
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
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"NNDR Execution time: {execution_time:.4f} seconds")


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
