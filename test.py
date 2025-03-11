import numpy as np
import time 
from utils.Utils import Utils
from metrics.Metrics import Metrics
from models.spikeresnet import spikeConvNN1, spikeConvNN2, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet10ModelAlt, SpikeResNet18Model  
from models.plain import spikeLinearNet1
import torch 
import snntorch.functional as SF 
import matplotlib.pyplot as plt

def test_accuracy(dataSet, modelType, batchSize, numberOfClasses, ResNetModel):
    '''
    check accuracy of trained model on ID test data
    '''

    # Load database
    # print(f"dataSet: {dataSet}")
    dataset_train, dataset_test = Utils.load_data(dataSet)
    testSize = len(dataset_test)
    # print(f"Number of test set images is: {testSize}")

    # Get image size
    channels, rows, cols = Utils.get_image_size(dataset_train, dataSet)

    # Load data
    train_loader, test_loader = Utils.data_loader(dataset_train, dataset_test, batchSize, dataSet, True)

    # Get the device
    device = Utils.get_device()
    # device = torch.device("cpu")

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
        elif ResNetModel == 2:
            model = spikeConvNN2(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 9:
            model = SpikeResNet9Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 10:
            model = SpikeResNet10Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 11:
            model = SpikeResNet10ModelAlt(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        elif ResNetModel == 18:
            model = SpikeResNet18Model(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        if ResNetModel == 21:
            model = spikeLinearNet1(numberOfChannels=channels, numberOfClasses=numberOfClasses, beta=beta, threshold=threshold)
        else:
            print("Not defined")
            return -1
        
        # Load weights
        torch.cuda.empty_cache()
        weightsName = 'weights/spike/' + 'resnet' + str(ResNetModel) + '_weights_' + dataSet + '.pth'
        # print(f"weightsName: {weightsName}")
        model.load_state_dict(torch.load(weightsName, weights_only=True))
        # file = torch.load(weightsName)
        # model.load_state_dict(file["model"])
        model = model.to(device)

        # Loss function
        loss_fn = SF.ce_rate_loss()

        # print("Check test images.")
        testAcc = np.zeros((1,testSize))
        testLen = 0
        testAccList = []
        testLoss = []
        i = 0
        print(f"Number of batches: {len(test_loader)}")
        model.eval()
        with torch.no_grad():
            for batch, labels in test_loader:
                testLen += len(batch)
                batch = batch.to(device)
                la = labels[0].item()
                labels = labels.to(device)
                # Generate predictions/ forward pass
                spikes, _, _ = model(batch, numberOfSteps)
                l = loss_fn(spikes, labels)
                testLoss.append(l.item())
                a1, a2 = model.accuracy_spike(model, numberOfSteps, batch, labels, device)
                # print(f"testLen: {testLen}, a1: {a1}, {a1.item()}, a2: {a2}, {a2.item()}")
                testAcc[0, i] = a2.item()
                testAccList.append(a1.item())
                del batch, labels
                # print(f"\rtestAcc[{i}]: {testAcc[0, i]}, label: {la}, count: {np.sum(testAcc[0, :])/(i+1)}, progress: {i}/{len(test_loader)}", end='', flush=True)
                i += 1
        # Test stats
        meanA1 = np.sum(testAcc) / testSize
        meanA2 = sum(testAccList) / len(testAccList)
        meanL = sum(testLoss) / len(testLoss)
        print(f'Test loss is {meanL:.2f}. Test accuracy1 is {meanA1*100:.2f}. Test accuracy2 is {meanA2*100:.2f}.')

        return None



# TODO make time consumption analysis
def test_with_output(case, nameID):
    '''
    case:       for example case 01 is '01'
    nameID:     name of the In Distribution features, example 'MNIST'
    '''
    if nameID == 'MNIST':
        namesOOD = ['FMNIST', 'KMNIST', 'Letters']
        # namesOOD = ['FMNIST', 'KMNIST']
        suffixID = '-on_mnist'
    elif nameID == 'FMNIST':
        namesOOD = ['MNIST', 'KMNIST', 'Letters']
        suffixID = '-on_fmnist'
    elif nameID == 'KMNIST':
        namesOOD = ['MNIST', 'FMNIST', 'Letters']
        suffixID = '-on_kmnist'
    elif nameID == 'CIFAR10':
        namesOOD = ['SVHN', 'Food101']
        suffixID = '-on_cifar10'
    elif nameID == 'SVHN':
        namesOOD = ['CIFAR10', 'Food101']
        suffixID = '-on_svhn'

    # methods = ['MSP', 'NCM', 'KNN', 'NNDR', 'MD']
    methods = ['MSP', 'SD', 'NCM']
    stats = np.zeros((len(namesOOD), len(methods)*3), dtype=np.float64)
    IDpath = 'features/spike/case_' + case + '/' + nameID + suffixID + '.npz'

    ID = np.load(IDpath)
    print(f"\nIDpath: {IDpath}")
    ID_spik_train = ID['arr0']  # In-Distribution training set spikes
    ID_feat_train = ID['arr1']  # In-Distribution training set features
    ID_prob_train = ID['arr2']  # In-Distribution training set outputs (usually with no softmax applied)
    ID_tags_train = ID['arr3']  # In-Distribution training set labels
    ID_spik_test  = ID['arr7']  # In-Distribution test set spikes
    ID_feat_test  = ID['arr4']  # In-Distribution test set features
    ID_prob_test  = ID['arr5']  # In-Distribution test set outputs (usually with no softmax applied)
    # ID_tags_test  = ID['arr6']  # In-Distribution test set labels

    for i in range(len(namesOOD)):
        OODpath = 'features/spike/case_' + case + '/' + namesOOD[i] + suffixID + '.npz'
        print(f"OODpath: {OODpath}")
        OOD = np.load(OODpath)
        OOD_spik_train = OOD['arr0']  # Out-of-Distribution training set spikes
        OOD_feat_train = OOD['arr1']  # Out-of-Distribution training set features
        OOD_prob_train = OOD['arr2']  # Out-of-Distribution training set outputs (usually with no softmax applied)
        OOD_tags_train = OOD['arr3']  # Out-of-Distribution training set labels
        OOD_spik_test  = OOD['arr7']  # Out-of-Distribution test set spikes
        OOD_feat_test  = OOD['arr4']  # Out-of-Distribution test set features
        OOD_prob_test  = OOD['arr5']  # Out-of-Distribution test set outputs (usually with no softmax applied)
        OOD_tags_test  = OOD['arr6']  # Out-of-Distribution test set labels

        # print(f"OOD_prob_train.shape: {OOD_prob_train.shape}, OOD_prob_test.shape: {OOD_prob_test.shape}")
        # print(f"ID_feat_train.shape: {ID_feat_train.shape}, ID_feat_test.shape: {ID_feat_test.shape}, OOD_feat_train.shape: {OOD_feat_train.shape}")

        # Methodology includes the IN/OOD classification where ID are positive samples taken from ID_test_set
        # and OOD are negative samples taken from OOD_train_set (or maybe OOD_train_set + OOD_test_set)

        for j in range(len(methods)):

            # iterate trought each of classification methods and calculate the metrics
            # This is the baseline method that uses the outputs of the network number_of_classes-D 
            if methods[j] == 'MSP':
                print("MSP")
                start_time = time.time() 
                ID_labels   = np.ones((len(ID_prob_test)))
                OOD_labels  = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                # print(f"test_labels: {test_labels.shape}")
                threshold = 0

                # Calculates the distances between row-wise max probabilities and 0,
                # only to check the acctual max of probabilities (predictions are not
                # important in this step). 
                _ , ID_distances = Metrics.MSP(ID_prob_test, threshold)
                _, OOD_distances = Metrics.MSP(OOD_prob_train, threshold)
                # OOD_prob = np.concatenate((OOD_prob_train, OOD_prob_test), axis=0)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                # plot test_label and test_distances

                # plt.figure(figsize=(10, 6))
                # plt.plot(test_distances[0:20000], alpha=0.5, label="Distances")
                # plt.plot(test_labels[0:20000], linewidth=2, label="Labels")
                # plt.title("MSP")
                # plt.xlabel("Sample")
                # plt.ylabel("Distance")
                # plt.grid(True)
                # plt.legend()
                # plt.show()


                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.MSP(ID_prob_test, threshold)
                OOD_predictions, _ = Metrics.MSP(OOD_prob_train, threshold)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"MSP Execution time: {execution_time:.4f} seconds")

            elif methods[j] == 'SD':
                print("Spike distance")
                print("Keep output spike pattern, numOfClasses-D")
                start_time = time.time() 
                number_classes = ID_prob_train.shape[1]
                ID_labels   = np.ones((len(ID_prob_test)))
                OOD_labels  = np.zeros((len(OOD_prob_train)))
                test_labels = np.concatenate((ID_labels, OOD_labels))
                # print(f"test_labels: {test_labels.shape}")
                threshold = 0

                # Calculates the distances between row-wise max probabilities and 0,
                # only to check the acctual max of probabilities (predictions are not
                # important in this step). 
                _ , ID_distances = Metrics.SD(ID_spik_train, ID_tags_train, ID_spik_test, threshold, number_classes)
                _, OOD_distances = Metrics.SD(ID_spik_train, ID_tags_train, OOD_spik_train, threshold, number_classes)
                # OOD_prob = np.concatenate((OOD_prob_train, OOD_prob_test), axis=0)
                test_distances = np.concatenate((ID_distances, OOD_distances))
                # plot test_label and test_distances

                # plt.figure(figsize=(10, 6))
                # plt.plot(test_distances[0:20000], alpha=0.5, label="Distances")
                # plt.plot(test_labels[0:20000], linewidth=2, label="Labels")
                # plt.title("MSP")
                # plt.xlabel("Sample")
                # plt.ylabel("Distance")
                # plt.grid(True)
                # plt.legend()
                # plt.show()


                _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
                threshold = threshold_tpr95
                ID_predictions, _  = Metrics.SD(ID_spik_train, ID_tags_train, ID_spik_test, threshold, number_classes)
                OOD_predictions, _ = Metrics.SD(ID_spik_train, ID_tags_train, OOD_spik_train, threshold, number_classes)
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"SD (spike distance) Execution time: {execution_time:.4f} seconds")

            elif methods[j] == 'NCM':
                print("NCM")
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
                # plt.figure(figsize=(10, 6))
                # plt.plot(test_distances[0:20000], alpha=0.5, label="Distances")
                # plt.plot(test_labels[0:20000], linewidth=2, label="Labels")
                # plt.title("NCM")
                # plt.xlabel("Sample")
                # plt.ylabel("Distance")
                # plt.grid(True)
                # plt.legend()
                # plt.show()

                ID_predictions, _  = Metrics.NCM(ID_feat_train, ID_tags_train, ID_feat_test, threshold, number_classes)
                OOD_predictions, _ = Metrics.NCM(ID_feat_train, ID_tags_train, OOD_feat_train, threshold, number_classes)
                # ID_predictions, _  = Metrics.NCM(ID_feat_train, ID_labels, ID_feat_test, threshold, number_classes)
                # OOD_predictions, _ = Metrics.NCM(ID_feat_train, OOD_labels, OOD_feat_train, threshold, number_classes)
                # print(f"ID_predictions: {ID_predictions[0:100]}")
                # print(f"OOD_predictions: {OOD_predictions[0:100]}")
                test_predictions = np.concatenate((ID_predictions, OOD_predictions))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"NCM Execution time: {execution_time:.4f} seconds")


            elif methods[j] == 'KNN':
                print("KNN\n")
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
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
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
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"NNDR Execution time: {execution_time:.4f} seconds")


            elif methods[j] == 'MD':
                start_time = time.time()
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
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                print(f"threshold_tpr95: {threshold_tpr95}")
                print(f"True positive rate: {tpr95}, False positive rate {fpr95}")
                stats[i, j] = auroc 
                stats[i, len(methods) + j] = aupr
                stats[i, len(methods)*2 + j] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"MD Execution time: {execution_time:.4f} seconds")

            else:
                pass

        del OOD, OOD_feat_train, OOD_prob_train, OOD_prob_test
        print()

    return stats
