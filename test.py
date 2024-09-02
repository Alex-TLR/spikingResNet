import numpy as np
import time 
from utils.Utils import Utils
from metrics.Metrics import Metrics


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
    elif nameID == 'CIFAR10':
        namesOOD = ['SVHN']
        suffixID = '-on_cifar10'

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
