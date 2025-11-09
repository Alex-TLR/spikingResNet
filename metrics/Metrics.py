import gc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestCentroid
import torch
from feature import feature_extraction_spike
from utils.Utils import Utils, distances_from_average_clusters
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import k_means, DBSCAN, KMeans
import time 
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
from train import training_population
import gc
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model
from models.plain import spikeLinearNet1

class Metrics():

    def __init__(self, name):
        pass

    @staticmethod
    def metrics(labels, predictions, distances):
        '''
        Calculates metrics for ID/OOD prediction

        Inputs:
            labels:         labels of test samples
            predictions:    predictions of test samples
            distances:    

        Outputs:
            auroc:          area under ROC
            aupr:           area under PR curve
            FPR:            false positive rate  
        '''

        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        # print(f"Confusion matrix:\n{cm}")
        TPR = cm[1][1] / (cm[1][1] + cm[1][0])
        FPR = cm[0][1] / (cm[0][1] + cm[0][0])
        # print(f"True positive rate: {TPR}")
        # print(f"False positive rate: {FPR}")
        auroc = roc_auc_score(labels, distances)
        apr = average_precision_score(labels, distances)
        precision, recall, _ = precision_recall_curve(labels, distances)
        aupr = auc(recall, precision)

        return auroc, aupr, TPR, FPR


    @staticmethod
    def softmax(x):
        '''
        Calculates softmax on input features.
        Features are stored in matrix row-wise.
        Max value per row is subrtracted for numerical stability.
        '''
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    @staticmethod
    def MSP(ID_feat_test, OOD_feat_test, threshold):
        '''
        Calculates the Maximum Softmax Probability (MSP) 
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        # Calculates softmax probabilities
        ID_distances = np.max(Metrics.softmax(ID_feat_test), axis=1)
        OOD_distances = np.max(Metrics.softmax(OOD_feat_test), axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  
        execution_time = end_time - start_time 
        print(f"MSP Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def MLS(ID_feat_test, OOD_feat_test, threshold):
        '''
        Calculates the Maximum Logit Score (MLS)
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        ID_distances = np.max(ID_feat_test, axis=1)
        OOD_distances = np.max(OOD_feat_test, axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances < threshold).astype(np.int32)
        OOD_predictions = (OOD_distances < threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  
        execution_time = end_time - start_time 
        print(f"MLS Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def ENGY(ID_feat_test, OOD_feat_test, T=1.0):
        '''
        Energy-based OOD detection method.
        Low energy of input logit means ID sample, while high energy indicates OOD sample.
        Energy is calculated as: E(x) = -T * log(sum(exp(f_i(x)/T))), where f_i(x) is the i-th logit of input x,
        and T is the temperature scaling parameter.

        Additional minus sign is added, so that higher energy means more likely ID sample.

        Inputs:
            test_data:      Matrix of input row-wise features
            threshold:      Threshold value
            T:              Temperature scaling parameter

        Outputs:
            predictions:    Array of predicted labels
            energies:       Array of energy values
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        ID_distances = (-T * torch.logsumexp(torch.Tensor(ID_feat_test) / T, dim=1)).numpy()
        OOD_distances = (-T * torch.logsumexp(torch.Tensor(OOD_feat_test) / T, dim=1)).numpy()
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        # Make predictions based on the distances
        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"ENERGY Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances

        # e = -(-T * torch.logsumexp(torch.Tensor(ID_feat_test) / T, dim=1)).numpy()
        # predictions = (e > threshold).astype(np.int32)
        # return predictions, e
    
    @staticmethod
    def ODIN(ID_feat_test, OOD_feat_test, eps=0.0014, T=1000.0):
        """
        Implements the ODIN algorithm for OoD detection without requiring original input data.
        Inputs:
            ID_feat_test:   Matrix of in-distribution test features (logits)
            OOD_feat_test:  Matrix of out-of-distribution test features (logits)
            eps:            Small perturbation value
            T:              Temperature scaling parameter
        """

        # # Combine in-distribution and out-of-distribution features
        # combined_features = np.concatenate((ID_features_test, OOD_features_test), axis=0)
        # combined_penultimate = np.concatenate((ID_penultimate_test, OOD_penultimate_test), axis=0)

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        ID_distances = np.max(np.exp(ID_feat_test / T) / np.sum(np.exp(ID_feat_test / T), axis=1, keepdims=True),axis=1)
        OOD_distances = np.max(np.exp(OOD_feat_test / T) / np.sum(np.exp(OOD_feat_test / T), axis=1, keepdims=True),axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        # Make predictions based on the distances
        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"ODIN Execution time: {execution_time:.4f} seconds.")

        # Return labels, predictions, and ODIN scores
        return test_labels, test_predictions, test_distances


    @staticmethod
    def VIM(ID_penultimate_test, ID_features_test, OOD_penultimate_test, OOD_features_test, mu, null_space_eigvecs, alpha=0.1):
        '''
        VIM method for OOD detection.
        
        Needs revision
        Inputs:
            test_data:      Matrix of input row-wise features (num_classes)
            test_features:  Matrix of input row-wise features (penultimate layer)
            mu:             Mean of the training features
            null_space_eigvecs: Null space eigenvectors
            threshold:      Threshold value
            alpha:         Regularization parameter

        Outputs:
            predictions:    Array of predicted labels
            max_probs:      Array of output features, 1-D
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_features_test)))
        OOD_labels = np.zeros((len(OOD_features_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        ID_residual = (ID_penultimate_test - mu) @ null_space_eigvecs
        r_norm = np.linalg.norm(ID_residual, axis=1)
        logit_norm = np.linalg.norm(ID_features_test, axis=1)
        ID_distances = -(logit_norm - alpha * r_norm)
        OOD_residual = (OOD_penultimate_test - mu) @ null_space_eigvecs
        r_norm = np.linalg.norm(OOD_residual, axis=1)
        logit_norm = np.linalg.norm(OOD_features_test, axis=1)
        OOD_distances = -(logit_norm - alpha * r_norm)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances < threshold).astype(np.int32)
        OOD_predictions = (OOD_distances < threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  
        execution_time = end_time - start_time 
        print(f"VIM Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def ASH(ID_feat_test, OOD_feat_test, keep_ratio=0.25):
        """
        This simplified ASH implementation can work with features from either the penultimate layer or the final layer, depending on the design of your model and the nature of the features.

        Key Considerations:
        Penultimate Layer Features:

        These features are typically high-dimensional and contain rich information about the input data.
        They are often used for OoD detection because they capture more general patterns and are less specialized for classification compared to the final layer.
        Final Layer Features:

        These features (e.g., logits or softmax probabilities) are lower-dimensional and more directly tied to the classification task.
        Using the final layer features may simplify the computation but could reduce the effectiveness of OoD detection, as they are more specialized for the in-distribution classes.
        Applicability:
        Penultimate Layer: The current implementation is designed to operate on high-dimensional features, making it well-suited for penultimate layer features.
        Final Layer: If you want to use the final layer features, you can directly pass them to the function. However, you may need to adjust the keep_ratio parameter, as the dimensionality of the final layer is typically much lower than the penultimate layer.
        Recommendation:
        If your goal is to maximize OoD detection performance, use penultimate layer features.
        If computational efficiency or simplicity is more important, you can use final layer features.

        """


        """
        Simplified ASH algorithm for OoD detection.

        Parameters:
        - ID_penultimate_test: Penultimate layer features of in-distribution test set.
        - OOD_penultimate_test: Penultimate layer features of out-of-distribution test set.
        - keep_ratio: Ratio of top activations to retain (default: 0.25).

        Returns:
        - test_labels: Ground truth labels for in-distribution (1) and out-of-distribution (0).
        - test_predictions: Predictions based on ASH scores.
        - test_distances: ASH scores for each sample.
        """

        start_time = time.time()
        # ID_labels = np.ones((len(ID_feat_test)))
        # OOD_labels = np.zeros((len(OOD_feat_test)))
        # test_labels = np.concatenate((ID_labels, OOD_labels))

        start_time = time.time()

        # Process ID features
        k = int(ID_feat_test.shape[1] * keep_ratio)
        ID_top_k_threshold = np.partition(ID_feat_test, -k, axis=1)[:, -k]
        ID_suppressed_features = np.where(ID_feat_test >= ID_top_k_threshold[:, None], ID_feat_test, 0)
        ID_suppressed_features[ID_suppressed_features.sum(axis=1) == 0] = 1e-12
        ID_norm_factor = ID_feat_test.sum(axis=1, keepdims=True) / (ID_suppressed_features.sum(axis=1, keepdims=True) + 1e-12)
        ID_norm_factor[np.isnan(ID_norm_factor)] = 1.0
        ID_normalized_features = ID_suppressed_features * ID_norm_factor
        ID_normalized_features = np.nan_to_num(ID_normalized_features, nan=1e-12, posinf=1e12, neginf=1e-12)
        ID_ash_scores = -np.log(np.sum(ID_normalized_features, axis=1) + 1e-12)

        # Process OOD features
        k = int(OOD_feat_test.shape[1] * keep_ratio)
        OOD_top_k_threshold = np.partition(OOD_feat_test, -k, axis=1)[:, -k]
        OOD_suppressed_features = np.where(OOD_feat_test >= OOD_top_k_threshold[:, None], OOD_feat_test, 0)
        OOD_suppressed_features[OOD_suppressed_features.sum(axis=1) == 0] = 1e-12
        OOD_norm_factor = OOD_feat_test.sum(axis=1, keepdims=True) / (OOD_suppressed_features.sum(axis=1, keepdims=True) + 1e-12)
        OOD_norm_factor[np.isnan(OOD_norm_factor)] = 1.0
        OOD_normalized_features = OOD_suppressed_features * OOD_norm_factor
        OOD_ash_scores = -np.log(np.sum(OOD_normalized_features, axis=1) + 1e-12)

        # Combine scores and labels
        test_labels = np.concatenate((np.ones(len(ID_feat_test)), np.zeros(len(OOD_feat_test))))
        test_scores = np.concatenate((ID_ash_scores, OOD_ash_scores))

        # Generate predictions based on ASH scores
        threshold = np.percentile(ID_ash_scores, 95)  # 95th percentile of ID scores
        test_predictions = (test_scores >= threshold).astype(int)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"ASH Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_scores

    @staticmethod
    def NCM(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes):
        '''
        Nearest Class Mean is the distance between test feature and the 
        nearest training class centroid.  

        Inputs:
            ID_feat_train:  matrix with train features, it is used to 
                            calculate centroids
            ID_tags_train:  an array with train labels, gives the label 
                            of each training feature
            ID_feat_test:   matrix with ID test features
            OOD_feat_test:  matrix with OOD test features
            number_classes: (int) number of classes

        Outputs:
            test_labels:
            test_predictions:    an array of predicted labels
            test_distances:

        '''
 
        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        # Find centroids for each class
        centroids = np.array([ID_feat_train[ID_tags_train == c].mean(axis=0) for c in range(number_classes)])

        # Calculate distance between each sample and the class centroids
        ID_distances = Utils.distance_from_class_centroids(ID_feat_test, centroids)
        OOD_distances = Utils.distance_from_class_centroids(OOD_feat_test, centroids)

        # Concatenate distances
        test_distances = np.concatenate((ID_distances, OOD_distances))
        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95
        # print(f"threshold_tpr95: {threshold_tpr95}")

        # Make predictions based on the distances
        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"NCM Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def MD(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes, number_features):
        '''
        Mahalanobis distance of ID and OOD feature vectors, based on the centroids
        of the training data. Mahalanobis distance is calculated as the
        squared distance between the feature vector and the class centroid,
        normalized by the covariance matrix of the training data. The distance
        is calculated for each class, and the minimum distance is used for
        classification. If the minimum distance is larger than the threshold,
        the feature vector is classified as OOD, otherwise it is classified as ID.

        Inputs:
            ID_feat_train:  matrix with train features, it is used to 
                            normalize feature vectors
            ID_tags_train:  an array with train labels, gives the label 
                            of each training feature
            ID_feat_test:   matrix with ID test features
            OOD_feat_test:  matrix with OOD test features
            number_classes: (int) number of classes

        Outputs:
            test_labels:
            test_predictions:    an array of predicted labels
            test_distances:
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        # Find a mean feature vector for each class
        mean_vectors = np.stack([ID_feat_train[ID_tags_train == c].mean(axis=0) for c in range(number_classes)])
        #centroids = np.array([ID_feat_train[ID_tags_train == c].mean(axis=0) for c in range(number_classes)])

        # Center training data by class mean
        centered = ID_feat_train - mean_vectors[ID_tags_train]

        # Covariance matrix and regularization
        cov = np.cov(centered, rowvar=False, bias=False) + 0.001 * np.eye(number_features)
        cov_inv = np.linalg.inv(cov)

        # Compute all Mahalanobis distances in a vectorized way
        # MDK shape: (len(ID_feat_test), num_classes)
        ID_MDK = cdist(ID_feat_test, mean_vectors, metric='mahalanobis', VI=cov_inv) ** 2
        OOD_MDK = cdist(OOD_feat_test, mean_vectors, metric='mahalanobis', VI=cov_inv) ** 2

        ID_distances = -np.min(ID_MDK, axis=1)
        OOD_distances = -np.min(OOD_MDK, axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"MD Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def KNN(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_neighbors):
        '''
        k Nearest Neighbor classification

        Inputs:
            ID_feat_train:  matrix with train features, used to fit the model
            ID_tags_train:  an array with train labels, gives the label 
                            of each training feature
            ID_feat_test:   matrix with ID test features
            OOD_feat_test:  matrix with OOD test features
            number_neighbors: (int) number of neighbors

        Outputs:
            test_labels:
            test_predictions:    an array of predicted labels
            test_distances:
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        neigh = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'euclidean')
        neigh.fit(ID_feat_train, ID_tags_train)

        # Calculate distances
        ID_distances, _ = neigh.kneighbors(ID_feat_test, return_distance = True)
        ID_distances = -np.average(ID_distances, axis=1)
        OOD_distances, _ = neigh.kneighbors(OOD_feat_test, return_distance = True)
        OOD_distances = -np.average(OOD_distances, axis=1)

        test_distances = np.concatenate((ID_distances, OOD_distances))
        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"KNN Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def FKM(ID_feat_train, ID_feat_test, OOD_feat_test, number_clusters):
        '''
        K-MEANS Full Clustering is done on all feature vectors, regardless on their labels.

        Inputs:
            ID_feat_train:  matrix with train features, used to fit the model
            ID_tags_train:  an array with train labels, gives the label 
                            of each training feature
            ID_feat_test:   matrix with ID test features
            OOD_feat_test:  matrix with OOD test features
            number_clusters: (int) number of clusters

        Outputs:
            test_labels:
            test_predictions:    an array of predicted labels
            test_distances:
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        clusters = np.zeros((number_clusters, ID_feat_train.shape[1]))
        clusters = k_means(ID_feat_train, number_clusters)[0]

        ID_distances = cdist(ID_feat_test, clusters, metric='euclidean')
        ID_distances = -np.min(ID_distances, axis=1)
        OOD_distances = cdist(OOD_feat_test, clusters, metric='euclidean')
        OOD_distances = -np.min(OOD_distances, axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"FKM Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def CKM(ID_feat_train, ID_tags_train, ID_feat_test, OOD_feat_test, number_classes, number_clusters_per_class):
        '''
        K-MEANS Full Clustering is done for each class separately, and then
        the distances are calculated for each test feature vector.

        Inputs:
            ID_feat_train:  matrix with train features, used to fit the model
            ID_tags_train:  an array with train labels, gives the label 
                            of each training feature
            ID_feat_test:   matrix with ID test features
            OOD_feat_test:  matrix with OOD test features
            number_clusters_per_class: (int) number of clusters per class

        Outputs:
            test_labels:
            test_predictions:    an array of predicted labels
            test_distances:
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        clusters = np.zeros((number_classes * number_clusters_per_class, ID_feat_train.shape[1]))
        # Find clusters for each class
        for c in range(number_classes):
            clusters[c: c + number_clusters_per_class] = k_means(ID_feat_train[ID_tags_train == c], number_clusters_per_class)[0]

        ID_distances = cdist(ID_feat_test, clusters, metric='euclidean')
        ID_distances = -np.min(ID_distances, axis=1)
        OOD_distances = cdist(OOD_feat_test, clusters, metric='euclidean')
        OOD_distances = -np.min(OOD_distances, axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"FKM Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    
    @staticmethod
    def SD(train_X, train_Y, test_X, threshold, num_classes):
        '''
        Spike distance
        '''
        centroids_X = np.array([train_X[train_Y == c].mean(axis=0) for c in range(num_classes)])
        predictions = np.zeros(len(test_X))
        distances = np.zeros(len(test_X))
        p = 100
        epsilon = 1e-10

        for i in range(len(test_X)):
            dist = np.zeros(num_classes)
            for j in range(num_classes):
                # dist[j] = np.sqrt(np.sum((centroids_X[j] - test_X[i])**2)) # euclidian
                # dist[j] = np.sum(np.abs(centroids_X[j] - test_X[i])) # Manhattan
                # dist[j] = 1 - np.dot(centroids_X[j], test_X[i]) / (np.linalg.norm(centroids_X[j]) * np.linalg.norm(test_X[i])) # cosine
                dist[j] = np.power(np.sum(np.abs(centroids_X[j] - test_X[i])**p), 1/p) # minkovwski
                # dist[j] = np.max(np.abs(centroids_X[j] - test_X[i])) #cebisevljev
                # dist[j] = np.sum((centroids_X[j] + epsilon) * np.log((centroids_X[j] + epsilon) / (test_X[i] + epsilon))) # Kullback-Leibler
                # dist[j] = np.sum(np.abs(centroids_X[j] - test_X[i])) / np.sum(centroids_X[j] + test_X[i]) # Bray-Curtis Dissimilarity
            distances[i] = -dist.min()

        # print(f"distances: {distances.shape}")
        # distances = (distances - distances.min()) / (distances.max() - distances.min())
            
        for i in range(len(test_X)):    
            predictions[i] = 1 if distances[i] > threshold else 0

        return predictions, distances
    

    @staticmethod
    def NNDR(train_X, train_Y, test_X, threshold, number_neighbors):
        '''
        Nearest Neighbor distance ratio classification

        Inputs:
            train_X:        matrix with train features
            train_Y:        array with train labels
            test_X:         matrix with test features
            threshold:      threshold
            num_features:   feature dimension

        Outputs:
            predictions:    an array of predicted labels
            distances:      an array of output features, 1-D 
        '''

        # neigh = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'euclidean')
        # neigh.fit(train_X, train_Y)

        # # predictions = neigh.predict(test_X)
        # neigh_dist, neigh_ind = neigh.kneighbors(test_X, return_distance = True)
        # print(f"neigh_dist: {neigh_dist.shape}, neigh_ind: {neigh_ind.shape}")

        # predictions = np.full(test_X.shape[0], fill_value=-1)  # Default prediction as -1 (for rejected matches)
        # distances = np.zeros(test_X.shape[0])

        # # distances = []
        # # m = []
        # for i in range(len(neigh_dist)):
        #     differing_neighbor_index = None
        #     for j in range(len(neigh_dist[i])):
        #         if(train_Y[neigh_ind[i][j]] != train_Y[neigh_ind[i][0]]):
        #             differing_neighbor_index = j
        #             break

        # for i in range(test_X.shape[0]):
        #     if number_neighbors > 1 and neigh_dist[i, 1] > 0:  # Avoid division by zero
        #         ratio = neigh_dist[i, 0] / neigh_dist[i, 1]  # NNDR computation
        #         if ratio < threshold:  # Accept match if ratio is below threshold
        #             predictions[i] = train_Y[neigh_ind[i, 0]]  # Assign the nearest neighbor's label
        #     else:
        #         predictions[i] = train_Y[neigh_ind[i, 0]]  # If only 1 neighbor, assign it directly


        # # for i in range(len(neigh_dist)):
        # #     distances.append(-neigh_dist[i][0]/neigh_dist[i][m[i]])

        # if differing_neighbor_index is None:
        #     distances[i] = -10000  # No differing class found
        # else:
        #     distances[i] = -neigh_dist[i][0] / neigh_dist[i][differing_neighbor_index]

        # if(distances[i] > threshold):
        #     predictions[i] = 1
        # else:
        #     predictions[i] = 0

        # return predictions, distances

        br = 0
        distances = []
        m = []
        neigh = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'euclidean')
        neigh.fit(train_X, train_Y)

        prediction = neigh.predict(test_X)
        neigh_dist, neigh_ind = neigh.kneighbors(test_X, return_distance = True)
        print(type(neigh_dist[0][0]))
        for i in range(len(neigh_dist)):
            for j in range(len(neigh_dist[i])):
                if(train_Y[neigh_ind[i][j]] != train_Y[neigh_ind[i][0]]):
                    m.append(j)
                    break

        for i in range(len(neigh_dist)):
            if(neigh_dist[i][m[i]]==0):
                print("NaN")
                print(i)
                print(neigh_dist[i][0])
                print(neigh_dist[i])
                br=br+1
                neigh_dist[i][m[i]] = 1e-20
            distances.append(-neigh_dist[i][0]/neigh_dist[i][m[i]])

            if(distances[i] > threshold):
                prediction[i] = 1
            else:
                prediction[i] = 0
        print(br)
        return prediction, distances

    

def test_metrics(config, case, nameID, methods, features='spikes'):
    '''
    case:       for example case 01 is '01'
    nameID:     name of the In Distribution features, example 'MNIST'
    '''
    if nameID == 'MNIST':
        suffixID = '-on_mnist'
    elif nameID == 'FMNIST':  
        suffixID = '-on_fmnist'
    elif nameID == 'KMNIST':
        suffixID = '-on_kmnist'
    elif nameID == 'Letters':
        suffixID = '-on_letters'
    elif nameID == 'CIFAR10':
        suffixID = '-on_cifar10'
    elif nameID == 'SVHN':
        suffixID = '-on_svhn'
    else:
        raise ValueError("Unknown ID dataset name")

    # Find the names of OOD datasets
    namesOOD = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]

    method_to_idx = {m: idx for idx, m in enumerate(methods)}

    stats = np.zeros((len(namesOOD), len(methods)*3), dtype=np.float64)
    IDpath = 'features/spike/case_' + case + '/' + nameID + suffixID + '.npz'

    # # For ODIN and ASH, we need to load the model
    # channels = 3
    # beta = 0.95
    # threshold = 0.25

    # # Get image size based on dataset
    # if config.dataset_ID in ['CIFAR10', 'CIFAR100']:
    #     feature_size = 32
    # elif config.dataset_ID in ['MNIST', 'FMNIST', 'KMNIST']:
    #     feature_size = 28
    # else:
    #     feature_size = 28

    # if config.resnet_model == 1:
    #     model = spikeConvNN1(numberOfChannels=channels, 
    #                             numberOfClasses=config.num_classes, 
    #                             beta=beta, 
    #                             threshold=threshold)
    #     featSize = 256
    # elif config.resnet_model == 2:
    #     model = spikeConvNN2(numberOfChannels=channels, 
    #                             numberOfClasses=config.num_classes,
    #                             beta=beta, threshold=threshold, 
    #                             feature_size=feature_size, 
    #                             expansion=config.expansion)    
    #     featSize = 300
    # elif config.resnet_model == 4:
    #     model = spikeConvNN4(numberOfChannels=channels, 
    #                             numberOfClasses=config.num_classes, 
    #                             beta=beta, threshold=threshold, 
    #                             feature_size=feature_size, 
    #                             numberOfSteps=config.num_time_steps_extract, 
    #                             expansion=config.expansion)
    #     featSize = 256
    # elif config.resnet_model == 9:
    #     model = SpikeResNet9Model(numberOfChannels=channels, 
    #                                 numberOfClasses=config.num_classes, 
    #                                 beta=beta, 
    #                                 threshold=threshold)
    #     featSize = 512
    # elif config.resnet_model == 10:
    #     model = SpikeResNet10Model(numberOfChannels=channels, 
    #                                 numberOfClasses=config.num_classes, 
    #                                 beta=beta, 
    #                                 threshold=threshold, 
    #                                 numberOfSteps=config.num_time_steps_extract, 
    #                                 expansion=config.expansion)
    #     featSize = 512
    # elif config.resnet_model == 18:
    #     model = SpikeResNet18Model(numberOfChannels=channels, 
    #                                 numberOfClasses=config.num_classes, 
    #                                 beta=beta, 
    #                                 threshold=threshold, 
    #                                 numberOfSteps=config.num_time_steps_extract, 
    #                                 expansion=config.expansion)
    #     featSize = 512
    # elif config.resnet_model == 20:
    #     model = SpikeResNet20Model(numberOfChannels=channels, 
    #                                 numberOfClasses=config.num_classes, 
    #                                 beta=beta, 
    #                                 threshold=threshold, 
    #                                 numberOfSteps=config.num_time_steps_extract, 
    #                                 expansion=config.expansion)
    #     featSize = 512
    # elif config.resnet_model == 21:
    #     model = spikeLinearNet1(numberOfChannels=channels, 
    #                                 numberOfClasses=config.num_classes, 
    #                                 beta=beta, 
    #                                 threshold=threshold)
    #     featSize = 300
    # else:
    #     print("Feature: Not defined")
    #     return -1
    
    # weightsName = 'weights/spike/resnet' + str(config.resnet_model) + '_weights_' + config.dataset_ID + '_T_'+str(config.num_time_steps_train)+'_E_'+str(config.expansion)+'_A_'+str(config.auto_aug)+'_S_'+str(config.seed)+'.pth'
    # device = Utils.get_device()
    # model.load_state_dict(torch.load(weightsName, weights_only=True))
    # model = model.to(device)

    # Load In-Distribution data
    ID = np.load(IDpath)
    ID_spik_train = ID['arr0']  # In-Distribution training set spikes
    ID_feat_train = ID['arr1']  # In-Distribution training set features
    ID_prob_train = ID['arr2']  # In-Distribution training set outputs (usually with no softmax applied)
    ID_tags_train = ID['arr3']  # In-Distribution training set labels
    ID_volt_train = ID['arr8']  # In-Distribution training set voltages
    ID_spik_test  = ID['arr7']  # In-Distribution test set spikes
    ID_feat_test  = ID['arr4']  # In-Distribution test set features
    ID_prob_test  = ID['arr5']  # In-Distribution test set outputs (usually with no softmax applied)
    ID_volt_test  = ID['arr9']  # In-Distribution test set voltages
    # ID_tags_test  = ID['arr6']  # In-Distribution test set labels
    number_classes = ID_prob_train.shape[1]

    if features == 'features':
        print(f"Using spiking feature vector {ID_feat_train.shape}:")
        ID_features_train = ID_feat_train
        ID_features_test = ID_feat_test
    elif features == 'spikes':
        print(f"Using final spikings {ID_spik_train.shape}:")
        ID_features_train = ID_spik_train
        ID_penultimate_train = ID_feat_train
        ID_features_test = ID_spik_test
        ID_penultimate_test = ID_feat_test
    elif features == 'probs':
        print(f"Using logits {ID_prob_train.shape}:")
        ID_features_train = ID_prob_train
        ID_penultimate_train = ID_volt_train
        ID_features_test = ID_prob_test
        ID_penultimate_test = ID_volt_test
    elif features == 'voltages':
        print(f"Using voltage feature vector {ID_volt_train.shape}:")
        ID_features_train = ID_volt_train
        ID_features_test = ID_volt_test
    else:
        raise ValueError("Unknown ID feature type")

    for i in range(len(namesOOD)):
        OODpath = 'features/spike/case_' + case + '/' + namesOOD[i] + suffixID + '.npz'

        # Load Out-of-Distribution data
        OOD = np.load(OODpath)
        OOD_spik_train = OOD['arr0']  # Out-of-Distribution training set spikes
        OOD_feat_train = OOD['arr1']  # Out-of-Distribution training set features
        OOD_prob_train = OOD['arr2']  # Out-of-Distribution training set outputs (usually with no softmax applied)
        OOD_tags_train = OOD['arr3']  # Out-of-Distribution training set labels
        OOD_volt_train = OOD['arr8']  # Out-of-Distribution training set voltages
        OOD_spik_test  = OOD['arr7']  # Out-of-Distribution test set spikes
        OOD_feat_test  = OOD['arr4']  # Out-of-Distribution test set features
        OOD_prob_test  = OOD['arr5']  # Out-of-Distribution test set outputs (usually with no softmax applied)
        OOD_tags_test  = OOD['arr6']  # Out-of-Distribution test set labels
        OOD_volt_test  = OOD['arr9']  # Out-of-Distribution test set voltages

        if features == 'features':
            OOD_features_train = OOD_feat_train
            OOD_features_test = OOD_feat_test
        elif features == 'spikes':
            OOD_features_train = OOD_spik_train
            OOD_penultimate_train = OOD_feat_train
            OOD_features_test = OOD_spik_test
            OOD_penultimate_test = OOD_feat_test
        elif features == 'probs':
            OOD_features_train = OOD_prob_train
            OOD_penultimate_train = OOD_volt_train
            OOD_features_test = OOD_prob_test
            OOD_penultimate_test = OOD_volt_test
        elif features == 'voltages':
            OOD_features_train = OOD_volt_train
            OOD_features_test = OOD_volt_test
        else:
            raise ValueError("Unknown OOD feature type")

        # Methodology includes the IN/OOD classification where ID are positive samples taken from ID_test_set
        # and OOD are negative samples taken from OOD_train_set (or maybe OOD_train_set + OOD_test_set)

        for j, method in enumerate(methods):
            idx = method_to_idx[method]

            print(f"{j}. {methods[j]}")

            #     # plt.figure(figsize=(10, 6))
            #     # plt.plot(test_distances[0:20000], alpha=0.5, label="Distances")
            #     # plt.plot(test_labels[0:20000], linewidth=2, label="Labels")
            #     # plt.title("MSP")
            #     # plt.xlabel("Sample")
            #     # plt.ylabel("Distance")
            #     # plt.grid(True)
            #     # plt.legend()
            #     # plt.show()

            if method == 'MSP':
                # print(f"MSP on {namesOOD[i]}")
                test_labels, test_predictions, test_distances = Metrics.MSP(ID_features_test, OOD_features_test, number_classes)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 
                # print(f"MSP Done.\n")

            elif method == 'MLS':
                # print(f"MLS on {namesOOD[i]}")
                test_labels, test_predictions, test_distances = Metrics.MLS(ID_features_test, OOD_features_test, number_classes)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 
                # print(f"MLS Done.\n")
            
            elif method == 'NCM':
                # print(f"NCM on {namesOOD[i]}")
                number_classes = ID_prob_train.shape[1]
                test_labels, test_predictions, test_distances = Metrics.NCM(ID_features_train, ID_tags_train, ID_features_test, OOD_features_test, number_classes)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 
                # print(f"NCM Done.\n")

            elif method == 'ENGY':
                print(f"ENGY on {namesOOD[i]}")
                # print(f"before energy stats {stats[i, :]}")
                test_labels, test_predictions, test_distances = Metrics.ENGY(ID_features_test, OOD_features_test, T=1.0)
                # print(f"ENGY test_labels: {test_labels}, test_predictions: {test_predictions}, test_distances: {test_distances}")
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                # print(f"Energy auroc: {auroc:.2f}, aupr: {aupr:.2f}")
                # print(f"Energy True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 
                # print(f"ENGY Done.\n")
                # print(f"after energy stats {stats[i, :]}")

            elif method == 'SD':
                # this method needs revision
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
                # print(f"threshold_tpr95: {threshold_tpr95}")
                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"SD (spike distance) Execution time: {execution_time:.4f} seconds.\n")

            elif method == 'MD':
                # print(f"MD on {namesOOD[i]}")
                number_classes = ID_prob_train.shape[1]
                number_features = ID_features_train.shape[1]
                test_labels, test_predictions, test_distances = Metrics.MD(ID_features_train, ID_tags_train, ID_features_test, OOD_features_test, number_classes, number_features)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 
                # print(f"MD Done.")

            elif method == 'KNN':
                # print(f"KNN on {namesOOD[i]}")
                number_neighbors = 10
                test_labels, test_predictions, test_distances = Metrics.KNN(ID_features_train, ID_tags_train, ID_features_test, OOD_features_test, number_neighbors)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 

            elif method == 'VIM':
                number_classes = ID_prob_train.shape[1]
                number_features = ID_features_train.shape[1]
                number_residuals = number_features - number_classes
                # print(f"before vim stats {stats[i, :]}")

                if features == 'probs' or features == 'spikes':
                    print(f"VIM on {namesOOD[i]}")
                    mu = ID_penultimate_train.mean(axis=0)
                    X = ID_penultimate_train - mu
                    _, _, Vt = np.linalg.svd(X, full_matrices=False)
                    V = Vt.T[:, number_classes:number_classes + number_residuals]
                    test_labels, test_predictions, test_distances = Metrics.VIM(ID_penultimate_test, ID_features_test, OOD_penultimate_test, OOD_features_test, mu, V)
                    # print(f"VIM test_labels: {test_labels}, test_predictions: {test_predictions}, test_distances: {test_distances}")
                    auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                    # print(f"VIM True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                else:
                    # print(f"VIM method is not applicable for {features}.")
                    auroc, aupr, fpr95 = -0.01, -0.01, -0.01
                print(f"Vim auroc: {auroc:.2f}, aupr: {aupr:.2f}")
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95
                # print(f"after vim stats {stats[i, :]}")

            elif method == 'ASH':
                number_classes = ID_prob_train.shape[1]
                # number_features = ID_features_train.shape[1]
                # number_residuals = number_features - number_classes

                if features == 'features' or features == 'voltages':
                    print(f"ASH on {namesOOD[i]}")
                    test_labels, test_predictions, test_distances = Metrics.ASH(ID_features_test, OOD_features_test, keep_ratio=0.5)

                    auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                else:
                    # print(f"VIM method is not applicable for {features}.")
                    auroc, aupr, fpr95 = -0.01, -0.01, -0.01
                # print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                # print(f"auroc: {auroc:.2f}, aupr {aupr:.2f}, tpr95 {tpr95:.2f}, fpr95 {fpr95:.2f}")
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95

            elif method == 'FKM':
                print(f"K-MEANS Full Clustering on {namesOOD[i]}")
                number_neighbors = 100
                test_labels, test_predictions, test_distances = Metrics.FKM(ID_features_train, ID_features_test, OOD_features_test, number_neighbors)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95
                # print(f"KMEANS Full Execution done")

            elif method == 'CKM':
                print(f"K-MEANS Clustering per Class on {namesOOD[i]}")
                number_classes = ID_prob_train.shape[1]
                ncpc = 5
                test_labels, test_predictions, test_distances = Metrics.CKM(ID_features_train, ID_tags_train, ID_features_test, OOD_features_test, number_classes, ncpc)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                
                print(f"True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95
                # print(f"CKM Done.")

            elif method == 'NNDR':
                # needs revision
                start_time = time.time()
                number_neighbors = 11000 
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
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95
                end_time = time.time()  # Record end time
                execution_time = end_time - start_time  # Calculate execution time
                print(f"NNDR Execution time: {execution_time:.4f} seconds")

            elif method == 'ODIN':

                # ODIN uses the output probabilities from the softmax layer of a neural network. 
                # These scores represent the model's confidence in its predictions for each class.
                
                # ODIN applies temperature scaling to the softmax scores to make the distribution 
                # sharper or smoother. This is done by dividing the logits (pre-softmax activations)
                # by a temperature parameter ( T ) before applying the softmax function.
                
                # ODIN perturbs the input slightly in the direction that maximizes the 
                # softmax score for the predicted class. This preprocessing step helps to 
                # amplify the difference between in-distribution and out-of-distribution samples.
                    
                print(f"ODIN on {namesOOD[i]}")
                test_labels, test_predictions, test_distances = Metrics.ODIN(ID_features_test, OOD_features_test, eps=0.0014, T=1000.0)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                print(f"ODIN auroc: {auroc:.2f}, aupr: {aupr:.2f}")
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95

            else:
                pass

        del OOD, OOD_feat_train, OOD_prob_train, OOD_prob_test
        print()

    return stats


def statistics(config, acc_spk, acc_mem):
    # Statistics
    feature_types = ['features', 'spikes', 'probs', 'voltages']
    results_filename = f'results/OoD_case_{config.case}_{config.dataset_ID}_ResNet{config.resnet_model}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_E_{config.expansion}_A_{config.auto_aug}_S_{config.seed}.txt'
    ood_datasets = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]

    # Open file for writing
    with open(results_filename, 'w') as f:
        # Write header information
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Out-of-Distribution Detection Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Out-of-Distribution Datasets: {', '.join(ood_datasets)}\n")
        f.write(f"  Model: ResNet{config.resnet_model}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        if config.expansion == 1:
            f.write(f"  No population coding.\n")
        elif config.expansion > 1:
            f.write(f"  Population coding with {config.expansion} expansions.\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Accuracy on spikes: {acc_spk:05.2f}\n")
        f.write(f"Accuracy on membrane: {acc_mem:05.2f}\n")
        f.write(f"{'='*60}\n\n")
        
        all_results = {}
        methods = config.methods

        # Process each feature type
        for feature_type in feature_types:
            print(f"Processing statistics for feature type: {feature_type}")
            f.write(f"Feature Type: {feature_type.upper()}\n")
            f.write(f"{'-'*40}\n")
            
            try:
                stats = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=feature_type)
                
                if stats is not None and len(stats) > 0:
                    formatted_stats = np.array([[f'{elem*100:.2f}' for elem in row] for row in stats])
                    all_results[feature_type] = formatted_stats
                    
                    # Build the exact format you want
                    # Each method gets 5 characters, separated by ' | ' (3 chars)
                    # AUROC section: 5 methods = 5 chars + 4 separators = 37 chars total
                    # AUPR section: same = 37 chars
                    # FPR95 section: same = 37 chars
                    span = len(methods) * 6 + (len(methods) - 1) * 3
                    
                    # Top header with metric spans
                    auroc_span = "AUROC".center(span)
                    aupr_span = "AUPR".center(span)
                    fpr95_span = "FPR95".center(span)

                    top_header = f"  {'Dataset':10s}: {auroc_span} | {aupr_span} | {fpr95_span}"
                    
                    # Method header - repeat methods 3 times
                    methods_line = ' | '.join([f'{method:>6s}' for method in methods])
                    method_header = f"  {'':10s}: {methods_line} | {methods_line} | {methods_line}"

                    # Separator line matching the total width
                    total_width = len(method_header)
                    separator = '-' * total_width
                    
                    # Write formatted table
                    f.write(f"{top_header}\n")
                    f.write(f"{method_header}\n")
                    f.write(f"{separator}\n")
                    
                    # Data rows
                    for i, row in enumerate(formatted_stats):
                        if i < len(ood_datasets):
                            dataset_name = ood_datasets[i]
                            formatted_row = ' | '.join([f'{cell:>6s}' for cell in row])
                            f.write(f"  {dataset_name:<10s}: {formatted_row}\n")

                    # Console output
                    # print(f"  Results for {feature_type}:")
                    # print(f"{top_header}")
                    # print(f"{method_header}")  
                    # print(f"{separator}")
                    # for i, row in enumerate(formatted_stats):
                    #     if i < len(ood_datasets):
                    #         dataset_name = ood_datasets[i]
                    #         formatted_row = ' | '.join([f'{cell:>6s}' for cell in row])
                    #         print(f"  {dataset_name:<10s}: {formatted_row}")
                else:
                    f.write("  No data available\n")
                    print(f"  No data available for {feature_type}")
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                f.write(f"  {error_msg}\n")
                print(f"  {error_msg}")
            finally:
                f.write("\n")
                print()

    print(f"\nDetailed statistics saved to: {results_filename}")


def statistics_test_population(config, acc_spk, acc_mem):
    from train import test_accuracy_population_2, test_accuracy
    from feature import feature_extraction_spike
    # Statistics for Population Coding Comparison
    feature_types = ['features', 'spikes', 'probs', 'voltages']
    results_filename = f'results/PCC_{config.case}_{config.dataset_ID}_ResNet{config.resnet_model}_E_{config.expansion}_A_{config.auto_aug}.txt'
    ood_datasets = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]

    # Open file for writing
    with open(results_filename, 'w') as f:
        # Write header information
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Out-of-Distribution Detection Results\n")
        f.write(f"Compare the results for different time steps for feature extraction\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Out-of-Distribution Datasets: {', '.join(ood_datasets)}\n")
        f.write(f"  Model: ResNet{config.resnet_model}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        if config.expansion == 1:
            f.write(f"  No population coding.\n")
        elif config.expansion > 1:
            f.write(f"  Population coding with {config.expansion} expansions.\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        all_results = {}
        methods = config.methods
        auroc_results = {ftype: {ood: {} for ood in ood_datasets} for ftype in feature_types}
        acc_spk_results = {}
        acc_mem_results = {}

        # Loop over time steps
        for time_step in [2] + list(range(4, 12, 4)):
            config.num_time_steps_extract = time_step
            # acc_spk, acc_mem = test_accuracy_population_2(config)
            if config.expansion == 1:
                acc_spk, acc_mem = test_accuracy(config)
            elif config.expansion > 1:
                acc_spk, acc_mem = test_accuracy_population_2(config)
            acc_spk_results[time_step] = acc_spk
            acc_mem_results[time_step] = acc_mem
            print(f"Feature extraction using {config.num_time_steps_extract} time steps")
            feature_extraction_spike(config)

            # Process each feature type
            for feature_type in feature_types:
                print(f"Processing statistics for feature type: {feature_type}")
                try:
                    stats = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=feature_type)
                    print(f"stats: {stats}")
                    # stats shape: [num_ood_datasets, num_methods*3]
                    if stats is not None and len(stats) > 0:
                        for ood_idx, ood_dataset in enumerate(ood_datasets):
                            # AUROC values are the first len(methods) columns
                            auroc_vals = stats[ood_idx, :len(methods)]
                            # Store AUROC values for this time step
                            if time_step not in auroc_results[feature_type][ood_dataset]:
                                auroc_results[feature_type][ood_dataset][time_step] = auroc_vals
                            else:
                                # Overwrite if already present
                                auroc_results[feature_type][ood_dataset][time_step] = auroc_vals
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(f"  {error_msg}")

        # Write accuracy tables
        f.write(f"\n{'='*60}\n")
        f.write("Accuracy on Spikes vs. Time Steps\n")
        f.write('-'*40 + "\n")
        f.write("TimeStep   Accuracy\n")
        f.write('-'*22 + "\n")
        for time_step in sorted(acc_spk_results.keys()):
            f.write(f"{time_step:<10d} {acc_spk_results[time_step]:>8.2f}\n")

        f.write(f"\n{'='*60}\n")
        f.write("Accuracy on Membrane vs. Time Steps\n")
        f.write('-'*40 + "\n")
        f.write("TimeStep   Accuracy\n")
        f.write('-'*22 + "\n")
        for time_step in sorted(acc_mem_results.keys()):
            f.write(f"{time_step:<10d} {acc_mem_results[time_step]:>8.2f}\n")

        # After collecting all results, write tables for each feature type and OOD dataset
        for feature_type in feature_types:
            f.write(f"\n{'='*60}\n")
            f.write(f"Feature Type: {feature_type.upper()}\n")
            f.write(f"{'-'*60}\n")
            for ood_dataset in ood_datasets:
                f.write(f"\nAUROC vs. Time Steps for OOD Dataset: {ood_dataset}\n")
                # Table header
                method_header = 'TimeStep'.ljust(10) + ''.join([f"{method:>10s}" for method in methods]) + "\n"
                f.write(method_header)
                f.write('-' * (10 + 10*len(methods)) + "\n")
                # Table rows
                for time_step in sorted(auroc_results[feature_type][ood_dataset].keys()):
                    auroc_vals = auroc_results[feature_type][ood_dataset][time_step]
                    row = f"{time_step:<10d}" + ''.join([f"{val*100:>10.2f}" for val in auroc_vals]) + "\n"
                    f.write(row)
            f.write("\n")

    print(f"\nDetailed statistics saved to: {results_filename}")


def statistics_test_1(config, acc_spk, acc_mem):
    from feature import feature_extraction_spike

    feature_types = ['features', 'spikes', 'probs', 'voltages']
    ood_datasets = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]
    resnet_models = [4, 10, 18]

    results_filename = f'results/{config.methods[0]}_{config.case}_{config.dataset_ID}_A_{config.auto_aug}.txt'

    with open(results_filename, 'w') as f:
        # Write header information
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Out-of-Distribution Detection Results\n")
        f.write(f"Compare the results for different population coding expansions\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Out-of-Distribution Datasets: {', '.join(ood_datasets)}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Accuracy on spikes: {acc_spk:05.2f}\n")
        f.write(f"Accuracy on membrane: {acc_mem:05.2f}\n")
        f.write(f"{'='*60}\n\n")

        # for expansion in expansions:
        #     config.expansion = expansion
        f.write(f"\n{' ' * 30}population coding expansion {config.expansion}\n")

        # For each resnet model, train and extract features ONCE
        all_stats = {}
        for resnet_model in resnet_models:
            print(f"Training ResNet{resnet_model} with expansion {config.expansion}")
            if resnet_model == 4:
                model_name = "spike-Conv"
            elif resnet_model == 10:
                model_name = "spike-ResNet10"
            elif resnet_model == 18:
                model_name = "spike-ResNet18"
            else:
                model_name = f"resnet{resnet_model}"
            config.resnet_model = resnet_model
    
            torch.cuda.empty_cache()
            training_population(config)
            feature_extraction_spike(config)
            gc.collect()
            torch.cuda.empty_cache()
            # Collect stats for all feature types
            all_stats[resnet_model] = {}
            for feature_type in feature_types:
                stats = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=config.methods, features=feature_type)
                print(f"stats: {stats}")
                all_stats[resnet_model][feature_type] = stats
            gc.collect()
            torch.cuda.empty_cache()

        # Now, for each OOD dataset, build the table row
        for ood_idx, ood_dataset in enumerate(ood_datasets):
            row = ood_dataset.ljust(15)
            for resnet_model in resnet_models:
                feature_aurocs = []
                for feature_type in feature_types:
                    stats = all_stats[resnet_model][feature_type]
                    auroc = stats[ood_idx, 0] if stats is not None and len(stats) > ood_idx else float('nan')
                    feature_aurocs.append(f"{auroc*100:.2f}")
                cell = " / ".join(feature_aurocs)
                row += f"| {cell:<40}"
            f.write(row + "\n")
        f.write("\n")
    print(f"\nDetailed statistics saved to: {results_filename}")


def statistics_exp_1(config, seeds):
    feature_types = ['features', 'spikes', 'probs', 'voltages']
    methods = config.methods
    ood_datasets = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]
    results_filename = (
        f'results/EX1_{config.dataset_ID}_ResNet{config.resnet_model}_'
        f'T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_'
        f'E_{config.expansion}_A_{config.auto_aug}_seeds.txt'
    )

    # stats_all[feature_type][ood_dataset][method][metric][seed]
    stats_all = {ftype: {ood: {m: [[] for _ in range(3)] for m in methods} for ood in ood_datasets} for ftype in feature_types}

    for seed in seeds:
        config.seed = seed
        # If you need to load weights/features per seed, do it here
        # e.g., config.weights_path = f'weights/seed_{seed}_...pth'
        # or set config.feature_file = ...
        feature_extraction_spike(config)
        for feature_type in feature_types:
            # test_metrics should load the correct features for the current seed
            stats = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=feature_type)
            print(f"Stats for seed {seed}, feature type {feature_type}: {stats}")
           
            for ood_idx, ood_dataset in enumerate(ood_datasets):
                for m_idx, method in enumerate(methods):
                    for metric_idx in range(3):  # AUROC, AUPR, FPR95
                        stats_all[feature_type][ood_dataset][method][metric_idx].append(stats[ood_idx, m_idx + metric_idx*len(methods)])

    # Write results
    with open(results_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Out-of-Distribution Detection Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Out-of-Distribution Datasets: {', '.join(ood_datasets)}\n")
        f.write(f"  Model: ResNet{config.resnet_model}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        if config.expansion == 1:
            f.write(f"  No population coding.\n")
        elif config.expansion > 1:
            f.write(f"  Population coding with {config.expansion} expansions.\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{'='*60}\n\n")

        for feature_type in feature_types:
            f.write(f"Feature Type: {feature_type.upper()}\n")
            f.write(f"{'-'*40}\n")
            # Table header formatting
            col_width = 13  # width for each metric cell
            sep = ' | '
            # Top header with metric spans
            metric_titles = ['AUROC', 'AUPR', 'FPR95']
            header = '  {:10s}:'.format('Dataset')
            for title in metric_titles:
                header += sep + title.center(col_width * len(methods) + len(sep)*(len(methods)-1))
            f.write('-' * len(header) + '\n')
            f.write(header + '\n')
            # Method header
            method_row = '  {:10s}:'.format('')
            for _ in metric_titles:
                method_row += sep + sep.join([f'{m:^{col_width}s}' for m in methods])
            f.write(method_row + '\n')
            f.write('-' * len(header) + '\n')
            # Data rows
            for ood_dataset in ood_datasets:
                row = f"  {ood_dataset:<10s}:"
                for metric_idx in range(3):  # AUROC, AUPR, FPR95
                    vals = []
                    for method in methods:
                        arr = np.array(stats_all[feature_type][ood_dataset][method][metric_idx])
                        mean = np.mean(arr)
                        std = np.std(arr)
                        vals.append(f"{mean*100:6.2f}({std*100:4.2f})")
                    row += sep + sep.join([f'{v:^{col_width}s}' for v in vals])
                f.write(row + '\n')
            f.write('-' * len(header) + '\n\n')
    print(f"\nDetailed statistics saved to: {results_filename}")
