import gc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestCentroid
import torch
from feature import feature_extraction_spike, feature_extraction_conv
from utils.Utils import Utils, distances_from_average_clusters
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import k_means, DBSCAN, KMeans
import time 
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
from train import training
import gc
import os
from torchvision import datasets, transforms
from models.spikeresnet import spikeConvNN1, spikeConvNN2, spikeConvNN4, SpikeResNet9Model, SpikeResNet10Model, SpikeResNet18Model, SpikeResNet20Model
from models.plain import spikeLinearNet1
import os
import matplotlib.pyplot as plt
import json
from scipy.special import logsumexp

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
        Energy-based OOD detection method, matching OpenOOD.
        Score is T * logsumexp(logits / T), higher for ID.
        '''

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        ID_distances = T * logsumexp(ID_feat_test / T, axis=1)
        OOD_distances = T * logsumexp(OOD_feat_test / T, axis=1)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop = False)
        threshold = threshold_tpr95

        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"ENGY Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
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
    def ODIN_pert(ID_inputs_test, OOD_inputs_test, model, ID_feat_test, OOD_feat_test, eps=0.0014, T=1000.0):
        """
        Implements the full ODIN algorithm for OoD detection with input perturbation.
        Inputs:
            ID_inputs_test:   Matrix of in-distribution test inputs (e.g., images or spike inputs)
            OOD_inputs_test:  Matrix of out-of-distribution test inputs
            model:            Trained model for forward passes
            ID_feat_test:     Matrix of in-distribution test logits (for fallback or comparison)
            OOD_feat_test:    Matrix of out-of-distribution test logits
            eps:              Small perturbation value
            T:                Temperature scaling parameter
        """

        start_time = time.time()
        ID_labels = np.ones((len(ID_feat_test)))
        OOD_labels = np.zeros((len(OOD_feat_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        # Function to compute ODIN score with perturbation
        def compute_odin_score(inputs, model, eps, T):
            scores = []
            input_std = [0.5, 0.5, 0.5]  # Approximate for CIFAR10/SVHN
            for x in inputs:
                x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
                logits = model(x_tensor.unsqueeze(0))
                softmax_T = torch.softmax(logits / T, dim=1)
                pred_class = torch.argmax(softmax_T, dim=1)
                loss = -torch.log(softmax_T[0, pred_class])
                loss.backward()
                grad = x_tensor.grad.detach()
                # Match OpenOOD: sign of grad, scaled by input_std
                gradient = torch.ge(grad, 0).float() * 2 - 1  # sign
                gradient[:, 0] /= input_std[0]
                gradient[:, 1] /= input_std[1]
                gradient[:, 2] /= input_std[2]
                x_pert = x_tensor - eps * gradient
                logits_pert = model(x_pert.unsqueeze(0))
                softmax_T_pert = torch.softmax(logits_pert / T, dim=1)
                score = torch.max(softmax_T_pert, dim=1)[0].item()
                scores.append(score)
            return np.array(scores)

        ID_distances = compute_odin_score(ID_inputs_test, model, eps, T)
        OOD_distances = compute_odin_score(OOD_inputs_test, model, eps, T)
        test_distances = np.concatenate((ID_distances, OOD_distances))

        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_distances, 1, drop=False)
        threshold = threshold_tpr95

        # Make predictions based on the distances
        ID_predictions = (ID_distances > threshold).astype(np.int32)
        OOD_predictions = (OOD_distances > threshold).astype(np.int32)

        # Concatenate predictions
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"ODIN_pert Execution time: {execution_time:.4f} seconds.")

        # Return labels, predictions, and ODIN scores
        return test_labels, test_predictions, test_distances


    @staticmethod
    def VIM(ID_penultimate_test, OOD_penultimate_test, ID_logits_test, OOD_logits_test, mu, null_space_eigvecs, alpha=0.1):
        '''
        VIM method for OOD detection.
        
        Matches OpenOOD: Use energy (logsumexp of logits) - alpha * residual_norm.
        Inputs:
            ID_penultimate_test:   Penultimate layer features for ID test set
            OOD_penultimate_test:  Penultimate layer features for OOD test set
            ID_logits_test:        Logits for ID test set
            OOD_logits_test:       Logits for OOD test set
            mu:                    Mean of training penultimate features
            null_space_eigvecs:    Null space eigenvectors
            alpha:                 Regularization parameter
        Outputs:
            test_labels:           Ground truth labels (1 for ID, 0 for OOD)
            test_predictions:      Predictions based on threshold
            test_distances:        Detection scores (higher for ID)
        '''
        from scipy.special import logsumexp
        
        start_time = time.time()
        ID_labels = np.ones((len(ID_penultimate_test)))
        OOD_labels = np.zeros((len(OOD_penultimate_test)))
        test_labels = np.concatenate((ID_labels, OOD_labels))

        # Compute residuals
        ID_residual = (ID_penultimate_test - mu) @ null_space_eigvecs
        r_norm_ID = np.linalg.norm(ID_residual, axis=1)
        energy_ID = logsumexp(ID_logits_test, axis=1)
        ID_scores = energy_ID - alpha * r_norm_ID

        OOD_residual = (OOD_penultimate_test - mu) @ null_space_eigvecs
        r_norm_OOD = np.linalg.norm(OOD_residual, axis=1)
        energy_OOD = logsumexp(OOD_logits_test, axis=1)
        OOD_scores = energy_OOD - alpha * r_norm_OOD

        # For consistency with other methods, use S(x) directly, higher values indicate ID
        test_distances = np.concatenate((ID_scores, OOD_scores))

        # Set threshold at 5th percentile of ID scores (literature standard for higher=ID)
        if len(ID_scores) > 0:
            threshold = np.percentile(ID_scores, 5)
        else:
            threshold = 0.0  # Fallback

        ID_predictions = (ID_scores > threshold).astype(np.int32)  # Higher S(x) for ID
        OOD_predictions = (OOD_scores <= threshold).astype(np.int32)
        test_predictions = np.concatenate((ID_predictions, OOD_predictions))

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"VIM Execution time: {execution_time:.4f} seconds.")

        return test_labels, test_predictions, test_distances
    
    @staticmethod
    def ASH_Old(ID_feat_test, OOD_feat_test, keep_ratio=0.25):
        """
        ASH (Activation Sparsity Hypothesis) for OoD detection, based on Chun et al. (ICLR 2022).
        Uses penultimate layer features, sparsifies top-k activations, normalizes to preserve mass,
        and scores as -log(sum). Threshold at 95th percentile of ID scores; predict ID if score < threshold.
        """
        
        start_time = time.time()
        
        # Process ID features
        k = max(1, int(ID_feat_test.shape[1] * keep_ratio))
        ID_top_k_threshold = np.partition(ID_feat_test, -k, axis=1)[:, -k]
        ID_suppressed_features = np.where(ID_feat_test >= ID_top_k_threshold[:, None], ID_feat_test, 0.0)
        ID_suppressed_features = np.where(ID_suppressed_features < 0.0, 0.0, ID_suppressed_features)
        
        ID_row_sums = ID_suppressed_features.sum(axis=1, keepdims=True)
        zero_rows = (ID_row_sums == 0).flatten()
        if np.any(zero_rows):
            ID_suppressed_features[zero_rows, :] = 1e-12
            ID_row_sums = ID_suppressed_features.sum(axis=1, keepdims=True)
        
        ID_orig_pos_sum = np.where(ID_feat_test > 0.0, ID_feat_test, 0.0).sum(axis=1, keepdims=True)
        ID_scale = np.where(ID_orig_pos_sum > 0.0, (ID_orig_pos_sum + 1e-12) / (ID_row_sums + 1e-12), 1.0)
        ID_normalized_features = ID_suppressed_features * ID_scale
        ID_normalized_features = np.nan_to_num(ID_normalized_features, nan=1e-12, posinf=1e12, neginf=1e-12)
        ID_ash_scores = -np.log(np.sum(ID_normalized_features, axis=1) + 1e-12)
        
        # Process OOD features (identical to ID)
        k = max(1, int(OOD_feat_test.shape[1] * keep_ratio))
        OOD_top_k_threshold = np.partition(OOD_feat_test, -k, axis=1)[:, -k]
        OOD_suppressed_features = np.where(OOD_feat_test >= OOD_top_k_threshold[:, None], OOD_feat_test, 0.0)
        OOD_suppressed_features = np.where(OOD_suppressed_features < 0.0, 0.0, OOD_suppressed_features)
        
        OOD_row_sums = OOD_suppressed_features.sum(axis=1, keepdims=True)
        zero_rows_o = (OOD_row_sums == 0).flatten()
        if np.any(zero_rows_o):
            OOD_suppressed_features[zero_rows_o, :] = 1e-12
            OOD_row_sums = OOD_suppressed_features.sum(axis=1, keepdims=True)
        
        OOD_orig_pos_sum = np.where(OOD_feat_test > 0.0, OOD_feat_test, 0.0).sum(axis=1, keepdims=True)
        OOD_scale = np.where(OOD_orig_pos_sum > 0.0, (OOD_orig_pos_sum + 1e-12) / (OOD_row_sums + 1e-12), 1.0)
        OOD_normalized_features = OOD_suppressed_features * OOD_scale
        OOD_normalized_features = np.nan_to_num(OOD_normalized_features, nan=1e-12, posinf=1e12, neginf=1e-12)
        OOD_ash_scores = -np.log(np.sum(OOD_normalized_features, axis=1) + 1e-12)
        
        # Combine and negate scores for consistency (higher = ID)
        test_labels = np.concatenate((np.ones(len(ID_feat_test)), np.zeros(len(OOD_feat_test))))
        test_scores = np.concatenate((-ID_ash_scores, -OOD_ash_scores))
        
        # Threshold: 5th percentile of negated ID scores (higher = ID)
        if len(ID_ash_scores) > 0:
            threshold = np.percentile(-ID_ash_scores, 5)
        else:
            threshold = 0.0  # Fallback
        
        # Predictions: ID (1) if score > threshold, OOD (0) if <=
        test_predictions = (test_scores > threshold).astype(int)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"ASH Execution time: {execution_time:.4f} seconds.")
        
        return test_labels, test_predictions, test_scores

    @staticmethod
    def ASH(ID_logits_test, OOD_logits_test, keep_ratio=0.25):
        """
        ASH (Activation Sparsity Hypothesis) for OoD detection, matching OpenOOD.
        Sparsifies top-k logits, computes logsumexp as score (higher for ID).
        """
        start_time = time.time()
        
        # Process ID logits
        k = max(1, int(ID_logits_test.shape[1] * keep_ratio))
        ID_top_k_threshold = np.partition(ID_logits_test, -k, axis=1)[:, -k]
        ID_suppressed = np.where(ID_logits_test >= ID_top_k_threshold[:, None], ID_logits_test, -1000.0)
        ID_energy = logsumexp(ID_suppressed, axis=1)
        
        # Process OOD logits
        k = max(1, int(OOD_logits_test.shape[1] * keep_ratio))
        OOD_top_k_threshold = np.partition(OOD_logits_test, -k, axis=1)[:, -k]
        OOD_suppressed = np.where(OOD_logits_test >= OOD_top_k_threshold[:, None], OOD_logits_test, -1000.0)
        OOD_energy = logsumexp(OOD_suppressed, axis=1)
        
        # Combine scores
        test_labels = np.concatenate((np.ones(len(ID_logits_test)), np.zeros(len(OOD_logits_test))))
        test_scores = np.concatenate((ID_energy, OOD_energy))

        # Use the same TPR95 thresholding protocol as other methods.
        _, threshold_tpr95 = Utils.find_threshold(test_labels, test_scores, 1, drop=False)
        threshold = threshold_tpr95

        # Predictions: ID (1) if score > threshold
        test_predictions = (test_scores > threshold).astype(np.int32)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"ASH_OpenOOD Execution time: {execution_time:.4f} seconds.")
        
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
    elif nameID == 'CIFAR100':
        suffixID = '-on_cifar100'
    elif nameID == 'SVHN':
        suffixID = '-on_svhn'
    else:
        raise ValueError("Unknown ID dataset name")

    # Find the names of OOD datasets
    namesOOD = [dataset for dataset in config.dataset_feat if dataset != config.dataset_ID]

    method_to_idx = {m: idx for idx, m in enumerate(methods)}

    stats = np.zeros((len(namesOOD), len(methods)*3), dtype=np.float64)
    if config.model_type == 'spike':
        IDpath = 'features/spike/exp' + case + '/' + nameID + suffixID + '.npz'
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
    else:
        IDpath = 'features/conv/exp' + case + '/' + nameID + suffixID + '.npz'
        # Load In-Distribution data
        ID = np.load(IDpath)
        ID_prob_train = ID['arr2']  # In-Distribution training set outputs (usually with no softmax applied)
        ID_tags_train = ID['arr3']  # In-Distribution training set labels
        ID_volt_train = ID['arr8']  # In-Distribution training set voltages
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

    if 'ODIN_pert' in methods:
        # Load model
        model_path = os.path.join('weights', f'{config.dataset_ID}_ResNet{config.resnet_model}_E{config.expansion}_S{config.seed}.pth')
        model = torch.load(model_path, map_location='cpu')
        model.eval()

        # Load ID inputs
        if nameID == 'CIFAR10':
            transform = Utils.get_cifar10_transforms(auto_aug=False, training=False)  # Use Utils for consistency
            id_dataset = datasets.CIFAR10(root='data/cifar10', train=False, transform=transform)
        elif nameID == 'SVHN':
            # Assuming similar transform logic; if Utils has get_svhn_transforms, use it
            transform = Utils.get_cifar10_transforms(auto_aug=False, training=False)  # Placeholder; adapt as needed
            id_dataset = datasets.SVHN(root='data/svhn', split='test', transform=transform)
        else:
            raise ValueError(f"Unsupported ID dataset {nameID} for ODIN_pert")

        id_loader = torch.utils.data.DataLoader(id_dataset, batch_size=config.batch_size, shuffle=False)
        id_inputs = []
        for batch in id_loader:
            id_inputs.append(batch[0])
        id_inputs = torch.cat(id_inputs, dim=0)

        # Load OOD inputs
        ood_inputs = {}
        for ood_name in namesOOD:
            if ood_name == 'SVHN':
                transform = Utils.get_cifar10_transforms(auto_aug=False, training=False)  # Adapt for SVHN
                ood_dataset = datasets.SVHN(root='data/svhn', split='test', transform=transform)
            elif ood_name == 'CIFAR10':
                transform = Utils.get_cifar10_transforms(auto_aug=False, training=False)
                ood_dataset = datasets.CIFAR10(root='data/cifar10', train=False, transform=transform)
            elif ood_name == 'CIFAR100':
                transform = Utils.get_cifar100_transforms(auto_aug=False, training=False)  # Use Utils for CIFAR-100
                ood_dataset = datasets.CIFAR100(root='data/cifar100', train=False, transform=transform)
            else:
                raise ValueError(f"Unsupported OOD dataset {ood_name} for ODIN_pert")

            ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=config.batch_size, shuffle=False)
            ood_data = []
            for batch in ood_loader:
                ood_data.append(batch[0])
            ood_inputs[ood_name] = torch.cat(ood_data, dim=0)

    for i in range(len(namesOOD)):

        if config.model_type == 'spike':
            OODpath = 'features/spike/exp' + case + '/' + namesOOD[i] + suffixID + '.npz'

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
        
        elif config.model_type == 'conv':
            OODpath = 'features/conv/exp' + case + '/' + namesOOD[i] + suffixID + '.npz'

            # Load Out-of-Distribution data
            OOD = np.load(OODpath)
            OOD_prob_train = OOD['arr2']  # Out-of-Distribution training set outputs (usually with no softmax applied)
            OOD_tags_train = OOD['arr3']  # Out-of-Distribution training set labels
            OOD_volt_train = OOD['arr8']  # Out-of-Distribution training set voltages
            OOD_prob_test  = OOD['arr5']  # Out-of-Distribution test set outputs (usually with no softmax applied)
            OOD_tags_test  = OOD['arr6']  # Out-of-Distribution test set labels
            OOD_volt_test  = OOD['arr9']  # Out-of-Distribution test set voltages
        print(f"Shapes of OOD data for {namesOOD[i]}: OOD_prob_train: {OOD_prob_train.shape}, OOD_volt_train: {OOD_volt_train.shape}, OOD_prob_test: {OOD_prob_test.shape}, OOD_volt_test: {OOD_volt_test.shape}")

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

            elif method == 'ODIN':
                print(f"ODIN on {namesOOD[i]}")
                test_labels, test_predictions, test_distances = Metrics.ODIN(ID_features_test, OOD_features_test, T=1000)
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 

            elif method == 'ODIN_pert':
                print(f"ODIN_pert on {namesOOD[i]}")
                test_labels, test_predictions, test_distances = Metrics.ODIN_pert(id_inputs, ood_inputs[namesOOD[i]], model, ID_features_test, OOD_features_test, eps=getattr(config, 'odin_eps', 0.0014), T=getattr(config, 'odin_T', 1000))
                auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)

                stats[i, idx] = auroc 
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95 

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

                if features == 'probs' or features == 'spikes':
                    print(f"VIM on {namesOOD[i]}")
                    expected_pen_dim = {4: 256, 10: 512, 18: 512}.get(getattr(config, 'resnet_model', None), None)
                    id_train_dim = ID_penultimate_train.shape[1]
                    id_test_dim = ID_penultimate_test.shape[1]
                    ood_test_dim = OOD_penultimate_test.shape[1]

                    dims_match = (id_train_dim == id_test_dim == ood_test_dim)
                    expected_match = (expected_pen_dim is None) or (
                        id_train_dim == expected_pen_dim and
                        id_test_dim == expected_pen_dim and
                        ood_test_dim == expected_pen_dim
                    )

                    if not dims_match or not expected_match:
                        print(
                            f"[VIM] Skipping due to penultimate-dimension mismatch "
                            f"(resnet={getattr(config, 'resnet_model', 'NA')}, "
                            f"expected={expected_pen_dim}, "
                            f"ID_train={id_train_dim}, ID_test={id_test_dim}, OOD_test={ood_test_dim})."
                        )
                        print(
                            "[VIM] This usually means cached features were generated with a different model "
                            "(e.g., ResNet10/18=512 reused for ResNet4=256). "
                            "Regenerate features with override_feature_extraction=true."
                        )
                        auroc, aupr, fpr95 = -0.01, -0.01, -0.01
                        print(f"VIM auroc: {auroc:.2f}, aupr: {aupr:.2f}")
                        stats[i, idx] = auroc
                        stats[i, len(methods) + idx] = aupr
                        stats[i, len(methods)*2 + idx] = fpr95
                        continue

                    number_features = ID_penultimate_train.shape[1]  # Use penultimate features for residuals
                    number_residuals = number_features - number_classes
                    mu = ID_penultimate_train.mean(axis=0)
                    X = ID_penultimate_train - mu
                    _, _, Vt = np.linalg.svd(X, full_matrices=False)
                    V = Vt.T[:, number_classes:number_classes + number_residuals]
                    # Compute adaptive alpha as in OpenOOD
                    r_norm_train = np.linalg.norm((ID_penultimate_train - mu) @ V, axis=1)
                    mean_r_norm = r_norm_train.mean()
                    if mean_r_norm == 0:
                        alpha = 1.0  # Default alpha if residual norms are zero
                    else:
                        alpha = logsumexp(ID_prob_train, axis=1).mean() / mean_r_norm
                    print(f"Adaptive alpha: {alpha:.4f}")
                    test_labels, test_predictions, test_distances = Metrics.VIM(ID_penultimate_test, OOD_penultimate_test, ID_prob_test, OOD_prob_test, mu, V, alpha=alpha)
                    # print(f"VIM test_labels: {test_labels}, test_predictions: {test_predictions}, test_distances: {test_distances}")
                    auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                    # print(f"VIM True positive rate: {tpr95:.2f}, False positive rate {fpr95:.2f}")
                else:
                    # print(f"VIM method is not applicable for {features}.")
                    auroc, aupr, fpr95 = -0.01, -0.01, -0.01
                print(f"VIM auroc: {auroc:.2f}, aupr: {aupr:.2f}")
                stats[i, idx] = auroc
                stats[i, len(methods) + idx] = aupr
                stats[i, len(methods)*2 + idx] = fpr95
                # print(f"after vim stats {stats[i, :]}")

            elif method == 'ASH':
                number_classes = ID_prob_train.shape[1]
                # number_features = ID_features_train.shape[1]
                # number_residuals = number_features - number_classes

                if features == 'probs' or features == 'spikes':
                    print(f"ASH on {namesOOD[i]}")
                    test_labels, test_predictions, test_distances = Metrics.ASH(ID_features_test, OOD_features_test, keep_ratio=0.25)

                    auroc, aupr, tpr95, fpr95 = Metrics.metrics(test_labels, test_predictions, test_distances)
                else:
                    # print(f"VIM method is not applicable for {features}.")
                    auroc, aupr, fpr95 = -0.01, -0.01, -0.01
                print(f"ASH auroc: {auroc:.2f}, fpr95: {fpr95:.2f}")
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

        # remove references to large arrays and close file handles if present
        try:
            del OOD_feat_train, OOD_prob_train, OOD_prob_test
        except Exception:
            pass
        try:
            if 'OOD' in locals() and hasattr(OOD, 'close'):
                OOD.close()
        except Exception:
            pass
        try:
            del OOD
        except Exception:
            pass
        # hint to the GC and free GPU cached memory if any
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print()

    # ensure ID file handle is closed and large arrays removed
    try:
        if 'ID' in locals() and hasattr(ID, 'close'):
            ID.close()
    except Exception:
        pass
    try:
        del ID, ID_spik_train, ID_feat_train, ID_prob_train, ID_tags_train, ID_volt_train, ID_spik_test, ID_feat_test, ID_prob_test, ID_volt_test
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass

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
            training(config)
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


def statistics_exp_1(config, seeds, expansions, resnet_models):
    """Compute experiment-1 statistics with separate near/far OOD tables and plots.

    signature changed to accept expansions and resnet_models so main can pass them in.
    Outputs: writes a TXT file with per-resnet Near / Far tables and saves plots (one per feature type)
    where each ResNet provides two curves (near and far).
    """
    if config.model_type == 'spike':
        feature_types = ['features', 'voltages']
        model_tags = {4: 'spike-Conv', 10: 'spike-ResNet10', 18: 'spike-ResNet18'}
    elif config.model_type == 'conv':
        feature_types = ['voltages']
        model_tags = {4: 'Conv', 10: 'ResNet10', 18: 'ResNet18'}
    methods = config.methods
    yaml_override_feature_extraction = bool(getattr(config, 'override_feature_extraction', False))

    out_dir = os.path.join('results', 'ex_1', 'exp'+str(config.case))
    os.makedirs(out_dir, exist_ok=True)

    # Prepare lists of near and far OOD from config (do not filter by dataset_ID)
    # We will compute averages over whichever of these appear in the computed stats.
    near_list = list(getattr(config, 'near_ood', []) or [])
    far_list = list(getattr(config, 'far_ood', []) or [])
    # Map resnet model numbers to human-friendly tags for legend/table output
    

    # Results container: results[resnet_model][feature_type] = list over expansions (dict with 'near'/'far' values or 'not trained')
    # Load existing JSON if present so we can resume partially-completed runs. If not present, create skeleton and save it.
    if config.model_type == 'spike':
        datafile = os.path.join(out_dir, f'EX1_{config.dataset_ID}_A_{config.auto_aug}_L_{config.loss}_data.json')
    elif config.model_type == 'conv':
        datafile = os.path.join(out_dir, f'EX1_{config.dataset_ID}_A_{config.auto_aug}_L_{config.loss}_conv_data.json')
    data_out = None
    if os.path.exists(datafile):
        try:
            with open(datafile, 'r') as jf:
                data_out = json.load(jf)
        except Exception:
            data_out = None

    if data_out is None:
        # build empty skeleton with 'not trained' placeholders
        data_out = {
            'dataset_ID': config.dataset_ID,
            'case': config.case,
            'expansions': list(expansions),
            'feature_types': feature_types,
            'model_tags': {str(k): v for k, v in model_tags.items()},
            'results': {}
        }
        for rm in resnet_models:
            data_out['results'][str(rm)] = {}
            for ft in feature_types:
                data_out['results'][str(rm)][ft] = [{'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'} for _ in expansions]
        try:
            with open(datafile, 'w') as jf:
                json.dump(data_out, jf, indent=2)
        except Exception as e:
            print(f"Warning: failed to write initial EX1 JSON skeleton: {e}")

    # Ensure per-seed checkpoint structure exists so interrupted runs can resume from the last completed seed.
    data_out.setdefault('seed_results', {})
    for rm in resnet_models:
        rm_key = str(rm)
        data_out['seed_results'].setdefault(rm_key, {})
        for ft in feature_types:
            data_out['seed_results'][rm_key].setdefault(ft, {})
            for i in range(len(expansions)):
                data_out['seed_results'][rm_key][ft].setdefault(str(i), {})

    # Initialize results from loaded or newly-created JSON skeleton. This allows skipping already-done entries.
    results = {}
    for rm in resnet_models:
        results[rm] = {}
        key = str(rm)
        for ft in feature_types:
            if key in data_out.get('results', {}) and ft in data_out['results'][key]:
                serial = data_out['results'][key][ft]
                # ensure the list length matches expansions
                entries = []
                for i in range(len(expansions)):
                    if i < len(serial):
                        entries.append(serial[i])
                    else:
                        entries.append({'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'})
                results[rm][ft] = entries
            else:
                results[rm][ft] = [{'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'} for _ in expansions]

    # Loop over resnet models and expansions
    for resnet_model in resnet_models:
        config.resnet_model = resnet_model
        print(f"Processing ResNet{resnet_model}")
        for idx, expansion in enumerate(expansions):
            config.expansion = expansion
            # If the JSON already contains valid results (not 'not trained') for this resnet+expansion for all
            # feature types, skip running it. Otherwise we'll run and overwrite the placeholder(s).
            all_trained = True
            for ft in feature_types:
                entry = results[resnet_model][ft][idx]
                near_val = entry.get('near') if isinstance(entry, dict) else 'not trained'
                far_val = entry.get('far') if isinstance(entry, dict) else 'not trained'
                if isinstance(near_val, str) and near_val == 'not trained':
                    all_trained = False
                    break
                if isinstance(far_val, str) and far_val == 'not trained':
                    all_trained = False
                    break
            if all_trained:
                print(f"Skipping ResNet{resnet_model} expansion {expansion} — already trained in JSON.")
                continue
            print(f"Processing expansion: {config.expansion}")
            config.override_feature_extraction = yaml_override_feature_extraction

            # For each feature type compute per-seed AUROC and then average across seeds separately for near and far
            # We will attempt all seeds and only mark the expansion as 'done' in JSON when every seed produced valid results.
            expansion_values = {ft: {'near': [], 'far': [], 'near_fpr': [], 'far_fpr': []} for ft in feature_types}
            # Track per-seed success (True if this seed produced numeric near/far for all feature types)
            seed_success = {seed: True for seed in seeds}

            def _valid_seed_entry_exp1(entry):
                if not isinstance(entry, dict):
                    return False
                try:
                    float(entry.get('near'))
                    float(entry.get('far'))
                    float(entry.get('near_fpr'))
                    float(entry.get('far_fpr'))
                    return True
                except Exception:
                    return False

            # Preload already-computed seed values for this expansion and skip those seeds.
            completed_seeds = set()
            for seed in seeds:
                seed_done = True
                for ft in feature_types:
                    cached = data_out['seed_results'][str(resnet_model)][ft][str(idx)].get(str(seed))
                    if _valid_seed_entry_exp1(cached):
                        expansion_values[ft]['near'].append(float(cached['near']))
                        expansion_values[ft]['far'].append(float(cached['far']))
                        expansion_values[ft]['near_fpr'].append(float(cached['near_fpr']))
                        expansion_values[ft]['far_fpr'].append(float(cached['far_fpr']))
                    else:
                        seed_done = False
                        break
                if seed_done:
                    completed_seeds.add(seed)
                    seed_success[seed] = True

            for seed in seeds:
                if seed in completed_seeds:
                    print(f"[exp1] Skipping completed seed: ResNet{resnet_model} E={expansion} S={seed}")
                    continue
                config.seed = seed
                print(f"Processing seed: {config.seed}")
                # We deliberately do NOT break on per-seed errors; instead record NaNs and continue so all seeds are attempted
                seed_ft_values = {}
                try:
                    # For efficiency, extract features once for the union of near and far OOD sets
                    orig_dataset_feat = getattr(config, 'dataset_feat', None)
                    union_list = []
                    try:
                        union_list = sorted(set(list(near_list) + list(far_list)))
                    except Exception:
                        union_list = list(near_list) + list(far_list)
                    print(f"Union list: {union_list}")

                    # If union_list is non-empty, temporarily set it and extract once
                    extracted_ok = True
                    if union_list:
                        try:
                            config.dataset_feat = list(union_list)
                            # print(f"Extracting features for union list: {config.dataset_feat}")
                            if config.model_type == 'spike':
                                feature_extraction_spike(config)
                            elif config.model_type == 'conv':
                                feature_extraction_conv(config)
                        except Exception as e:
                            # mark extraction failure for this seed; record NaNs below and continue
                            print(f"[exp1] Feature extraction failed for ResNet{resnet_model} E={expansion} S={seed}: {e}")
                            try:
                                import traceback
                                traceback.print_exc()
                            except Exception:
                                pass
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    print(f"[exp1] CUDA memory allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")
                            except Exception:
                                pass
                            extracted_ok = False

                        # Aggressive cleanup after extraction attempt to reduce GPU fragmentation
                        try:
                            import gc
                            gc.collect()
                        except Exception:
                            pass
                        try:
                            import torch
                            # synchronize and release cached memory
                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.synchronize()
                                except Exception:
                                    pass
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                # collect IPC tensors and reset peak stats if available
                                if hasattr(torch.cuda, 'ipc_collect'):
                                    try:
                                        torch.cuda.ipc_collect()
                                    except Exception:
                                        pass
                                try:
                                    torch.cuda.reset_peak_memory_stats()
                                except Exception:
                                    pass
                                try:
                                    print(f"[exp1] post-extract CUDA allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # For each feature type, run test_metrics for near and far respectively
                    num_methods = len(methods)
                    for ft in feature_types:
                        try:
                            if not extracted_ok:
                                # if extraction failed, append NaNs for this seed and mark seed as unsuccessful
                                expansion_values[ft]['near'].append(np.nan)
                                expansion_values[ft]['far'].append(np.nan)
                                seed_success[seed] = False
                                continue

                            # Run test on near list (if present)
                            if not near_list:
                                near_mean = np.nan
                                near_fpr_mean = np.nan
                            else:
                                config.dataset_feat = list(near_list)
                                stats_local = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=ft)
                                if stats_local is None or stats_local.size == 0:
                                    near_mean = np.nan
                                    near_fpr_mean = np.nan
                                else:
                                    near_mean = float(np.nanmean(stats_local[:, :num_methods])) * 100.0
                                    near_fpr_mean = float(np.nanmean(stats_local[:, 2*num_methods:3*num_methods])) * 100.0
                                # free stats_local and do a light GPU cleanup to reduce memory pressure
                                try:
                                    del stats_local
                                except Exception:
                                    pass
                                try:
                                    import gc
                                    gc.collect()
                                except Exception:
                                    pass
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        try:
                                            torch.cuda.empty_cache()
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            # Run test on far list (if present)
                            if not far_list:
                                far_mean = np.nan
                                far_fpr_mean = np.nan
                            else:
                                config.dataset_feat = list(far_list)
                                print(f"Datasets for testing (far): {config.dataset_feat}")
                                stats_local = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=ft)
                                if stats_local is None or stats_local.size == 0:
                                    far_mean = np.nan
                                    far_fpr_mean = np.nan
                                else:
                                    far_mean = float(np.nanmean(stats_local[:, :num_methods])) * 100.0
                                    far_fpr_mean = float(np.nanmean(stats_local[:, 2*num_methods:3*num_methods])) * 100.0
                                # free stats_local and do a light GPU cleanup to reduce memory pressure
                                try:
                                    del stats_local
                                except Exception:
                                    pass
                                try:
                                    import gc
                                    gc.collect()
                                except Exception:
                                    pass
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        try:
                                            torch.cuda.empty_cache()
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            # restore original dataset_feat for safety
                            config.dataset_feat = orig_dataset_feat

                            # If either side is NaN treat this seed as unsuccessful for this feature type
                            if np.isnan(near_mean) or np.isnan(far_mean):
                                seed_success[seed] = False

                            expansion_values[ft]['near'].append(near_mean)
                            expansion_values[ft]['far'].append(far_mean)
                            expansion_values[ft]['near_fpr'].append(near_fpr_mean)
                            expansion_values[ft]['far_fpr'].append(far_fpr_mean)
                            seed_ft_values[ft] = {
                                'near': near_mean,
                                'far': far_mean,
                                'near_fpr': near_fpr_mean,
                                'far_fpr': far_fpr_mean,
                            }
                            print(f"expansion_values: {expansion_values}")
                        except Exception as e:
                            print(f"[exp1] Error during testing feature={ft} resnet={resnet_model} exp={expansion} seed={seed}: {e}")
                            try:
                                import traceback
                                traceback.print_exc()
                            except Exception:
                                pass
                            # record failure for this seed
                            expansion_values[ft]['near'].append(np.nan)
                            expansion_values[ft]['far'].append(np.nan)
                            expansion_values[ft]['near_fpr'].append(np.nan)
                            expansion_values[ft]['far_fpr'].append(np.nan)
                            seed_ft_values[ft] = {
                                'near': np.nan,
                                'far': np.nan,
                                'near_fpr': np.nan,
                                'far_fpr': np.nan,
                            }
                            seed_success[seed] = False
                            # continue to next feature type / seed without breaking
                            continue

                except Exception as e:
                    # Unexpected error during the per-seed processing: log and mark seed as unsuccessful, but continue with other seeds
                    print(f"[exp1] Unexpected error for resnet={resnet_model} exp={expansion} seed={seed}: {e}")
                    try:
                        import traceback
                        traceback.print_exc()
                    except Exception:
                        pass
                    try:
                        import torch
                        if torch.cuda.is_available():
                            print(f"[exp1] CUDA memory allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")
                    except Exception:
                        pass
                    seed_success[seed] = False
                    # continue with next seed
                    continue

                # Persist this seed immediately so interruptions can resume from the next unfinished seed.
                try:
                    for ft in feature_types:
                        vals = seed_ft_values.get(ft)
                        entry = {'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'}
                        if isinstance(vals, dict):
                            try:
                                n = float(vals.get('near'))
                                f = float(vals.get('far'))
                                nf = float(vals.get('near_fpr'))
                                ff = float(vals.get('far_fpr'))
                                if not (np.isnan(n) or np.isnan(f) or np.isnan(nf) or np.isnan(ff)):
                                    entry = {'near': n, 'far': f, 'near_fpr': nf, 'far_fpr': ff}
                            except Exception:
                                pass
                        data_out['seed_results'][str(resnet_model)][ft][str(idx)][str(seed)] = entry
                    with open(datafile, 'w') as jf:
                        json.dump(data_out, jf, indent=2)
                except Exception as e:
                    print(f"Warning: failed to persist EX1 per-seed checkpoint for ResNet{resnet_model} E={expansion} S={seed}: {e}")

            # after processing all seeds for this expansion, release any remaining caches
            try:
                config.dataset_feat = orig_dataset_feat
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            # finalize expansion entry (set by index into the pre-populated lists)
            # Derive a missing-any flag from per-seed success tracking. If any seed failed
            # then we consider the expansion incomplete and keep it as 'not trained'.
            try:
                missing_any = not all(seed_success.get(s, False) for s in seeds) if seeds else False
            except Exception:
                missing_any = True

            for ft in feature_types:
                arr_near = np.array(expansion_values[ft]['near'], dtype=float)
                arr_far = np.array(expansion_values[ft]['far'], dtype=float)
                arr_near_fpr = np.array(expansion_values[ft]['near_fpr'], dtype=float)
                arr_far_fpr = np.array(expansion_values[ft]['far_fpr'], dtype=float)

                # If feature extraction/test failed (missing_any) or we have no valid values, mark as not trained
                if missing_any or arr_near.size == 0 or arr_far.size == 0 or (np.all(np.isnan(arr_near)) and np.all(np.isnan(arr_far))):
                    results[resnet_model][ft][idx] = {'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'}
                else:
                    # compute nan-aware means; if one side is all-NaN, keep that side as 'not trained'
                    if np.all(np.isnan(arr_near)):
                        avg_near = 'not trained'
                    else:
                        avg_near = float(np.nanmean(arr_near))

                    if np.all(np.isnan(arr_far)):
                        avg_far = 'not trained'
                    else:
                        avg_far = float(np.nanmean(arr_far))

                    if np.all(np.isnan(arr_near_fpr)):
                        avg_near_fpr = 'not trained'
                    else:
                        avg_near_fpr = float(np.nanmean(arr_near_fpr))

                    if np.all(np.isnan(arr_far_fpr)):
                        avg_far_fpr = 'not trained'
                    else:
                        avg_far_fpr = float(np.nanmean(arr_far_fpr))

                    results[resnet_model][ft][idx] = {'near': avg_near, 'far': avg_far, 'near_fpr': avg_near_fpr, 'far_fpr': avg_far_fpr}
                    print(f"Results updated for ResNet{resnet_model} expansion {expansion}: {results[resnet_model][ft][idx]}")

            # Persist progress back to the JSON file so runs can be resumed
            try:
                # Build a serializable version and write
                data_out['results'][str(resnet_model)] = data_out['results'].get(str(resnet_model), {})
                for ft in feature_types:
                    serial_list = []
                    for entry in results[resnet_model][ft]:
                        def conv(v):
                            return v if isinstance(v, str) else float(v)
                        serial_list.append({'near': conv(entry['near']), 'far': conv(entry['far']),
                                            'near_fpr': conv(entry.get('near_fpr', 'not trained')),
                                            'far_fpr': conv(entry.get('far_fpr', 'not trained'))} if isinstance(entry, dict) else {'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'})
                    data_out['results'][str(resnet_model)][ft] = serial_list
                with open(datafile, 'w') as jf:
                    json.dump(data_out, jf, indent=2)
            except Exception as e:
                print(f"Warning: failed to persist EX1 JSON after ResNet{resnet_model} E={expansion}: {e}")

    # Write combined results file with tables per resnet model (separate near / far tables)
    if config.model_type == 'spike':
        results_filename = os.path.join(out_dir, f'EX1_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_seeds.txt')
    elif config.model_type == 'conv':
        results_filename = os.path.join(out_dir, f'EX1_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_conv_seeds.txt')
    with open(results_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        if config.model_type == 'spike':
            f.write(f"Spiking ResNet Multi-Test Results\n")
        elif config.model_type == 'conv':
            f.write(f"Conv/ResNet Multi-Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
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

        for resnet_model in resnet_models:
            f.write(f"ResNet Model: {resnet_model}\n")
            f.write(f"Near OOD sets: {near_list}\n")
            f.write(f"{'Expansion':<12}{'Spike pattern':<20}{'Voltage membrane':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                sp_val = None
                volt_val = None

                # Check if 'features' exist in results
                if 'features' in results[resnet_model]:
                    entry = results[resnet_model]['features'][idx]
                    sp_val = entry['near']
                # Handle 'voltages'
                if 'voltages' in results[resnet_model]:
                    volt_val = results[resnet_model]['voltages'][idx]['near']

                def fmt(v):
                    return f"{v:>6.2f}" if isinstance(v, float) else f"{v:<20}"

                f.write(f"{expansion:<12}{(fmt(sp_val) if sp_val is not None else 'N/A'):<20}{(fmt(volt_val) if volt_val is not None else 'N/A'):<20}\n")
            f.write(f"\n")

            f.write(f"Far OOD sets: {far_list}\n")
            f.write(f"{'Expansion':<12}{'Spike pattern':<20}{'Voltage membrane':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                sp_val = None
                volt_val = None

                # Check if 'features' exist in results
                if 'features' in results[resnet_model]:
                    entry = results[resnet_model]['features'][idx]
                    sp_val = entry['far']
                # Handle 'voltages'
                if 'voltages' in results[resnet_model]:
                    volt_val = results[resnet_model]['voltages'][idx]['far']

                f.write(f"{expansion:<12}{(fmt(sp_val) if sp_val is not None else 'N/A'):<20}{(fmt(volt_val) if volt_val is not None else 'N/A'):<20}\n")

            f.write(f"{'='*60}\n\n")

    # Write FPR95 results file (same structure, uses near_fpr / far_fpr values)
    if config.model_type == 'spike':
        fpr_filename = os.path.join(out_dir, f'EX1_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_seeds_fpr95.txt')
    elif config.model_type == 'conv':
        fpr_filename = os.path.join(out_dir, f'EX1_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_conv_seeds_fpr95.txt')
    with open(fpr_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        if config.model_type == 'spike':
            f.write(f"Spiking ResNet Multi-Test Results (FPR95)\n")
        elif config.model_type == 'conv':
            f.write(f"Conv/ResNet Multi-Test Results (FPR95)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
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

        def fmt_fpr(v):
            return f"{v:>6.2f}" if isinstance(v, float) else f"{v:<20}"

        for resnet_model in resnet_models:
            f.write(f"ResNet Model: {resnet_model}\n")
            f.write(f"Near OOD sets: {near_list}\n")
            f.write(f"{'Expansion':<12}{'Spike pattern':<20}{'Voltage membrane':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                sp_val = None
                volt_val = None
                if 'features' in results[resnet_model]:
                    entry = results[resnet_model]['features'][idx]
                    sp_val = entry.get('near_fpr', 'not trained') if isinstance(entry, dict) else 'not trained'
                if 'voltages' in results[resnet_model]:
                    entry = results[resnet_model]['voltages'][idx]
                    volt_val = entry.get('near_fpr', 'not trained') if isinstance(entry, dict) else 'not trained'
                f.write(f"{expansion:<12}{(fmt_fpr(sp_val) if sp_val is not None else 'N/A'):<20}{(fmt_fpr(volt_val) if volt_val is not None else 'N/A'):<20}\n")
            f.write(f"\n")

            f.write(f"Far OOD sets: {far_list}\n")
            f.write(f"{'Expansion':<12}{'Spike pattern':<20}{'Voltage membrane':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                sp_val = None
                volt_val = None
                if 'features' in results[resnet_model]:
                    entry = results[resnet_model]['features'][idx]
                    sp_val = entry.get('far_fpr', 'not trained') if isinstance(entry, dict) else 'not trained'
                if 'voltages' in results[resnet_model]:
                    entry = results[resnet_model]['voltages'][idx]
                    volt_val = entry.get('far_fpr', 'not trained') if isinstance(entry, dict) else 'not trained'
                f.write(f"{expansion:<12}{(fmt_fpr(sp_val) if sp_val is not None else 'N/A'):<20}{(fmt_fpr(volt_val) if volt_val is not None else 'N/A'):<20}\n")

            f.write(f"{'='*60}\n\n")

    # Save plotting data (results) so plots can be regenerated without re-extraction
    data_out = {
        'dataset_ID': config.dataset_ID,
        'case': config.case,
        'expansions': list(expansions),
        'feature_types': feature_types,
        'model_tags': {str(k): v for k, v in model_tags.items()},
        'results': {}
    }
    for rm in resnet_models:
        data_out['results'][str(rm)] = {}
        for ft in feature_types:
            serial_list = []
            for entry in results[rm][ft]:
                def conv(v):
                    return v if isinstance(v, str) else float(v)
                serial_list.append({'near': conv(entry['near']), 'far': conv(entry['far']),
                                    'near_fpr': conv(entry.get('near_fpr', 'not trained')),
                                    'far_fpr': conv(entry.get('far_fpr', 'not trained'))})
            data_out['results'][str(rm)][ft] = serial_list

    if config.model_type == 'spike':
        datafile = os.path.join(out_dir, f'EX1_{config.dataset_ID}_A_{config.auto_aug}_L_{config.loss}_data.json')
    elif config.model_type == 'conv':
        datafile = os.path.join(out_dir, f'EX1_{config.dataset_ID}_A_{config.auto_aug}_L_{config.loss}_conv_data.json')
    try:
        with open(datafile, 'w') as jf:
            json.dump(data_out, jf, indent=2)
    except Exception as e:
        print(f"Warning: failed to write plotting data JSON: {e}")

    # Generate plots for features and voltages: each plot contains near and far curves per ResNet
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'figure.titlesize': 20,
    })

    # Plot expansions on evenly spaced positions (labels are the expansion values)
    x = list(expansions)
    positions = np.arange(len(x))
    for ft in feature_types:
        if ft == 'features':
            ft_label = 'Spike pattern'
        elif ft == 'voltages':
            ft_label = 'Membrane voltage'
        plt.figure(figsize=(10,7))
        for resnet_model in resnet_models:
            vals = results[resnet_model][ft]
            y_near = [np.nan if vals[i]=='not trained' or vals[i]['near']=='not trained' else vals[i]['near'] for i in range(len(expansions))]
            y_far = [np.nan if vals[i]=='not trained' or vals[i]['far']=='not trained' else vals[i]['far'] for i in range(len(expansions))]
            # Plot on evenly spaced x positions so spacing is uniform regardless of numeric expansion values
            tag = model_tags.get(resnet_model, f'ResNet{resnet_model}')
            plt.plot(positions, y_near, marker='o', linestyle='-', label=f'{tag} - near')
            plt.plot(positions, y_far, marker='x', linestyle='--', label=f'{tag} - far')

        plt.xlabel('Expansion')
        plt.ylabel('Average AUROC')
        plt.title(f'Average AUROC vs Expansion ({ft_label})')

        # Force consistent y-axis across all plots for visual comparison
        y_min, y_max = 50.0, 100.0
        plt.ylim(y_min, y_max)

        # Y ticks every 5 units
        yticks = np.arange(y_min, y_max + 1, 5)
        plt.yticks(yticks)

        # Dashed grid on both axes (horizontal and vertical) but not dense
        plt.grid(axis='both', linestyle='--', linewidth=0.8)

        # Set x ticks evenly and label them with the expansion values
        plt.xticks(positions, x)

        # Legend with requested font size
        plt.legend(ncol=2, fontsize=18)

        if config.model_type == 'spike':
            plotfile = os.path.join(out_dir, f'EX1_{config.dataset_ID}_{ft}_A_{config.auto_aug}_L_{config.loss}.png')
        elif config.model_type == 'conv':
            plotfile = os.path.join(out_dir, f'EX1_{config.dataset_ID}_{ft}_A_{config.auto_aug}_L_{config.loss}_conv.png')
        plt.savefig(plotfile, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nDetailed statistics and plots saved to: {results_filename} and {out_dir}")


def statistics_exp_2(config, seeds, expansions, resnet_models):
    """
    this function should calculate the statistics similar as in statistics_exp_1,
    this time the feature set is extended 
    feature_types = ['features', 'voltages', 'spikes', 'probs']
    also, the distance methods are extended 
    config.methods = ['ASH', 'MSP', 'ODIN', 'ENGY', 'MLS', 'VIM']
    this function needs to go over all the feature types and distance methods 
    and to compute the averaged auroc over all seed values, for each expansion and each resnet model.
    finally, it needs to save the results in a txt file in tales format as in statistics_exp_1, 
    """

    # Feature types to evaluate
    # feature_types = ['features', 'voltages', 'spikes', 'probs']
    # Methods to evaluate (fall back to a sensible default if not set)
    methods = getattr(config, 'methods', ['ASH', 'MSP', 'ODIN', 'ENGY', 'MLS', 'VIM'])
    yaml_override_feature_extraction = bool(getattr(config, 'override_feature_extraction', False))

    out_dir = os.path.join('results', 'ex_2', 'exp'+str(config.case))
    os.makedirs(out_dir, exist_ok=True)

    # Prepare near and far lists (do not filter by dataset_ID)
    near_list = list(getattr(config, 'near_ood', []) or [])
    far_list = list(getattr(config, 'far_ood', []) or [])

    # Save original dataset_feat to restore later
    orig_dataset_feat = getattr(config, 'dataset_feat', None)

    # Map resnet model numbers to human-friendly tags for potential plotting
    if config.model_type == 'spike':
        feature_types = ['features', 'voltages', 'spikes', 'probs']
        model_tags = {4: 'spike-Conv', 10: 'spike-ResNet10', 18: 'spike-ResNet18'}
    elif config.model_type == 'conv':
        feature_types = ['voltages', 'probs']
        model_tags = {4: 'Conv', 10: 'ResNet10', 18: 'ResNet18'}

    # Results container: results[resnet_model][feature_type] = list over expansions (dict with 'near'/'far' values or 'not trained')
    # Load existing EX2 JSON if present so we can resume partially-completed runs. If not present, create skeleton.
    datafile = os.path.join(out_dir, f'EX2_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}.json')
    data_out = None

    if config.model_type == 'spike':
        datafile = os.path.join(out_dir, f'EX2_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}.json')
    elif config.model_type == 'conv':
        datafile = os.path.join(out_dir, f'EX2_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}_conv.json')
    data_out = None
    if os.path.exists(datafile):
        try:
            with open(datafile, 'r') as jf:
                data_out = json.load(jf)
        except Exception:
            data_out = None

    if data_out is None:
        data_out = {
            'dataset_ID': config.dataset_ID,
            'case': config.case,
            'expansions': list(expansions),
            'methods': methods,
            'feature_types': feature_types,
            'model_tags': {str(k): v for k, v in model_tags.items()},
            'results': {}
        }
        for rm in resnet_models:
            data_out['results'][str(rm)] = {}
            for ft in feature_types:
                data_out['results'][str(rm)][ft] = [
                    {'near': 'not trained', 'far': 'not trained',
                     'near_fpr': 'not trained', 'far_fpr': 'not trained'}
                    for _ in expansions
                ]
        try:
            with open(datafile, 'w') as jf:
                json.dump(data_out, jf, indent=2)
        except Exception as e:
            print(f"Warning: failed to write initial EX2 JSON skeleton: {e}")

    # Ensure per-seed checkpoint structure exists so interrupted runs can resume from the last completed seed.
    data_out.setdefault('seed_results', {})
    for rm in resnet_models:
        rm_key = str(rm)
        data_out['seed_results'].setdefault(rm_key, {})
        for ft in feature_types:
            data_out['seed_results'][rm_key].setdefault(ft, {})
            for i in range(len(expansions)):
                data_out['seed_results'][rm_key][ft].setdefault(str(i), {})

    # Initialize results from the JSON skeleton so we can skip completed experiments
    results = {}
    for rm in resnet_models:
        results[rm] = {}
        key = str(rm)
        for ft in feature_types:
            if key in data_out.get('results', {}) and ft in data_out['results'][key]:
                serial = data_out['results'][key][ft]
                entries = []
                for i in range(len(expansions)):
                    if i < len(serial):
                        entries.append(serial[i])
                    else:
                        entries.append({'near': 'not trained', 'far': 'not trained'})
                results[rm][ft] = entries
            else:
                results[rm][ft] = [{'near': 'not trained', 'far': 'not trained'} for _ in expansions]

    num_methods = len(methods)

    # Loop over resnet models and expansions
    for resnet_model in resnet_models:
        config.resnet_model = resnet_model
        for idx, expansion in enumerate(expansions):
            config.expansion = expansion
            config.override_feature_extraction = yaml_override_feature_extraction
            # If this resnet+expansion is already present in the JSON for all feature types (near and far filled), skip it
            already_done = True
            for ft in feature_types:
                entry = results[resnet_model][ft][idx]
                near_val = entry.get('near') if isinstance(entry, dict) else 'not trained'
                far_val = entry.get('far') if isinstance(entry, dict) else 'not trained'
                if isinstance(near_val, str) and near_val == 'not trained':
                    already_done = False
                    break
                if near_val is None:
                    already_done = False
                    break
                if isinstance(far_val, str) and far_val == 'not trained':
                    already_done = False
                    break
                if far_val is None:
                    already_done = False
                    break
            if already_done:
                print(f"Skipping ResNet{resnet_model} expansion {expansion} — already present in EX2 JSON.")
                continue

            # For each feature type compute per-seed AUROC and then average across seeds separately for near and far
            expansion_values = {ft: {'near': [], 'far': [], 'near_fpr': [], 'far_fpr': []} for ft in feature_types}
            missing_any = False

            def _valid_method_list(v):
                if not isinstance(v, list) or len(v) != num_methods:
                    return False
                try:
                    for x in v:
                        if x is None or np.isnan(float(x)):
                            return False
                    return True
                except Exception:
                    return False

            # Preload completed seeds for this expansion from checkpoint.
            completed_seeds = set()
            for seed in seeds:
                seed_done = True
                for ft in feature_types:
                    cached = data_out['seed_results'][str(resnet_model)][ft][str(idx)].get(str(seed))
                    if not isinstance(cached, dict):
                        seed_done = False
                        break
                    near_c = cached.get('near')
                    far_c = cached.get('far')
                    near_fpr_c = cached.get('near_fpr')
                    far_fpr_c = cached.get('far_fpr')
                    if _valid_method_list(near_c) and _valid_method_list(far_c) and _valid_method_list(near_fpr_c) and _valid_method_list(far_fpr_c):
                        expansion_values[ft]['near'].append([float(x) for x in near_c])
                        expansion_values[ft]['far'].append([float(x) for x in far_c])
                        expansion_values[ft]['near_fpr'].append([float(x) for x in near_fpr_c])
                        expansion_values[ft]['far_fpr'].append([float(x) for x in far_fpr_c])
                    else:
                        seed_done = False
                        break
                if seed_done:
                    completed_seeds.add(seed)

            for seed in seeds:
                if seed in completed_seeds:
                    print(f"[exp2] Skipping completed seed: ResNet{resnet_model} E={expansion} S={seed}")
                    continue
                config.seed = seed
                seed_ft_values = {}
                try:
                    # For efficiency, extract features once for the union of near and far OOD sets
                    union_list = []
                    try:
                        union_list = sorted(set(list(near_list) + list(far_list)))
                    except Exception:
                        union_list = list(near_list) + list(far_list)

                    extracted_ok = True
                    if union_list:
                        try:
                            config.dataset_feat = list(union_list)
                            print(f"[exp2] Extracting features for union list: {config.dataset_feat} (ResNet{resnet_model}, E={expansion}, S={seed})")
                            if config.model_type == 'spike':
                                feature_extraction_spike(config)
                            elif config.model_type == 'conv':
                                feature_extraction_conv(config)
                        except Exception as e:
                            print(f"[exp2] Warning: feature extraction failed for ResNet{resnet_model} E={expansion} S={seed}: {e}")
                            extracted_ok = False

                    for ft in feature_types:
                        try:
                            if not extracted_ok:
                                # record missing seed as None so we can filter later
                                expansion_values[ft]['near'].append(None)
                                expansion_values[ft]['far'].append(None)
                                continue

                            # Run test on near list (if present)
                            if not near_list:
                                near_mean = None
                                near_mean_fpr = None
                            else:
                                config.dataset_feat = list(near_list)
                                print(f"[exp2] Datasets for testing (near): {config.dataset_feat} (feature={ft})")
                                stats_local = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=ft)
                                if stats_local is None or stats_local.size == 0:
                                    near_mean = None
                                    near_mean_fpr = None
                                else:
                                    # compute per-method AUROC averaged across the listed OOD datasets
                                    per_method = np.nanmean(stats_local[:, :num_methods], axis=0)
                                    near_mean = (per_method * 100.0).tolist()
                                    # compute per-method FPR95 averaged across the listed OOD datasets
                                    per_method_fpr = np.nanmean(stats_local[:, 2*num_methods:3*num_methods], axis=0)
                                    near_mean_fpr = (per_method_fpr * 100.0).tolist()

                            # Run test on far list (if present)
                            if not far_list:
                                far_mean = None
                                far_mean_fpr = None
                            else:
                                config.dataset_feat = list(far_list)
                                print(f"[exp2] Datasets for testing (far): {config.dataset_feat} (feature={ft})")
                                stats_local = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods, features=ft)
                                if stats_local is None or stats_local.size == 0:
                                    far_mean = None
                                    far_mean_fpr = None
                                else:
                                    per_method = np.nanmean(stats_local[:, :num_methods], axis=0)
                                    far_mean = (per_method * 100.0).tolist()
                                    per_method_fpr = np.nanmean(stats_local[:, 2*num_methods:3*num_methods], axis=0)
                                    far_mean_fpr = (per_method_fpr * 100.0).tolist()

                            # restore original dataset_feat for safety
                            config.dataset_feat = orig_dataset_feat

                            expansion_values[ft]['near'].append(near_mean)
                            expansion_values[ft]['far'].append(far_mean)
                            expansion_values[ft]['near_fpr'].append(near_mean_fpr)
                            expansion_values[ft]['far_fpr'].append(far_mean_fpr)
                            seed_ft_values[ft] = {
                                'near': near_mean,
                                'far': far_mean,
                                'near_fpr': near_mean_fpr,
                                'far_fpr': far_mean_fpr,
                            }
                        except Exception as e:
                            print(f"[exp2] Error during testing feature={ft} resnet={resnet_model} exp={expansion} seed={seed}: {e}")
                            missing_any = True
                            seed_ft_values[ft] = {'near': None, 'far': None, 'near_fpr': None, 'far_fpr': None}
                            break

                # after finishing per-feature tests for this seed, do NOT remove on-disk .npz files here
                # (keep saved features for later reproducibility). We only perform in-memory
                # cleanup and GPU cache clearing in the finally block below to reduce peak memory usage.
                except Exception as e:
                    # Any unexpected error during the per-seed processing is handled here.
                    print(f"[exp2] Unexpected error for resnet={resnet_model} exp={expansion} seed={seed}: {e}")
                    missing_any = True
                    break
                finally:
                    # free Python and GPU caches regardless of success/failure
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                # Persist this seed immediately so interruptions can resume from the next unfinished seed.
                try:
                    for ft in feature_types:
                        vals = seed_ft_values.get(ft)
                        entry = {'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'}
                        if isinstance(vals, dict):
                            n = vals.get('near')
                            f = vals.get('far')
                            nf = vals.get('near_fpr')
                            ff = vals.get('far_fpr')
                            if _valid_method_list(n) and _valid_method_list(f) and _valid_method_list(nf) and _valid_method_list(ff):
                                entry = {
                                    'near': [float(x) for x in n],
                                    'far': [float(x) for x in f],
                                    'near_fpr': [float(x) for x in nf],
                                    'far_fpr': [float(x) for x in ff],
                                }
                        data_out['seed_results'][str(resnet_model)][ft][str(idx)][str(seed)] = entry
                    with open(datafile, 'w') as jf:
                        json.dump(data_out, jf, indent=2)
                except Exception as e:
                    print(f"Warning: failed to persist EX2 per-seed checkpoint for ResNet{resnet_model} E={expansion} S={seed}: {e}")

            # finalize expansion entry
            for ft in feature_types:
                # expansion_values[ft]['near'] and ['far'] contain one entry per seed (or None)
                seed_near = expansion_values[ft]['near']
                seed_far = expansion_values[ft]['far']
                seed_near_fpr = expansion_values[ft]['near_fpr']
                seed_far_fpr = expansion_values[ft]['far_fpr']

                # Require that ALL seeds produced valid (non-None) results for both near and far
                all_near_ok = (len(seed_near) == len(seeds)) and all(v is not None for v in seed_near)
                all_far_ok = (len(seed_far) == len(seeds)) and all(v is not None for v in seed_far)

                if not (all_near_ok and all_far_ok):
                    # at least one seed missing or incomplete -> keep as not trained
                    results[resnet_model][ft][idx] = {
                        'near': 'not trained', 'far': 'not trained',
                        'near_fpr': 'not trained', 'far_fpr': 'not trained',
                    }
                    continue

                # All seeds present: stack and average across seeds (nan-aware) per method
                def avg_over_seeds(seed_list):
                    try:
                        stacked = np.stack([np.array(x, dtype=float) for x in seed_list], axis=0)
                        mean_methods = np.nanmean(stacked, axis=0)
                        return [float(x) for x in mean_methods]
                    except Exception as e:
                        print(f"[exp2] Error averaging seeds for ft={ft}: {e}")
                        return 'not trained'

                avg_near = avg_over_seeds(seed_near)
                avg_far = avg_over_seeds(seed_far)
                all_near_fpr_ok = (len(seed_near_fpr) == len(seeds)) and all(v is not None for v in seed_near_fpr)
                all_far_fpr_ok = (len(seed_far_fpr) == len(seeds)) and all(v is not None for v in seed_far_fpr)
                avg_near_fpr = avg_over_seeds(seed_near_fpr) if all_near_fpr_ok else 'not trained'
                avg_far_fpr = avg_over_seeds(seed_far_fpr) if all_far_fpr_ok else 'not trained'
                results[resnet_model][ft][idx] = {
                    'near': avg_near, 'far': avg_far,
                    'near_fpr': avg_near_fpr, 'far_fpr': avg_far_fpr,
                }

            # Persist progress back to EX2 JSON after each expansion so runs are resumable
            try:
                data_out['results'][str(resnet_model)] = data_out['results'].get(str(resnet_model), {})
                # persist which methods were used for this EX2 run
                data_out['methods'] = methods
                for ft in feature_types:
                    serial_list = []
                    for entry in results[resnet_model][ft]:
                        def conv(v):
                            if isinstance(v, str):
                                return v
                            if v is None:
                                return None
                            try:
                                return [float(x) for x in v]
                            except Exception:
                                return v
                        serial_list.append({'near': conv(entry['near']), 'far': conv(entry['far']),
                                            'near_fpr': conv(entry.get('near_fpr', 'not trained')),
                                            'far_fpr': conv(entry.get('far_fpr', 'not trained'))})
                    data_out['results'][str(resnet_model)][ft] = serial_list
                with open(datafile, 'w') as jf:
                    json.dump(data_out, jf, indent=2)
            except Exception as e:
                print(f"Warning: failed to persist EX2 JSON after ResNet{resnet_model} E={expansion}: {e}")

    # Write combined results file with tables per resnet model (separate near / far tables)
    if config.model_type == 'spike':
        results_filename = os.path.join(out_dir, f'EX2_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_seeds.txt')
    elif config.model_type == 'conv':
        results_filename = os.path.join(out_dir, f'EX2_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_conv_seeds.txt')
    with open(results_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Experiment-2 Multi-Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Methods considered: {', '.join(methods)}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")

        for resnet_model in resnet_models:
            f.write(f"ResNet Model: {resnet_model}\n")
            f.write(f"Near OOD sets: {near_list}\n")

            # For each feature type, print a table with expansions as rows and methods as columns (Near)
            for ft in feature_types:
                f.write(f"\nFeature Type (NEAR): {ft}\n")
                # Header: Expansion then each method
                header = 'Expansion'.ljust(12) + ''.join([f"{m:>10s}" for m in methods]) + '\n'
                f.write(header)
                f.write('-' * (12 + 10 * len(methods)) + '\n')
                for idx, expansion in enumerate(expansions):
                    entry = results[resnet_model][ft][idx]
                    val = entry['near'] if isinstance(entry, dict) else entry
                    if isinstance(val, str):
                        # not trained
                        row = f"{expansion:<12}{val}\n"
                    elif val is None:
                        cols = ''.join([f"{'missing':>10s}" for _ in methods])
                        row = f"{expansion:<12}{cols}\n"
                    else:
                        # val should be a list of per-method floats
                        try:
                            cols = ''.join([f"{(v if v is not None else float('nan')):10.2f}" for v in val])
                        except Exception:
                            # fallback to a simple string
                            cols = ' '.join([str(v) for v in val])
                        row = f"{expansion:<12}{cols}\n"
                    f.write(row)

            f.write('\n')
            f.write(f"Far OOD sets: {far_list}\n")

            # For each feature type, print a table with expansions as rows and methods as columns (Far)
            for ft in feature_types:
                f.write(f"\nFeature Type (FAR): {ft}\n")
                header = 'Expansion'.ljust(12) + ''.join([f"{m:>10s}" for m in methods]) + '\n'
                f.write(header)
                f.write('-' * (12 + 10 * len(methods)) + '\n')
                for idx, expansion in enumerate(expansions):
                    entry = results[resnet_model][ft][idx]
                    val = entry['far'] if isinstance(entry, dict) else entry
                    if isinstance(val, str):
                        row = f"{expansion:<12}{val}\n"
                    elif val is None:
                        cols = ''.join([f"{'missing':>10s}" for _ in methods])
                        row = f"{expansion:<12}{cols}\n"
                    else:
                        try:
                            cols = ''.join([f"{(v if v is not None else float('nan')):10.2f}" for v in val])
                        except Exception:
                            cols = ' '.join([str(v) for v in val])
                        row = f"{expansion:<12}{cols}\n"
                    f.write(row)

            f.write(f"{'='*60}\n\n")

    # Save plotting data (results) so plots can be regenerated without re-extraction
    data_out = {
        'dataset_ID': config.dataset_ID,
        'case': config.case,
        'expansions': list(expansions),
        'methods': methods,
        'feature_types': feature_types,
        'model_tags': {str(k): v for k, v in model_tags.items()},
        'results': {}
    }
    for rm in resnet_models:
        data_out['results'][str(rm)] = {}
        for ft in feature_types:
            serial_list = []
            for entry in results[rm][ft]:
                def conv(v):
                    # pass strings through, convert lists to simple lists of floats
                    if isinstance(v, str):
                        return v
                    if v is None:
                        return None
                    try:
                        return [float(x) for x in v]
                    except Exception:
                        return v
                serial_list.append({'near': conv(entry['near']), 'far': conv(entry['far']),
                                    'near_fpr': conv(entry.get('near_fpr', 'not trained')),
                                    'far_fpr': conv(entry.get('far_fpr', 'not trained'))})
            data_out['results'][str(rm)][ft] = serial_list

    if config.model_type == 'spike':
        datafile = os.path.join(out_dir, f'EX2_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}.json')
    elif config.model_type == 'conv':
        datafile = os.path.join(out_dir, f'EX2_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}_conv.json')
    try:
        with open(datafile, 'w') as jf:
            json.dump(data_out, jf, indent=2)
    except Exception as e:
        print(f"Warning: failed to write EX2 plotting data JSON: {e}")

    # # Generate plots for each feature type: each plot contains near and far curves per ResNet
    # plt.rcParams.update({
    #     'text.usetex': True,
    #     'font.family': 'serif',
    #     'font.serif': ['Times New Roman'],
    #     'mathtext.fontset': 'stix',
    #     'axes.titlesize': 20,
    #     'axes.labelsize': 20,
    #     'xtick.labelsize': 18,
    #     'ytick.labelsize': 18,
    #     'figure.titlesize': 20,
    # })

    # x = list(expansions)
    # positions = np.arange(len(x))
    # for ft in feature_types:
    #     if ft == 'features':
    #         ft_label = 'Spike pattern'
    #     elif ft == 'voltages':
    #         ft_label = 'Membrane voltage'
    #     elif ft == 'spikes':
    #         ft_label = 'Spikes'
    #     else:
    #         ft_label = 'Probabilities'

    #     plt.figure(figsize=(10,7))
    #     for resnet_model in resnet_models:
    #         vals = results[resnet_model][ft]
    #         y_near = [np.nan if vals[i]=='not trained' or vals[i]['near']=='not trained' else vals[i]['near'] for i in range(len(expansions))]
    #         y_far = [np.nan if vals[i]=='not trained' or vals[i]['far']=='not trained' else vals[i]['far'] for i in range(len(expansions))]
    #         tag = model_tags.get(resnet_model, f'ResNet{resnet_model}')
    #         plt.plot(positions, y_near, marker='o', linestyle='-', label=f'{tag} - near')
    #         plt.plot(positions, y_far, marker='x', linestyle='--', label=f'{tag} - far')

    #     plt.xlabel('Expansion')
    #     plt.ylabel('Average AUROC')
    #     plt.title(f'Average AUROC vs Expansion ({ft_label})')

    #     y_min, y_max = 50.0, 100.0
    #     plt.ylim(y_min, y_max)
    #     plt.yticks(np.arange(y_min, y_max + 1, 5))
    #     plt.grid(axis='both', linestyle='--', linewidth=0.8)
    #     plt.xticks(positions, x)
    #     plt.legend(ncol=2, fontsize=18)

    #     plotfile = os.path.join(out_dir, f'EX2_{config.dataset_ID}_{ft}_L_{config.loss}.png')
    #     plt.savefig(plotfile, dpi=300, bbox_inches='tight')
    #     plt.close()

    # Write FPR95 results file with the same structure
    if config.model_type == 'spike':
        fpr_filename = os.path.join(out_dir, f'EX2_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_seeds_fpr95.txt')
    elif config.model_type == 'conv':
        fpr_filename = os.path.join(out_dir, f'EX2_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_conv_seeds_fpr95.txt')
    with open(fpr_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Spiking ResNet Experiment-2 Multi-Test Results (FPR95)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Methods considered: {', '.join(methods)}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")

        def _write_fpr_table(f, resnet_model, ft, ood_label, key):
            f.write(f"\nFeature Type ({ood_label}): {ft}\n")
            header = 'Expansion'.ljust(12) + ''.join([f"{m:>10s}" for m in methods]) + '\n'
            f.write(header)
            f.write('-' * (12 + 10 * len(methods)) + '\n')
            for idx, expansion in enumerate(expansions):
                entry = results[resnet_model][ft][idx]
                val = entry.get(key) if isinstance(entry, dict) else None
                if val is None or (isinstance(val, str) and val == 'not trained'):
                    row = f"{expansion:<12}{str(val or 'not trained')}\n"
                else:
                    try:
                        cols = ''.join([f"{(v if v is not None else float('nan')):10.2f}" for v in val])
                    except Exception:
                        cols = ' '.join([str(v) for v in val])
                    row = f"{expansion:<12}{cols}\n"
                f.write(row)

        for resnet_model in resnet_models:
            f.write(f"ResNet Model: {resnet_model}\n")
            f.write(f"Near OOD sets: {near_list}\n")
            for ft in feature_types:
                _write_fpr_table(f, resnet_model, ft, 'NEAR', 'near_fpr')
            f.write('\n')
            f.write(f"Far OOD sets: {far_list}\n")
            for ft in feature_types:
                _write_fpr_table(f, resnet_model, ft, 'FAR', 'far_fpr')
            f.write(f"{'='*60}\n\n")

    print(f"\nExperiment-2 detailed statistics and plots saved to: {results_filename} and {out_dir}")
    print(f"FPR95 statistics saved to: {fpr_filename}")


def statistics_exp_3(config, seeds, expansions, resnet_models):
    """Experiment-3: combine EX1 and EX2 with a single extraction pass per seed.

    - KNN branch: uses `voltages` representation with method ['KNN'].
    - VIM branch: uses scoring-based path with method ['VIM'] on `probs`.
    - The current paper uses the EX3 setup with voltages for KNN and probs for VIM.
    - Writes resumable JSON checkpoints (including per-seed values) and TXT summaries.
    - Does not generate plots.
    """
    methods_1 = getattr(config, 'methods_1', None)
    methods_2 = getattr(config, 'methods_2', None)
    if not isinstance(methods_1, list) or len(methods_1) == 0:
        methods_1 = ['KNN']
    if not isinstance(methods_2, list) or len(methods_2) == 0:
        methods_2 = ['VIM']

    # EX3 couples EX1 and EX2 branches: methods_1 run on `voltages`, methods_2 on `probs`.
    feature_method_map = {
        'voltages': list(methods_1),
        'probs': list(methods_2),
    }

    def _method_alias(name):
        # Normalize method key names for JSON fields.
        return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name)).strip('_')

    # Dynamic result aliases driven by YAML methods, e.g. KNN -> knn, VIM -> vim.
    method_alias_to_branch = {}
    for m in methods_1:
        method_alias_to_branch[_method_alias(m)] = 'voltages'
    for m in methods_2:
        method_alias_to_branch[_method_alias(m)] = 'probs'

    branch_to_aliases = {'voltages': [], 'probs': []}
    for alias, branch in method_alias_to_branch.items():
        branch_to_aliases.setdefault(branch, []).append(alias)
    yaml_override_feature_extraction = bool(getattr(config, 'override_feature_extraction', False))

    out_dir = os.path.join('results', 'ex_3', 'exp' + str(config.case))
    os.makedirs(out_dir, exist_ok=True)

    near_list = list(getattr(config, 'near_ood', []) or [])
    far_list = list(getattr(config, 'far_ood', []) or [])
    orig_dataset_feat = getattr(config, 'dataset_feat', None)

    if config.model_type == 'spike':
        model_tags = {4: 'spike-Conv', 10: 'spike-ResNet10', 18: 'spike-ResNet18'}
    elif config.model_type == 'conv':
        model_tags = {4: 'Conv', 10: 'ResNet10', 18: 'ResNet18'}
    else:
        model_tags = {}

    if config.model_type == 'spike':
        datafile = os.path.join(out_dir, f'EX3_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}.json')
    else:
        datafile = os.path.join(out_dir, f'EX3_{config.dataset_ID}_L_{config.loss}_A_{config.auto_aug}_conv.json')

    data_out = None
    if os.path.exists(datafile):
        try:
            with open(datafile, 'r') as jf:
                data_out = json.load(jf)
        except Exception:
            data_out = None

    def _empty_entry():
        return {'near': 'not trained', 'far': 'not trained', 'near_fpr': 'not trained', 'far_fpr': 'not trained'}

    if data_out is None:
        data_out = {
            'dataset_ID': config.dataset_ID,
            'case': config.case,
            'expansions': list(expansions),
            'methods_1': list(methods_1),
            'methods_2': list(methods_2),
            'feature_method_map': feature_method_map,
            'model_tags': {str(k): v for k, v in model_tags.items()},
            'results': {},
        }
        for rm in resnet_models:
            data_out['results'][str(rm)] = {}
            for ft in feature_method_map:
                data_out['results'][str(rm)][ft] = [_empty_entry() for _ in expansions]
            # Explicit method-specific aliases requested for EX3 consumers (dynamic from YAML)
            for alias in method_alias_to_branch:
                data_out['results'][str(rm)][alias] = [_empty_entry() for _ in expansions]
        try:
            with open(datafile, 'w') as jf:
                json.dump(data_out, jf, indent=2)
        except Exception as e:
            print(f"Warning: failed to write initial EX3 JSON skeleton: {e}")

    # Do not reuse legacy `features` checkpoints for EX3 KNN.
    # This paper defines EX3 KNN on `voltages`, so older feature-based runs must be recomputed.

    data_out.setdefault('seed_results', {})
    for rm in resnet_models:
        rm_key = str(rm)
        data_out['seed_results'].setdefault(rm_key, {})
        for ft in feature_method_map:
            data_out['seed_results'][rm_key].setdefault(ft, {})
            for i in range(len(expansions)):
                data_out['seed_results'][rm_key][ft].setdefault(str(i), {})
        # Keep per-seed method aliases too, so resume can continue from explicit method fields.
        for alias in method_alias_to_branch:
            data_out['seed_results'][rm_key].setdefault(alias, {})
            for i in range(len(expansions)):
                data_out['seed_results'][rm_key][alias].setdefault(str(i), {})

    results = {}
    for rm in resnet_models:
        results[rm] = {}
        rm_key = str(rm)
        for ft in feature_method_map:
            serial = data_out.get('results', {}).get(rm_key, {}).get(ft, [])
            entries = []
            for i in range(len(expansions)):
                if i < len(serial) and isinstance(serial[i], dict):
                    entries.append(serial[i])
                else:
                    entries.append(_empty_entry())
            results[rm][ft] = entries

        # Keep explicit method keys in-memory as aliases (driven by methods_1/methods_2)
        for alias, branch in method_alias_to_branch.items():
            alias_serial = data_out.get('results', {}).get(rm_key, {}).get(alias, [])
            alias_entries = []
            for i in range(len(expansions)):
                if i < len(alias_serial) and isinstance(alias_serial[i], dict):
                    alias_entries.append(alias_serial[i])
                elif i < len(results[rm][branch]):
                    alias_entries.append(results[rm][branch][i])
                else:
                    alias_entries.append(_empty_entry())
            results[rm][alias] = alias_entries

    for resnet_model in resnet_models:
        config.resnet_model = resnet_model
        print(f"[exp3] Processing ResNet{resnet_model}")
        for idx, expansion in enumerate(expansions):
            config.expansion = expansion
            config.override_feature_extraction = yaml_override_feature_extraction

            already_done = True
            for ft in feature_method_map:
                entry = results[resnet_model][ft][idx]
                near_v = entry.get('near') if isinstance(entry, dict) else 'not trained'
                far_v = entry.get('far') if isinstance(entry, dict) else 'not trained'
                if (isinstance(near_v, str) and near_v == 'not trained') or near_v is None:
                    already_done = False
                    break
                if (isinstance(far_v, str) and far_v == 'not trained') or far_v is None:
                    already_done = False
                    break
            if already_done:
                print(f"[exp3] Skipping ResNet{resnet_model} expansion {expansion} - already present in EX3 JSON.")
                continue

            expansion_values = {ft: {'near': [], 'far': [], 'near_fpr': [], 'far_fpr': []} for ft in feature_method_map}

            def _valid_scalar_entry(v):
                if not isinstance(v, dict):
                    return False
                try:
                    vals = [float(v.get('near')), float(v.get('far')), float(v.get('near_fpr')), float(v.get('far_fpr'))]
                    return not any(np.isnan(x) for x in vals)
                except Exception:
                    return False

            completed_seeds = set()

            def _get_seed_cached_entry(rm_value, feature_key, exp_idx, seed_value):
                # Prefer canonical branch key, then fall back to any alias of that branch.
                seed_obj = data_out['seed_results'].get(str(rm_value), {})
                v = seed_obj.get(feature_key, {}).get(str(exp_idx), {}).get(str(seed_value))
                if _valid_scalar_entry(v):
                    return v
                for alias in branch_to_aliases.get(feature_key, []):
                    v2 = seed_obj.get(alias, {}).get(str(exp_idx), {}).get(str(seed_value))
                    if _valid_scalar_entry(v2):
                        return v2
                return None

            for seed in seeds:
                seed_done = True
                for ft in feature_method_map:
                    cached = _get_seed_cached_entry(resnet_model, ft, idx, seed)
                    if cached is not None:
                        expansion_values[ft]['near'].append(float(cached['near']))
                        expansion_values[ft]['far'].append(float(cached['far']))
                        expansion_values[ft]['near_fpr'].append(float(cached['near_fpr']))
                        expansion_values[ft]['far_fpr'].append(float(cached['far_fpr']))
                    else:
                        seed_done = False
                        break
                if seed_done:
                    completed_seeds.add(seed)

            for seed in seeds:
                if seed in completed_seeds:
                    print(f"[exp3] Skipping completed seed: ResNet{resnet_model} E={expansion} S={seed}")
                    continue

                config.seed = seed
                seed_ft_values = {}

                try:
                    union_list = sorted(set(list(near_list) + list(far_list))) if (near_list or far_list) else []
                except Exception:
                    union_list = list(near_list) + list(far_list)

                extracted_ok = True
                if union_list:
                    try:
                        config.dataset_feat = list(union_list)
                        print(f"[exp3] Extracting once for union list: {config.dataset_feat} (ResNet{resnet_model}, E={expansion}, S={seed})")
                        if config.model_type == 'spike':
                            feature_extraction_spike(config)
                        elif config.model_type == 'conv':
                            feature_extraction_conv(config)
                    except Exception as e:
                        print(f"[exp3] Warning: feature extraction failed for ResNet{resnet_model} E={expansion} S={seed}: {e}")
                        extracted_ok = False

                for ft, methods_local in feature_method_map.items():
                    if not extracted_ok:
                        seed_ft_values[ft] = {'near': np.nan, 'far': np.nan, 'near_fpr': np.nan, 'far_fpr': np.nan}
                        expansion_values[ft]['near'].append(np.nan)
                        expansion_values[ft]['far'].append(np.nan)
                        expansion_values[ft]['near_fpr'].append(np.nan)
                        expansion_values[ft]['far_fpr'].append(np.nan)
                        continue

                    num_methods = len(methods_local)
                    try:
                        if not near_list:
                            near_mean = np.nan
                            near_fpr = np.nan
                        else:
                            config.dataset_feat = list(near_list)
                            stats_local = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods_local, features=ft)
                            if stats_local is None or stats_local.size == 0:
                                near_mean = np.nan
                                near_fpr = np.nan
                            else:
                                near_mean = float(np.nanmean(stats_local[:, :num_methods])) * 100.0
                                near_fpr = float(np.nanmean(stats_local[:, 2*num_methods:3*num_methods])) * 100.0

                        if not far_list:
                            far_mean = np.nan
                            far_fpr = np.nan
                        else:
                            config.dataset_feat = list(far_list)
                            stats_local = test_metrics(config, case=config.case, nameID=config.dataset_ID, methods=methods_local, features=ft)
                            if stats_local is None or stats_local.size == 0:
                                far_mean = np.nan
                                far_fpr = np.nan
                            else:
                                far_mean = float(np.nanmean(stats_local[:, :num_methods])) * 100.0
                                far_fpr = float(np.nanmean(stats_local[:, 2*num_methods:3*num_methods])) * 100.0

                        seed_ft_values[ft] = {
                            'near': near_mean,
                            'far': far_mean,
                            'near_fpr': near_fpr,
                            'far_fpr': far_fpr,
                        }
                        expansion_values[ft]['near'].append(near_mean)
                        expansion_values[ft]['far'].append(far_mean)
                        expansion_values[ft]['near_fpr'].append(near_fpr)
                        expansion_values[ft]['far_fpr'].append(far_fpr)
                    except Exception as e:
                        print(f"[exp3] Error during testing feature={ft} resnet={resnet_model} exp={expansion} seed={seed}: {e}")
                        seed_ft_values[ft] = {'near': np.nan, 'far': np.nan, 'near_fpr': np.nan, 'far_fpr': np.nan}
                        expansion_values[ft]['near'].append(np.nan)
                        expansion_values[ft]['far'].append(np.nan)
                        expansion_values[ft]['near_fpr'].append(np.nan)
                        expansion_values[ft]['far_fpr'].append(np.nan)

                try:
                    per_ft_entry = {}
                    for ft in feature_method_map:
                        vals = seed_ft_values.get(ft, {})
                        entry = _empty_entry()
                        try:
                            n = float(vals.get('near'))
                            f = float(vals.get('far'))
                            nf = float(vals.get('near_fpr'))
                            ff = float(vals.get('far_fpr'))
                            if not (np.isnan(n) or np.isnan(f) or np.isnan(nf) or np.isnan(ff)):
                                entry = {'near': n, 'far': f, 'near_fpr': nf, 'far_fpr': ff}
                        except Exception:
                            pass
                        per_ft_entry[ft] = entry
                        data_out['seed_results'][str(resnet_model)][ft][str(idx)][str(seed)] = entry

                    # Mirror branch entries under dynamic method aliases for robust resume.
                    for alias, branch in method_alias_to_branch.items():
                        alias_entry = per_ft_entry.get(branch, _empty_entry())
                        data_out['seed_results'][str(resnet_model)][alias][str(idx)][str(seed)] = alias_entry

                    with open(datafile, 'w') as jf:
                        json.dump(data_out, jf, indent=2)
                except Exception as e:
                    print(f"Warning: failed to persist EX3 per-seed checkpoint for ResNet{resnet_model} E={expansion} S={seed}: {e}")
                finally:
                    try:
                        config.dataset_feat = orig_dataset_feat
                    except Exception:
                        pass
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            for ft in feature_method_map:
                arr_near = np.array(expansion_values[ft]['near'], dtype=float)
                arr_far = np.array(expansion_values[ft]['far'], dtype=float)
                arr_near_fpr = np.array(expansion_values[ft]['near_fpr'], dtype=float)
                arr_far_fpr = np.array(expansion_values[ft]['far_fpr'], dtype=float)

                all_seed_values = (
                    len(arr_near) == len(seeds) and
                    len(arr_far) == len(seeds) and
                    len(arr_near_fpr) == len(seeds) and
                    len(arr_far_fpr) == len(seeds)
                )

                if not all_seed_values or np.any(np.isnan(arr_near)) or np.any(np.isnan(arr_far)) or np.any(np.isnan(arr_near_fpr)) or np.any(np.isnan(arr_far_fpr)):
                    results[resnet_model][ft][idx] = _empty_entry()
                else:
                    results[resnet_model][ft][idx] = {
                        'near': float(np.nanmean(arr_near)),
                        'far': float(np.nanmean(arr_far)),
                        'near_fpr': float(np.nanmean(arr_near_fpr)),
                        'far_fpr': float(np.nanmean(arr_far_fpr)),
                    }

            try:
                data_out['results'][str(resnet_model)] = data_out['results'].get(str(resnet_model), {})
                data_out['methods_1'] = list(methods_1)
                data_out['methods_2'] = list(methods_2)
                data_out['feature_method_map'] = feature_method_map
                for ft in feature_method_map:
                    serial_list = []
                    for entry in results[resnet_model][ft]:
                        serial_list.append({
                            'near': entry['near'] if isinstance(entry['near'], str) else float(entry['near']),
                            'far': entry['far'] if isinstance(entry['far'], str) else float(entry['far']),
                            'near_fpr': entry['near_fpr'] if isinstance(entry['near_fpr'], str) else float(entry['near_fpr']),
                            'far_fpr': entry['far_fpr'] if isinstance(entry['far_fpr'], str) else float(entry['far_fpr']),
                        })
                    data_out['results'][str(resnet_model)][ft] = serial_list

                # Persist explicit method-specific aliases for easier downstream parsing.
                for alias, branch in method_alias_to_branch.items():
                    data_out['results'][str(resnet_model)][alias] = list(data_out['results'][str(resnet_model)].get(branch, []))
                    results[resnet_model][alias] = list(results[resnet_model].get(branch, []))
                with open(datafile, 'w') as jf:
                    json.dump(data_out, jf, indent=2)
            except Exception as e:
                print(f"Warning: failed to persist EX3 JSON after ResNet{resnet_model} E={expansion}: {e}")

    if config.model_type == 'spike':
        results_filename = os.path.join(out_dir, f'EX3_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_seeds.txt')
        fpr_filename = os.path.join(out_dir, f'EX3_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_seeds_fpr95.txt')
    else:
        results_filename = os.path.join(out_dir, f'EX3_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_conv_seeds.txt')
        fpr_filename = os.path.join(out_dir, f'EX3_{config.dataset_ID}_T1_{config.num_time_steps_train}_T2_{config.num_time_steps_extract}_A_{config.auto_aug}_L_{config.loss}_conv_seeds_fpr95.txt')

    with open(results_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write("Spiking ResNet Experiment-3 (KNN+VIM) Multi-Test Results\n")
        f.write(f"{'='*60}\n")
        f.write("Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Branches: voltages->{methods_1}, probs->{methods_2}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for resnet_model in resnet_models:
            f.write(f"ResNet Model: {resnet_model}\n")
            f.write(f"Near OOD sets: {near_list}\n")
            f.write(f"{'Expansion':<12}{'KNN(voltages)':<20}{'VIM(probs)':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                knn_near = results[resnet_model]['voltages'][idx]['near']
                vim_near = results[resnet_model]['probs'][idx]['near']
                knn_str = f"{knn_near:>6.2f}" if isinstance(knn_near, float) else f"{knn_near:<20}"
                vim_str = f"{vim_near:>6.2f}" if isinstance(vim_near, float) else f"{vim_near:<20}"
                f.write(f"{expansion:<12}{knn_str:<20}{vim_str:<20}\n")

            f.write("\n")
            f.write(f"Far OOD sets: {far_list}\n")
            f.write(f"{'Expansion':<12}{'KNN(voltages)':<20}{'VIM(probs)':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                knn_far = results[resnet_model]['voltages'][idx]['far']
                vim_far = results[resnet_model]['probs'][idx]['far']
                knn_str = f"{knn_far:>6.2f}" if isinstance(knn_far, float) else f"{knn_far:<20}"
                vim_str = f"{vim_far:>6.2f}" if isinstance(vim_far, float) else f"{vim_far:<20}"
                f.write(f"{expansion:<12}{knn_str:<20}{vim_str:<20}\n")

            f.write(f"{'='*60}\n\n")

    with open(fpr_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write("Spiking ResNet Experiment-3 (KNN+VIM) Multi-Test Results (FPR95)\n")
        f.write(f"{'='*60}\n")
        f.write("Experiment Configuration:\n")
        f.write(f"  In-Distribution Dataset: {config.dataset_ID}\n")
        f.write(f"  Case: {config.case}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Trained on: {config.num_time_steps_train} time steps\n")
        f.write(f"  Feature extracted using: {config.num_time_steps_extract} time steps\n")
        f.write(f"  Number of epochs: {config.epochs}\n")
        f.write(f"  Fitting method: {config.fit}\n")
        f.write(f"  Loss function: {config.loss}\n")
        f.write(f"  Augmentation: {config.auto_aug}\n")
        f.write(f"  Branches: voltages->{methods_1}, probs->{methods_2}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for resnet_model in resnet_models:
            f.write(f"ResNet Model: {resnet_model}\n")
            f.write(f"Near OOD sets: {near_list}\n")
            f.write(f"{'Expansion':<12}{'KNN(voltages)':<20}{'VIM(probs)':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                knn_near = results[resnet_model]['voltages'][idx]['near_fpr']
                vim_near = results[resnet_model]['probs'][idx]['near_fpr']
                knn_str = f"{knn_near:>6.2f}" if isinstance(knn_near, float) else f"{knn_near:<20}"
                vim_str = f"{vim_near:>6.2f}" if isinstance(vim_near, float) else f"{vim_near:<20}"
                f.write(f"{expansion:<12}{knn_str:<20}{vim_str:<20}\n")

            f.write("\n")
            f.write(f"Far OOD sets: {far_list}\n")
            f.write(f"{'Expansion':<12}{'KNN(voltages)':<20}{'VIM(probs)':<20}\n")
            f.write(f"{'-'*60}\n")
            for idx, expansion in enumerate(expansions):
                knn_far = results[resnet_model]['voltages'][idx]['far_fpr']
                vim_far = results[resnet_model]['probs'][idx]['far_fpr']
                knn_str = f"{knn_far:>6.2f}" if isinstance(knn_far, float) else f"{knn_far:<20}"
                vim_str = f"{vim_far:>6.2f}" if isinstance(vim_far, float) else f"{vim_far:<20}"
                f.write(f"{expansion:<12}{knn_str:<20}{vim_str:<20}\n")

            f.write(f"{'='*60}\n\n")

    print(f"\nExperiment-3 detailed statistics saved to: {results_filename}")
    print(f"Experiment-3 FPR95 statistics saved to: {fpr_filename}")


def plot_ex1_from_data(data_or_path, out_dir=None, y_min=50.0, y_max=100.0, y_step=5, legend_fontsize=18):
    """Generate EX1 plots from JSON-like data or from a JSON file path.

    Arguments:
        data_or_path: dict (parsed JSON) or str (path to JSON file)
        out_dir: optional output directory (if None, uses 'results/ex_1')
        y_min, y_max, y_step: y-axis range and tick spacing
        legend_fontsize: font size for legend
    """
    import json
    import numpy as _np
    import matplotlib.pyplot as _plt
    import os as _os

    # Load data if a path was given
    if isinstance(data_or_path, str):
        with open(data_or_path, 'r') as _f:
            data = json.load(_f)
    else:
        data = data_or_path

    if out_dir is None:
        out_dir = _os.path.join('results', 'ex_1')
    _os.makedirs(out_dir, exist_ok=True)

    dataset_ID = data.get('dataset_ID', 'dataset')
    loss = data.get('case', '')
    expansions = list(data.get('expansions', []))
    feature_types = data.get('feature_types', [])
    model_tags = {int(k): v for k, v in data.get('model_tags', {}).items()} if data.get('model_tags') else {}
    results = data.get('results', {})

    # x positions
    x = list(expansions)
    positions = _np.arange(len(x))

    # For each feature type, create the same style plot used in statistics_exp_1
    for ft in feature_types:
        if ft == 'features':
            ft_label = 'Spike pattern'
        elif ft == 'voltages':
            ft_label = 'Membrane voltage'
        else:
            ft_label = ft

        _plt.figure(figsize=(10, 7))

        # results keys are strings of resnet numbers
        for rm_key, rm_vals in results.items():
            try:
                rm = int(rm_key)
            except Exception:
                # if non-int key, skip
                continue
            vals = rm_vals.get(ft, [])
            # convert entries to floats or nan
            y_near = []
            y_far = []
            for entry in vals:
                n = entry.get('near')
                f = entry.get('far')
                y_near.append(_np.nan if (isinstance(n, str) and n == 'not trained') or n is None else float(n))
                y_far.append(_np.nan if (isinstance(f, str) and f == 'not trained') or f is None else float(f))

            tag = model_tags.get(rm, f'ResNet{rm}')
            _plt.plot(positions, y_near, marker='o', linestyle='-', label=f'{tag} - near')
            _plt.plot(positions, y_far, marker='x', linestyle='--', label=f'{tag} - far')

        _plt.xlabel('Expansion')
        _plt.ylabel('Average AUROC')
        _plt.title(f'Average AUROC vs Expansion ({ft_label})')

        # Force consistent y-axis
        _plt.ylim(y_min, y_max)
        _plt.yticks(_np.arange(y_min, y_max + 1, y_step))

        # dashed grid on both axes
        _plt.grid(axis='both', linestyle='--', linewidth=0.8)

        # Set x ticks and labels
        _plt.xticks(positions, x)

        _plt.legend(ncol=2, fontsize=legend_fontsize)

        plotfile = _os.path.join(out_dir, f'EX1_{dataset_ID}_{ft}_L_{data.get("loss", "")}.png')
        try:
            _plt.savefig(plotfile, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: failed to save plot {plotfile}: {e}")
        _plt.close()

    print(f"Plots generated in: {out_dir}")
