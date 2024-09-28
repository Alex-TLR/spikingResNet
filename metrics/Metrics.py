from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
import time 
import numpy as np

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
        TPR = cm[1][1] / (cm[1][1] + cm[1][0])
        FPR = cm[0][1] / (cm[0][1] + cm[0][0])
        auroc = roc_auc_score(labels, distances)
        apr = average_precision_score(labels, distances)
        precision, recall, _ = precision_recall_curve(labels, distances)
        aupr = auc(recall, precision)

        return auroc, aupr, FPR


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
    def MSP(test_data, threshold):
        '''
        Calculates the Maximum Softmax Probability metrics of input 
        features, regarding the threshold value.

        Inputs:
            test_data:      matrix of input row-wise features
            threshold:      threshold value

        Outputs:
            predictions:    an array of predicted labels
            max_probs:      an array of output features, 1-D 
        '''

        s = Metrics.softmax(test_data)
        max_probs = np.max(s, axis=1)
        predictions = (max_probs > threshold).astype(np.int32)
        return predictions, max_probs
    

    @staticmethod
    def NCM(train_X, train_Y, test_X, threshold, num_classes):
        '''
        Nearest Class Mean is the distance between test feature and the nearest
        training class centroid.  

        Inputs:
            train_X:        matrix with train features
            train_Y:        an array with train labels
            test_X:         matrix with test features
            num_classes:    number of classes

        Outputs:
            predictions:    an array of predicted labels
            distances:      an array of output features, 1-D 

        '''

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
    

    @staticmethod
    def KNN(train_X, train_Y, test_X, threshold, number_neighbors):
        '''
        k Nearest Neighbor classification

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

        neigh = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'euclidean')
        neigh.fit(train_X, train_Y)

        predictions = neigh.predict(test_X)
        neigh_dist, _ = neigh.kneighbors(test_X, return_distance = True)

        distances = -np.average(neigh_dist, axis=1)
        predictions = (distances > threshold).astype(np.int32)

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

        neigh = KNeighborsClassifier(n_neighbors = number_neighbors, metric = 'euclidean')
        neigh.fit(train_X, train_Y)

        predictions = neigh.predict(test_X)
        neigh_dist, neigh_ind = neigh.kneighbors(test_X, return_distance = True)

        distances = []
        m = []
        for i in range(len(neigh_dist)):
            for j in range(len(neigh_dist[i])):
                if(train_Y[neigh_ind[i][j]] != train_Y[neigh_ind[i][0]]):
                    m.append(j)
                    break
            
        for i in range(len(neigh_dist)):
            distances.append(-neigh_dist[i][0]/neigh_dist[i][m[i]])
        if(distances[i] > threshold):
            predictions[i] = 1
        else:
            predictions[i] = 0

        return predictions, distances
    

    @staticmethod
    def MD(train_X, train_Y, test_X, threshold, num_classes, num_features):
        '''
        Mahalanobis distance

        Inputs:
            train_X:        matrix with train features
            train_Y:        array with train labels
            test_X:         matrix with test features
            threshold:      threshold
            num_classes:    number of classes
            num_features:   feature dimension

        Outputs:
            predictions:    an array of predicted labels
            distances:      an array of output features, 1-D 
        '''

        mean_vectors = np.zeros((num_classes, num_features))

        b = 0

        for i in range(0, num_classes):
            for j in range(len(train_Y)):
                if(train_Y[j] == i):
                    b = b + 1
                    mean_vectors[i] = mean_vectors[i] + train_X[j]
            mean_vectors[i] = mean_vectors[i] / b
            b = 0

        # Compute the mean for each class
        # for i in range(n_classes):
        #     class_samples = train_X[train_Y == i]
        #     mean_vectors[i] = class_samples.mean(axis=0)

        pom = np.zeros((len(train_Y), num_features))

        cov = np.zeros((num_features, num_features))

        for i in range(0, num_classes):
            for j in range(len(train_Y)):
                if(train_Y[j] == i):
                    pom[j] = train_X[j]-mean_vectors[i]

        data = np.asmatrix(pom).transpose()

        cov = np.cov(data, bias=False)

        #print(cov)
        cov = cov+0.001*np.identity(num_features)

        cov_inv = np.linalg.inv(cov)

        MDK = np.zeros((len(test_X), num_classes))
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
