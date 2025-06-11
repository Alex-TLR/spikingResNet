from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestCentroid
from utils.Utils import Utils, distances_from_average_clusters
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import k_means, DBSCAN, KMeans
import time 
import numpy as np
from scipy.spatial.distance import cdist

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
        print(f"Confusion matrix:\n{cm}")
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
    def MSP(test_data, threshold):
        '''
        Calculates the Maximum Softmax Probability (MSP) metrics of input 
        features, regarding the threshold value. MSP finds maximum softmax
        row-wise, and checks if it is larger then threshold.

        Large sofmtax values indicates larger probability that feature vector
        belongs to the In-distribution pattern.

        Inputs:
            test_data:      Matrix of input row-wise features
            threshold:      Threshold value

        Outputs:
            predictions:    Array of predicted labels
            max_probs:      Array of output features, 1-D 
        '''

        s = Metrics.softmax(test_data)
        max_probs = np.max(s, axis=1)
        predictions = (max_probs > threshold).astype(np.int32)
        return predictions, max_probs
    

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
            predictions:    an array of predicted labels

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
    def AGGLO(test_X, averagePerClass, threshold, number_classes):
        '''
        Distances after agglomerative clustering

        Inputs:
            train_X:        matrix with train features
            train_Y:        an array with train labels
            test_X:         matrix with test features
            clusters:       list of cluster per class

        Outputs:
            predictions:    an array of predicted labels
            distances:      an array of output features, 1-D 

        '''

        predictions = np.zeros(len(test_X))
        distances = np.zeros(len(test_X))

        distances = get_dist(test_X, averagePerClass)
        # print(f"distances.shape: {distances.shape}")
        # print(f"test_X.shape: {test_X.shape}")
        # print(f"Number of clusters: {np.sum(i.shape[1] for i in averagePerClass)}")
        print(f"Number of clusters: {averagePerClass.shape}")

        for i in range(len(test_X)):    
            predictions[i] = 1 if distances[i] > threshold else 0

        return predictions, distances
    
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
    def DBSCAN(train_X, test_X, threshold):
        '''
        Density-Based Spatial Clustering of Applications with Noise
        '''

        predictions = np.zeros(len(test_X))

        # Apply dbscan to training data
        dbscan = DBSCAN(eps=200, min_samples=5)
        predictated_labels = dbscan.fit_predict(train_X)
        print(f"predictated_labels: {predictated_labels.shape}")

        # Identify core points
        core_samples_mask = predictated_labels != -1  # Ignore noise (-1)
        core_points = train_X[core_samples_mask]
        print(f"core_points: {core_points}")

        # Compute cluster centers
        unique_clusters = np.unique(predictated_labels[core_samples_mask])
        cluster_centers = np.array([train_X[predictated_labels == c].mean(axis=0) for c in unique_clusters])
        print(f"DBSCAN: {cluster_centers}")
        print(f"DBSCAN: {cluster_centers.shape}")
        if cluster_centers.ndim == 1:
            cluster_centers = cluster_centers.reshape(1, -1)

        print(f"cluster_centers: {cluster_centers.shape}")

        # Compute distance of each test sample from the nearest cluster center
        # distances = cdist(test_X, cluster_centers, metric='euclidean')  # Pairwise distance
        distances = np.zeros(len(test_X))

        for i in range(len(test_X)):
            dist = np.zeros(len(cluster_centers))
            for j in range(len(cluster_centers)):
                dist[j] = np.sqrt(np.sum((cluster_centers[j] - test_X[i])**2))
                # dist[j] = np.power(np.sum(np.abs(centroids_X[j] - test_X[i])**p), 1/p)
            distances[i] = -dist.min()



        # print(f"distances: {distances}")
        # print(f"distances: {distances.shape}")
        # nearest_distances = np.min(distances, axis=1)  # Minimum distance to any cluster center

        # (Optional) Convert distances to scores (e.g., negative distance for ROC analysis)
        # distances = -nearest_distances  # Higher score means closer to a cluster
         
        for i in range(len(test_X)):    
            predictions[i] = 1 if distances[i] > threshold else 0

        print(f"predictions: {predictions}")

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


    @staticmethod
    def KMEANS(data: np.ndarray, labels: np.ndarray, ncpc, y_test, threshold):
        num_classes = len(set(labels))
        
        clusters = np.zeros((num_classes * ncpc, data.shape[1]))

        for i in range(num_classes):
            clusters[i: i + ncpc] = k_means(data[labels == i], ncpc)[0]
        
        # print(clusters)

        return predictions(y_test, clusters, threshold), get_dist(y_test, clusters)
    
    @staticmethod
    def KMEANSFull(data: np.ndarray, labels: np.ndarray, ncpc, y_test, threshold):
        num_classes = len(set(labels))
        
        clusters = np.zeros((ncpc, data.shape[1]))

        clusters = k_means(data, ncpc)[0]
        
        print(f"clusters: {clusters.shape}")

        return predictions(y_test, clusters, threshold), get_dist(y_test, clusters)


def get_dist(features, centroids):
    '''
        Calculate distance of each feature samples from centroids.
    '''
    # TODO: Test different distances/metrics other than Euclidian
    # TODO: Replace 10 with number of classes

    distances = np.zeros(len(features))
    epsilon = 1e-10
    p = 100

    num_centroids = len(centroids)

    for i in range(len(features)):
        dist = np.zeros(num_centroids)
        for j in range(num_centroids):
            # dist[j] = np.sqrt(np.sum((centroids[j] - features[i])**2))
            # dist[j] = np.power(np.sum(np.abs(centroids[j] - data[i])**p), 1/p)
            dist[j] = np.sum(np.abs(centroids[j] - features[i]))
        distances[i] = -dist.min()
    
    return distances


def predictions(data, centroids, threshold):
    '''
        Make Id/Ood predictions regarding the given distances and threshold
    '''

    distances = get_dist(data, centroids)
    predictions = [1 if x > threshold else 0 for x in distances]
    return predictions