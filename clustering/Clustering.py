
# Extract the spikes and logits for OoD
    #   accuracy_ood, preds_ood, logits_ood, _spk_count_ood = test(model, DEVICE, test_loader_ood, return_logits=True, return_conv_spikes=False)
    #   logger.info(f'Accuracy for the ood dataset {ood_dataset} is {accuracy_ood:.3f} %')

    #   #Create the median aggregations for each cluster of each class
    #   agg_counts_per_class_cluster = average_per_class_and_cluster(spk_count_train_clusters,preds_train_clusters,clusters_per_class,n_samples=1000, option='median')

    #   # Computation of the distances of train, test and ood
    #   distances_train_per_class, _ = distance_to_clusters_averages(spk_count_train, preds_train, agg_counts_per_class_cluster)
    #   distances_test_per_class, _ = distance_to_clusters_averages(spk_count_test, preds_test, agg_counts_per_class_cluster)
    #   distances_ood_per_class, _ = distance_to_clusters_averages(spk_count_ood, preds_ood, agg_counts_per_class_cluster)



import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

class Clustering():

    def __init__(self, name):
        pass

    @staticmethod
    def clustering_1(features, labels, numClasses):
        '''
        The clustering method: 
        https://github.com/aitor-martinez-seras/OoD_on_SNNs/blob/main/Explainable_OoD_detection_on_SNNs.ipynb
        '''
    
#     subset_train_loader_clusters: Dataset object for training data (specifically the subset_train_loader).
# preds_train_clusters: Predicted labels or classifications (likely from a model).
# spk_count_train_clusters: Feature values for each sample (likely speaker counts or similar features).
# size: Number of samples to consider for each class (default is 1000).
# distance_for_clustering: Range of distance values for clustering (default is (800, 3000)).

        # Create distance thresholds
        distance_thresholds = np.linspace(100, 5000, 50)
        silhScoresPerClass = []
        clusterLabels = []

        opt_dist_thr_per_class = []
        opt_silh_score_values_per_class = []

        # check how to extract number of classes from data loader
        #   n_classes = len(subset_train_loader_clusters.dataset.dataset.classes)

        # Loop trought the number of classes
        for i in range(numClasses):
            silh_scores = []
            for dist in distance_thresholds:
                # define cluster model
                cluster_model = AgglomerativeClustering(n_clusters=None,affinity='manhattan',linkage='average',distance_threshold=dist)
                cluster_model.fit(features[labels == i])
                clusterLabels.append(cluster_model.labels_)
                silh_scores.append(silhouette_score(features[labels == i], cluster_model.labels_, metric='manhattan'))
            silhScoresPerClass.append(silh_scores)

            # ovo bi trebalo da moze mnogo jednostavnije
            # Iterate the inverted to catch the smallest distance value with the 
            # greatest silhouette score
            max_score = 0
            max_index = 0
            for idx, current_score in enumerate(silh_scores):
                # Store the greatest value we encounter traveling the curve
                # Only update the value if it is greater, not if it equal
                if current_score > max_score: 
                    max_index = idx
                    max_score = current_score
            # We append the distance treshold to a list where they are going to be
            # stored, one for each class
            opt_dist_thr_per_class.append(distance_thresholds[max_index])
            opt_silh_score_values_per_class.append(silh_scores[max_index])

        # Create the clusters by extracting the labels for every sample
        clusters_per_class = []
        for i in range(numClasses):
            cluster_model = AgglomerativeClustering(n_clusters=None,affinity='manhattan',linkage='complete',distance_threshold=opt_dist_thr_per_class)
            cluster_model.fit(features[labels == i])
            # Save the cluster models
            clusters_per_class.append(cluster_model)

        return clusters_per_class


