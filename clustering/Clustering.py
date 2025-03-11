
# We should recreate the clustering code from:
# https://github.com/aitor-martinez-seras/OoD_on_SNNs/blob/main/Explainable_OoD_detection_on_SNNs.ipynb
# 
# def create_clusters(subset_train_loader_clusters, preds_train_clusters,
#                     spk_count_train_clusters,
#                     size=1000, distance_for_clustering=None, verbose=2):
#   """
#   Verbose = 0 -> No prints and plots neither loggin info
#   verbose = 1 -> Returns loggin info only
#   Verbose = 2 -> Prints and plots
#   """
#   # Select a distance threshold for each class
#   if distance_for_clustering is None:
#     distance_for_clustering = (800,3000)
#   opt_dist_thr_per_class = []
#   opt_silh_score_values_per_class = []
#   dist_thrs          = np.linspace(distance_for_clustering[0], 
#                                    distance_for_clustering[1],50)
#   silhScoresPerClass = []
#   clusterLabels = []

#   n_classes = len(subset_train_loader_clusters.dataset.dataset.classes)
#   for class_index in tqdm(range(n_classes), desc='Computing silhuette score for various distance thresholds'):
#     dunnIndexes = []
#     silh_scores = []
#     for dist in dist_thrs:
#       indices = searchIndicesOfClass(class_index, preds_train_clusters, size)
#       cluster_model = AgglomerativeClustering(n_clusters=None,affinity='manhattan',linkage='average',distance_threshold=dist)
#       try: # Handle the case that one class has no representation in the training samples
#         cluster_model.fit(spk_count_train_clusters[indices])
#         clusterLabels.append(cluster_model.labels_)
#       except ValueError as e:
#         print('Error probably caused by the lack of training samples for one specific class')
#         raise(e)
#       try:
#         silh_scores.append(silhouette_score(spk_count_train_clusters[indices], cluster_model.labels_, metric='manhattan'))
#       except ValueError:
#         silh_scores.append(0)
#     silhScoresPerClass.append(silh_scores)

#     # Iterate the inverted to catch the smallest distance value with the 
#     # greatest silhouette score
#     max_score = 0
#     max_index = 0
#     for idx, current_score in enumerate(silh_scores):
#       # Store the greatest value we encounter traveling the curve
#       # Only update the value if it is greater, not if it equal
#       if current_score > max_score: 
#         max_index = idx
#         max_score = current_score
#     # We append the distance treshold to a list where they are going to be
#     # stored, one for each class
#     opt_dist_thr_per_class.append(dist_thrs[max_index])
#     opt_silh_score_values_per_class.append(silh_scores[max_index])
  
#   # Plot the silhouette score for every distance threshold
#   if verbose == 2:
#     # Plot to see the silhouette scores
#     print('Selected distance thresholds:\n', opt_dist_thr_per_class)
#     if n_classes == 10:
#       fig, axes = plt.subplots(2,5,figsize=(6*n_classes/2, 12))
#     elif n_classes == 26:
#       fig, axes = plt.subplots(2,13,figsize=(6*n_classes/2, 12))
#     else:
#       raise NameError(f'The number of classes {n_classes} is not implemented for the plots')

#     for class_index, ax in enumerate(axes.flat):
#       ax.plot(dist_thrs,silhScoresPerClass[class_index], color='blue')
#       ax.plot(opt_dist_thr_per_class[class_index], opt_silh_score_values_per_class[class_index], 'ro')
#       ax.set_title(subset_train_loader_clusters.dataset.dataset.classes[class_index])
#     plt.savefig('silhouetteScores.pdf')

#   # Create the clusters by extracting the labels for every sample
#   clusters_per_class = []
#   for class_index in range(n_classes):
#     indices = searchIndicesOfClass(class_index, preds_train_clusters, 1000)
#     if isinstance(opt_dist_thr_per_class, list):
#       cluster_model = AgglomerativeClustering(n_clusters=None,affinity='manhattan',linkage='complete',distance_threshold=opt_dist_thr_per_class[class_index])
#     else:
#       cluster_model = AgglomerativeClustering(n_clusters=None,affinity='manhattan',linkage='complete',distance_threshold=opt_dist_thr_per_class)
    
#     cluster_model.fit(spk_count_train_clusters[indices])
#     # Save the cluster models
#     clusters_per_class.append(cluster_model)

#   if verbose == 2:
#     # Plot the top three levels of the dendrogram
#     if n_classes == 10:
#       fig, axes = plt.subplots(2,5,figsize=(6*n_classes/2, 12))
#     elif n_classes == 26:
#       fig, axes = plt.subplots(2,13,figsize=(6*n_classes/2, 12))
#     else:
#       raise NameError(f'The number of classes {n_classes} is not implemented for the plots')
#     fig.suptitle('Hierarchical Clustering Dendrogram', fontsize=22, y=0.94)
#     #fig.supxlabel('X axis: Number of points in node (index of the number if not in parenthesis)',fontsize = h + w*0.1,y=0.065)
  
#     for class_index, ax in tqdm(enumerate(axes.flat), desc='Create the clusters with the selected distance thresholds'):
#       plot_dendrogram(cluster_model, truncate_mode='level', p=3, ax=ax)
#       ax.set_title('Class {}'.format(subset_train_loader_clusters.dataset.dataset.classes[class_index]),fontsize=22)
#       #ax[i,j].set_xlabel("Number of points in node",fontsize=h)

#     plt.savefig(f'DendrogramPerClass.pdf')
#     fig.show()

#     print_created_clusters_per_class(clusters_per_class)

#   if verbose == 1:
#     string_for_logger = 'Created clusters:\n' + '-'*75 + '\n'
#     for class_index in range(len(train_data.classes)):
#       unique, counts = np.unique(clusters_per_class[class_index].labels_, return_counts=True)
#       string_for_logger += f'Clase {train_data.classes[class_index].ljust(15)} \t {dict(zip(unique, counts))}\n' + '-'*75 + '\n'

#     return clusters_per_class, string_for_logger
  
#   return clusters_per_class

# Later the code gives:
# Create clusters
    dist_clustering = (500, 5000)

    clusters_per_class, logging_info = create_clusters(
         # training_subset_clusters = torch.utils.data.Subset(train_data, [x for x in selected_indices_per_class])
        subset_train_loader_clusters,   # subset_train_loader_clusters = torch.utils.data.DataLoader(training_subset_clusters, batch_size=256, shuffle=False)
          # dakle subset_train_loader_clusters je DataLoader za podskup trening podataka
                              
                              # pred = output.argmax(dim=1, keepdim=True)
        preds_train_clusters, # ovo je np.concatenate(preds).squeeze(), zapravo predikcija 
                              
        spk_count_train_clusters,  # spk_count_train_clusters = np.sum(_spk_count_train_clusters, axis=0, dtype='uint16')
        distance_for_clustering=dist_clustering, # ovdje je lista pragova za koje se 
        size=1000, # indices = searchIndicesOfClass(class_index, preds_train_clusters, size) broj primjera za clustering (po kategoriji ? )
        verbose=1)
    logger.info(logging_info)

    # Initialize for every model, as we save the results for every model
    results_list = []

    for ood_dataset in tqdm(out_of_distribution_datasets_benchmark, desc='Out-of-Distribution dataset loop'):

      logger.info(f'Logs for benchmark with the OoD dataset {ood_dataset}')
      
      # Load OoD dataset from the dictionary. In case it is MNIST-C, load the 
      # selected option
      if ood_dataset.split('/')[0] == 'MNIST-C':
        test_loader_ood = OUT_DISTRIBUTION_DATASETS[ood_dataset.split('/')[0]](
            BATCH_SIZE, 
            test_only=True,
            option=ood_dataset.split('/')[1]
        )
      else:
        test_loader_ood = OUT_DISTRIBUTION_DATASETS[ood_dataset](BATCH_SIZE, 
                                                                 test_only=True)

      # Extract the spikes and logits for OoD
      accuracy_ood, preds_ood, logits_ood, _spk_count_ood = test(model, DEVICE, test_loader_ood, return_logits=True, return_conv_spikes=False)
      logger.info(f'Accuracy for the ood dataset {ood_dataset} is {accuracy_ood:.3f} %')

      # Convert spikes to counts
      # OoD
      if isinstance(_spk_count_ood, tuple):
        _spk_count_ood, _ = _spk_count_ood
      spk_count_ood = np.sum(_spk_count_ood,axis=0, dtype='uint16')
      logger.info(f'OoD set: {spk_count_ood.shape}')

      #Create the median aggregations for each cluster of each class
      agg_counts_per_class_cluster = average_per_class_and_cluster(spk_count_train_clusters,preds_train_clusters,clusters_per_class,n_samples=1000, option='median')

      # Computation of the distances of train, test and ood
      distances_train_per_class, _ = distance_to_clusters_averages(spk_count_train, preds_train, agg_counts_per_class_cluster)
      distances_test_per_class, _ = distance_to_clusters_averages(spk_count_test, preds_test, agg_counts_per_class_cluster)
      distances_ood_per_class, _ = distance_to_clusters_averages(spk_count_ood, preds_ood, agg_counts_per_class_cluster)



#  
class Clustering():

    def __init__(self, name):
        pass


