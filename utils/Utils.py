from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10, SVHN, Places365, EMNIST, Food101
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import os
from sklearn.manifold import TSNE

'''
Utility functions
'''

class Utils():

    def __init__(self, name):
        pass

    @staticmethod
    def load_data(database_name):
        '''
        database_name:      Name of the database (MNIST, KMNIST, FMNIST)
        '''

        transformData_gray_28 = transforms.Compose([transforms.Resize((28, 28)),
                                                    transforms.Grayscale(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0,), (1,))])
        
        transformData_rgb_32  = transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0,0,0,), (1,1,1,))])

        if database_name == 'MNIST':
            name = 'mnist' 
            Name = 'MNIST'
            transformData = transformData_gray_28
        elif database_name == 'KMNIST':
            name = 'kmnist' 
            Name = 'KMNIST'
            transformData = transformData_gray_28
        elif database_name == 'FMNIST':
            name = 'fmnist'
            Name = 'FashionMNIST'
            transformData = transformData_gray_28
        elif database_name == 'EMNIST':
            name = 'emnist'
            Name = 'EMNIST'
            transformData = transformData_gray_28
        elif database_name == 'Letters':
            name = 'letters'
            Name = 'EMNIST'
            transformData = transformData_gray_28
        elif database_name == 'CIFAR10':
            name = 'cifar10'
            Name =  'CIFAR10'
            transformData = transformData_rgb_32
        elif database_name == 'SVHN':
            name = 'svhn'
            Name =  'SVHN'
            transformData = transformData_rgb_32
        elif database_name == 'Places365':
            name = 'places'
            Name =  'Places365'
            transformData = transformData_rgb_32
        elif database_name == 'Food101':
            name = 'food'
            Name =  'Food101'
            transformData = transformData_rgb_32
        else:
            print("Wrong database name!")
            return -1

        if (database_name != 'SVHN') and (database_name != 'Places365') and (database_name != 'EMNIST') and (database_name != 'Letters') and (database_name != 'Food101'):
            command_train = Name + '(root = \'data/' + name + '/\', download=True, train = True, transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, train = False, transform = transformData)'
            # print(command_test)
        elif database_name == 'SVHN':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', target_transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', target_transform = transformData)'
            # print(command_test)
        elif database_name == 'Places365':
            command_train = Name + '(root = \'data/' + name + '/\', download=False, split = \'train-standard\', small = True, target_transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=False, split = \'val\', small = True, target_transform = transformData)'
            # print(command_test)
        elif database_name == 'EMNIST':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'digits\', train = True, transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'digits\', train = False, transform = transformData)'
            # print(command_test)
        elif database_name == 'Letters':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = True, transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'letters\', train = False, transform = transformData)'
            # print(command_test)
        elif database_name == 'Food101':
            command_train = Name + '(root = \'data/' + name + '/\', download=True, split = \'train\', transform = transformData)'
            # print(command_train)
            command_test = Name + '(root = \'data/' + name + '/\', download=True, split = \'test\', transform = transformData)'
            # print(command_test)
        else:
            print(f"Wrong data srt name.")
        
        # print(command_train)
        dataset_train = eval(command_train)
        dataset_test = eval(command_test)

        return dataset_train, dataset_test
    

    @staticmethod
    def get_image_size(dataset, name):
        '''
        dataset:        loaded dataset
        name:           database name 
        '''

        if (name == 'SVHN'):
            train_tensor = dataset.data[0]
            imageSize = train_tensor.shape
        elif (name == 'Food101'):
            # print(dataset[0])
            train_tensor, _ = dataset[0]
            imageSize = train_tensor.shape
        else:
            train_tensor, _ = dataset[0]
            imageSize = train_tensor.size()
        # print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')

        return imageSize[0], imageSize[1], imageSize[2]
    

    @staticmethod
    def make_features_dir(modelType, case = '00'):

        folderPath = 'features/' + modelType + '/case_' + case
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
            print(f"Directory {folderPath} created.")
        else:
            print(f"Directory {folderPath} already exists.")

        return None
    
    @staticmethod
    def does_file_exists(fileName):

        '''
        Reverse logic
        '''

        if os.path.exists(fileName):
            print(f"The file {fileName} exists.")
            return False
        else:
            print(f"The file {fileName} does not exist.")
            return True

    @staticmethod
    def data_loader(dataset_train, dataset_test, batchSize, dataset_name, fullTrain=False):
        '''
        Define data loaders
        '''
        if (fullTrain == False):
            print("Define training/validation split ...")
            dataSize = len(dataset_train)
            tSize = int(0.8 * dataSize)
            vSize = dataSize - tSize

            train_data, val_data = random_split(dataset_train, [tSize, vSize])
            test_data = dataset_test
            if dataset_name == 'SVHN':
                train_data = SVHNDataset(data=train_data.dataset.data, labels=train_data.dataset.labels)
                val_data = SVHNDataset(data=val_data.dataset.data, labels=val_data.dataset.labels)
                test_data = SVHNDataset(data=test_data.data, labels=test_data.labels)

            print("Train data length ", len(train_data))
            print("Valid data length ", len(val_data))
            print("Test data length ", len(test_data))
            train_loader = DataLoader(train_data, batchSize, shuffle=True)
            val_loader = DataLoader(val_data, batchSize)
            test_loader = DataLoader(test_data, batchSize)

            return train_loader, val_loader, test_loader

        else:
            train_data = dataset_train
            test_data = dataset_test
            # print("Train data length ", len(train_data))
            # print("Test data length ", len(test_data))
            if dataset_name == 'SVHN':
                train_data = SVHNDataset(data=dataset_train.data, labels=dataset_train.labels)
                test_data = SVHNDataset(data=dataset_test.data, labels=dataset_test.labels)
                train_loader = DataLoader(train_data, batchSize, shuffle=True)
                test_loader = DataLoader(test_data, batchSize, shuffle=False)
            else:
                train_loader = DataLoader(train_data, batchSize, shuffle=True)
                test_loader = DataLoader(test_data, batchSize, shuffle=False)

            return train_loader, test_loader

  
    @staticmethod
    def showBatch(inputData):
        '''
        Show the batch of images in the grid
        '''
        for images, labels in inputData:
            fig, ax = plt.subplots(figsize=(12,6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            plt.show()
            break

    @staticmethod
    def showBatchImages(inputImages):
        '''
        Show the batch of images in the grid
        '''
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.imshow(make_grid(inputImages, nrow=8).permute(1, 2, 0))  # Arrange images in a grid
        plt.show()


    @staticmethod
    def get_device():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        return device 
    

    @staticmethod
    def find_threshold(test_labels, test_dist, positive_label, drop = True): 
        '''
        Find threshold values input values

        Inputs:
            test_Y:         an array with train labels
            test_dist:      an array of test features distances 
            positive_label: set if 1 is positive label, or 0 is positive label

        Outputs:
            th_tpr80:       threshold for which TPR is 80 percent
            th_tpr95:       threshold for which TPR is 95 percent
        '''

        test_dist = np.array(test_dist)
        fpr, tpr, thresholds = roc_curve(test_labels, test_dist, pos_label = positive_label, drop_intermediate = drop)
        # print(f"thresholds: {thresholds.min()}, {thresholds.max()}")
        # Plot ROC curve
        # plt.figure()
        # plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.grid(True)
        # plt.show()

        dtpr80 = np.absolute(tpr-0.8)
        th_tpr80 = thresholds[dtpr80.argmin()]

        dtpr95 = np.absolute(tpr-0.95)
        th_tpr95 = thresholds[dtpr95.argmin()]

        # predictions = np.zeros(len(test_dist))
        # for i in range(len(test_dist)):    
        #     predictions[i] = 0 if test_dist[i] < th_tpr95 else 1

        # print(f"Result: {np.sum(a == b for a, b in zip(test_labels, predictions) if a == 1)  / len(predictions[test_labels == 1])}")
        # print(f"Utils, th_tpr80: {th_tpr80}, th_tpr95: {th_tpr95}")

        return th_tpr80, th_tpr95


    @staticmethod
    def plot_train_val_stats(history):
        '''
        Plot history
        '''
        pass 

    @staticmethod
    def visualize_feature(Id, Ood, case):
        '''
            Plot reduced features Id/Ood
        
            Id (str):   In distribution dataset
            Ood (str):  Out of distribution dataset  

            feat_train = F['arr1']  
            prob_train = F['arr2']  
            tags_train = F['arr3']  
            feat_test  = F['arr4']  
            prob_test  = F['arr5']  
            tags_test  = F['arr6']  
        
        '''

        fileName = 'tsne/' + str(case) + '_' + str(Id) + '_' + str(Ood) + '.npz'
        print(fileName)
        
        if (Utils.does_file_exists(fileName)):
            
            tsne = TSNE(n_components=2, random_state=42)
            folderName = 'features/spike/case_' + case + '/'
            print(folderName)
            files = os.listdir(folderName)
            print(files)

            filePathId = folderName + str(Id) + '-on_' + Id.lower() + '.npz'
            if Ood is not None:
                filePathOod = folderName + str(Ood) + '-on_' + Id.lower() + '.npz'

            features = []
            border = 0

            F_id = np.load(filePathId)
            features1 = F_id['arr4']

            if Ood is not None:
                border = len(features1)
                F_ood = np.load(filePathOod)
                features2 = F_ood['arr4']

            print(f'features1.shape is {features1.shape}')

            if Ood is not None:
                print(f'features2.shape is {features2.shape}')
                features = np.vstack((features1, features2))
            else:
                features = features1 
            print(f'features.shape is {features.shape}')
            features_tsne = tsne.fit_transform(features)

            if Ood is not None:
                features1_tsne = features_tsne[:border]
                features2_tsne = features_tsne[border:]

            if Ood is not None:
                np.savez(fileName, arr1=features1_tsne, arr2=features2_tsne)
            else:
                np.savez(fileName, arr1=features_tsne)
            print(f'features_tsne shape is {features_tsne.shape}')

        else:
            # Load data
            F = np.load(fileName)
            features1_tsne = F['arr1']
            features2_tsne = F['arr2']

        # available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        # print(available_fonts)

        plt.rcParams.update({
            'text.usetex': True,  # Use LaTeX for rendering text
            'font.family': 'serif',  # Set the base font to serif
            'font.serif': ['Times New Roman'],  # Set Times New Roman (if available)
            'mathtext.fontset': 'stix',  # Use the STIX math font family (similar to IEEE)
            'axes.titlesize': 24,
            'axes.labelsize': 24,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'figure.titlesize': 24,
        })

        plt.figure(figsize=(10, 8))

        if Ood is not None:
            # Plot the first set of features
            plt.scatter(features1_tsne[:, 0], features1_tsne[:, 1], marker='.', c='blue', label=str(Id), alpha=0.5)
            # Plot the second set of features
            plt.scatter(features2_tsne[:, 0], features2_tsne[:, 1], marker='x', c='orange', label=str(Ood), alpha=0.5)
        else:
            plt.scatter(features_tsne[:, 0], features_tsne[:, 1], marker='.', c='blue', label=str(Id), alpha=0.5)

        # Add labels and legend
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2'
        # , fontname='Liberation Serif',

        if case == '03':
            plt.title('t-SNE visualization of spike-ResNet10 feature vectors', fontsize=24)
        elif case == '04':
            plt.title('t-SNE visualization of spike-Conv feature vectors', fontsize=24)
        elif case == '06':
            plt.title('t-SNE visualization of spike-ResNet18 feature vectors', fontsize=24)
        plt.legend(fontsize=16)
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # ax = plt.gca()  # Get current axis
        # # for spine in ax.spines.values():
        # #     spine.set_visible(False)
        # # plt.tight_layout(pad=0)
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # # Show the plot
        # plt.show()

        fileName = 'tsne/' + str(case) + '_' + str(Id) + '_' + str(Ood) + '.pdf'
        plt.savefig(fileName, format="pdf", dpi=300, bbox_inches="tight", transparent=False)

        # Initialize interactive mode
        # plt.ion()

        # # Create a 3D plot
        # fig = plt.figure(figsize=(12, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot the first set of features
        # ax.scatter(features1_tsne[:, 0], features1_tsne[:, 1], features1_tsne[:, 2], 
        #         c='blue', label='Dataset 1', alpha=0.6, s=50)

        # # Plot the second set of features
        # ax.scatter(features2_tsne[:, 0], features2_tsne[:, 1], features2_tsne[:, 2], 
        #         c='red', label='Dataset 2', alpha=0.6, s=50)

        # # Add labels and legend
        # ax.set_xlabel('t-SNE Dimension 1')
        # ax.set_ylabel('t-SNE Dimension 2')
        # ax.set_zlabel('t-SNE Dimension 3')
        # ax.set_title('3D t-SNE Visualization of Feature Vectors')
        # ax.legend()

        # # Show the plot
        # plt.show()

        # # Disable interactive mode
        # plt.ioff()

        return None
    
    @staticmethod
    def prediction_spikes(spikes):
        '''
        Function that calculates the predicted class for each sample in the batch. The sample
        is a feature vector, i.e. spike pattern. Prediction is done as an argmax over the spike pattern.
        The predicted class is the one with the highest spike count.

        Inputs:
            spikes (np.array): tensor with the spike patterns (batch_size, num_classes)

        Outputs:
            predicted (np.array): array with the predicted class for each sample (batch_size,)
        '''
        return np.argmax(spikes, axis=1)
    
    @staticmethod
    def prediction_probs(probs):
        '''
        Function that calculates the predicted class for each sample in the batch. The sample
        is a vectro with class probabilities. The predicted class is the one with the highest probability.

        Inputs:
            probs (np.array): tensor with the sample class probabilities (batch_size, num_classes)

        Outputs:
            predicted (np.array): array with the predicted class for each sample (batch_size,)
        '''
        return np.argmax(probs, axis=1)
    
    @staticmethod
    def distance_from_centroids(spikes, centroids, net_outputs, prob):
        '''
        Function that computes the distances for each feature vector (spike pattern) from the each centroid
        given as input parameter.

        Inputs:
            spikes (np.array): tensor with the spike patterns (batch_size, feature_size)
            centroids (list): list of cluster centroids
            net_outputs (np.array): tensor with the network outputs (batch_size, num_classes)
            
            prob (np.array): tensor with the sample class probabilities (batch_size, num_classes)

        Outputs:
            result (list): list of distances for each class
        '''
        
        print(f"centroids.shape: {centroids.shape}")

        # Initialize the result list with empty lists for each class
        result = [[]] * len(centroids)

        # Check if the predictions based on spikes and probabilities are the same
        predictions_1 = Utils.prediction_spikes(net_outputs) 
        predictions_2 = Utils.prediction_probs(prob)

        compare_predictions = np.array_equal(predictions_1, predictions_2)
        print(f"Are predictions the same: {compare_predictions}")

        for i, spike in enumerate(spikes):
            # Get the predicted class for the current spikes
            pred = predictions_1[i]
            cent = np.array(centroids[pred])

            # Compute minimum distance for each test sample for corresponding class
            # Euclidan distance
            min_dist = -np.min(np.sqrt(np.sum((cent - spike) ** 2, axis=1)))
            # Manhattan distance
            # min_dist = np.min(np.sum(np.abs(cent - spike), axis=1))
            result[pred].append(min_dist)
    
        return [np.array(x) for x in result]
    
    @staticmethod
    def distance_from_class_centroids(spikes, centroids):
        '''
        Function that computes the distances for each feature vector (spike pattern) from 
        all class centroids.

        Used in NCM.

        Inputs:
            spikes (np.array): tensor with the spike patterns (batch_size, feature_size)
            centroids (list): list of cluster centroids

        Outputs:
            result (list): list of distances for each class
        '''

        # Compute the squared Euclidean distances between spikes and centroids
        # Using broadcasting to calculate distances in a vectorized manner
        diff = spikes[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # Shape: (batch_size, num_classes, feature_size)
        # dist_squared = np.sum(diff**2, axis=2)  # Shape: (batch_size, num_classes)

        # # Find the minimum distance for each spike and negate it
        # distances = -np.sqrt(np.min(dist_squared, axis=1))  # Shape: (batch_size,)

        distances = -np.min(np.linalg.norm(diff, axis=2), axis=1)      # (batch_size,)

        return distances

    # ### Explanation of Optimizations:
    # 1. **Broadcasting**:
    #    - Instead of iterating over each spike and centroid, we use NumPy's broadcasting to compute the pairwise differences between all spikes and centroids in one step.
    #    - `spikes[:, np.newaxis, :]` expands the `spikes` array to shape `(batch_size, 1, feature_size)`.
    #    - `centroids[np.newaxis, :, :]` expands the `centroids` array to shape `(1, num_classes, feature_size)`.
    #    - The subtraction results in a shape of `(batch_size, num_classes, feature_size)`.

    # 2. **Vectorized Squared Sum**:
    #    - The squared Euclidean distance is computed using `np.sum(diff**2, axis=2)` for all spikes and centroids simultaneously, avoiding the inner loop.

    # 3. **Efficient Minimum Calculation**:
    #    - The minimum distance for each spike is computed using `np.min(dist_squared, axis=1)`.

    # 4. **Avoid Explicit Loops**:
    #    - By replacing the nested loops with vectorized operations, the function becomes significantly faster, especially for large datasets.

    # ### Performance Improvement:
    # - The optimized function eliminates the `O(batch_size * num_classes)` complexity of the nested loops and replaces it with efficient NumPy operations that leverage low-level optimizations.
    # - This will result in a significant speedup, especially when `batch_size` or `num_classes` is large.
    
    @staticmethod
    def id_ood_predictions(distances, threshold):
        '''
        Make In-distribution / Out-of-distribution predictions based on the distances and threshold.

        Inputs:
            distances (list): list of distances for each class
            threshold (float): threshold value for classification

        Outputs:
            in_or_out_distribution (np.array): array with True if InD and False if OOD
        '''

        predictions = np.zeros(len(distances), dtype=np.int32)
        for i in range(len(distances)):    
            predictions[i] = 1 if distances[i] > threshold else 0

        return predictions


class SVHNDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)  
        self.labels = torch.tensor(labels, dtype=torch.long)  
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Apply transformation if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_preds_from_spike_vector(spikes):
    return spikes.argmax(axis=1)

def get_preds_from_probs_vector(probs):
    return probs.argmax(axis=1)

def distances_from_average_clusters(spikes, net_outputs, clusters, prob):
    result = [[] for i in range(len(clusters))]

    preds = get_preds_from_spike_vector(net_outputs) 
    pred2 = get_preds_from_probs_vector(prob) 

    for i, spike in enumerate(spikes):
        pred = preds[i]
        cent = np.array(clusters[pred])

        # Compute minimum distance for each test semple for corresponding class
        min_dist = np.min(np.sum(np.abs(cent - spike), axis=1))
        result[pred].append(min_dist)
    
    return [np.array(x) for x in result]

# Funkcija kopirana iz rada
def thresholds_per_class_for_each_TPR(dist_per_class):
    num_classes = len(dist_per_class)
    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    sorted_distances_per_class = [np.sort(x) for x in dist_per_class]
    tpr_range = np.arange(0,1,0.01)
    tpr_range[-1] = 0.99999999 # For selecting the last item correctly
    distance_thresholds_test = np.zeros((num_classes, len(tpr_range)))
    for class_index in range(num_classes):
        for index, tpr in enumerate(tpr_range):
            distance_thresholds_test[class_index, index] = sorted_distances_per_class[class_index][int(len(sorted_distances_per_class[class_index])*tpr)]
    
    return distance_thresholds_test


def compute_thresholds(dists):
    return thresholds_per_class_for_each_TPR(dists)


def compare_distances_per_class_to_distance_thr_per_class(distances_list_per_class, thr_distances_array):
    '''
    Function that creates an array of shape (tpr, InD_or_OD), where tpr has the lenght of the number of steps of the TPR list
    and second dimensions has the total lenght of the distances_list_per_class, and cotains True if its InD and False if is OD
    :distances_list_per_class: list with each element being an array with the distances to avg clusters of one class [array(.), array(.)]
    :thr_distances_array: array of shape (class, dist_for_each_tpr), where first dimension is the class and the second is the distance for the TPR
    corresponding to that position. For example, the TPR = 0.85 corresponds to the 85th position.
    '''
    in_or_out_distribution_per_tpr = np.zeros((len(np.transpose(thr_distances_array)),len(np.concatenate(distances_list_per_class))),dtype=bool)
    for tpr_index ,thr_distances_per_class in enumerate(np.transpose(thr_distances_array)):
        in_or_out_distribution_per_tpr[tpr_index] = np.concatenate([dist_one_class < thr_distances_per_class[cls_index] for cls_index, dist_one_class in enumerate(distances_list_per_class)])
    
    return in_or_out_distribution_per_tpr


def compute_precision_tpr_fpr_for_test_and_ood(dist_test_per_class, dist_ood_per_class,dist_thresholds):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_distances_per_class_to_distance_thr_per_class(dist_test_per_class, dist_thresholds)
    in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),dtype=bool) # To fix that one element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),dtype=bool) # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_ood = compare_distances_per_class_to_distance_thr_per_class(dist_ood_per_class, dist_thresholds)

    # Creation of arrays with TP, FN and FP, TN
    tp_fn_test = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_test)
    fp_tn_ood = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_ood)

    # Computing TPR, FPR and Precision
    tpr_values = tp_fn_test[:,0] / (tp_fn_test[:,0] + tp_fn_test[:,1])
    fpr_values = fp_tn_ood[:,0] / (fp_tn_ood[:,0] + fp_tn_ood[:,1])
    precision  = tp_fn_test[:,0] / (tp_fn_test[:,0] + fp_tn_ood[:,0])

    # Eliminating NaN value at TPR = 1
    precision[0] = 1
    return precision, tpr_values, fpr_values


def tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr):
    '''
    Function that creates an array with the number of values of tp and fp or fn and tn, depending on if the 
    passed array is InD or OD.
    :in_or_out_distribution_per_tpr: array with True if predicted InD and False if predicted OD, for each TPR
    ::return: array with shape (tpr, 2) with the 2 dimensions being tp,fn if passed array is InD, and fp and tn if the passed array is OD
    '''
    tp_fn_fp_tn = np.zeros((len(in_or_out_distribution_per_tpr),2),dtype='uint16')
    length_array = in_or_out_distribution_per_tpr.shape[1]
    for index, element in enumerate(in_or_out_distribution_per_tpr):
        n_True = int(len(element.nonzero()[0]))
        tp_fn_fp_tn[index,0] = n_True
        tp_fn_fp_tn[index,1] = length_array - n_True
    return tp_fn_fp_tn
